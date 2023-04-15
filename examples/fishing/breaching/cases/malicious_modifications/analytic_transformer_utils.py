import torch
from statistics import NormalDist
import logging

log = logging.getLogger(__name__)


@torch.inference_mode()
def compute_feature_distribution(model, target_layer, measurement, server):
    """Compute the mean and std of the feature layer of the given network."""
    features = dict()

    def named_hook(name):
        def hook_fn(module, input, output):
            features[name] = input[0]
            raise RuntimeError("Early exit")

        return hook_fn

    hook = target_layer.register_forward_hook(named_hook("linear_probe"))

    feats = []
    model.train()
    model.to(**server.setup)

    if server.external_dataloader is not None:
        log.info(f"Computing feature distribution before the probe layer {target_layer} from external data.")
        for i, batch in enumerate(server.external_dataloader):
            inputs = batch["input_ids"].to(device=server.setup["device"])
            try:
                model(inputs)
            except RuntimeError:
                pass
                # This is likely the worst-possible way to break execution after  the hook ...
            feats.append(features["linear_probe"].detach().view(inputs.shape[0] * inputs.shape[1], -1).clone())
    else:
        log.info(f"Computing feature distribution before the probe layer {target_layer} from random tokens.")
        cfg = server.cfg_data
        # weights = torch.ones((cfg.vocab_size))  # Uniform vocabulary
        # weights = 1 / torch.arange(1, cfg.vocab_size + 1)  # Zipfy enough?
        for i in range(100):
            inputs = torch.randint(0, cfg.vocab_size, (cfg.batch_size, *cfg.shape), device=server.setup["device"])
            # sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=cfg.batch_size * cfg.shape[0])
            # samples = list(iter(sampler))
            # inputs = torch.as_tensor(samples, device=server.setup["device"]).view((cfg.batch_size, *cfg.shape))
            try:
                model(inputs)
            except RuntimeError:
                pass
            feats.append(features["linear_probe"].detach().view(inputs.shape[0] * inputs.shape[1], -1).clone())

    std, mu = torch.std_mean(torch.matmul(torch.cat(feats), measurement))
    model.eval()
    model.cpu()
    hook.remove()
    log.info(f"Feature mean is {mu.item()}, feature std is {std.item()}.")
    return std, mu


def partially_disable_embedding(embedding_layer, v_length):
    """Disable the first v_proportion rows of all embeddings."""
    embedding_layer.weight.data[:, :v_length] = 0


def partially_norm_position(embedding_layer, v_length):
    for i in range(embedding_layer.weight.shape[0]):
        embedding_layer.weight[i].data /= torch.norm(embedding_layer.weight[i][v_length : v_length * 2])


def set_MHA(
    attention_layer,
    norm_layer0,
    pos_encoder,
    embedding_dim,
    ff_transposed,
    data_shape,
    sequence_token_weight=1,
    imprint_sentence_position=0,  # This position will be imprinted onto the sentence via attention
    softmax_skew=1000000,
    v_length=6,
):
    # Q,K,V matrices stored as single array (default) or as three separate arrays (bert) as in huggingface bert
    if attention_layer["mode"] == "default":
        _set_default_MHA(
            attention_layer,
            norm_layer0,
            pos_encoder,
            embedding_dim,
            ff_transposed,
            data_shape,
            sequence_token_weight,
            imprint_sentence_position,  # This position will be imprinted onto the sentence via attention
            softmax_skew,
            v_length,
        )
    elif attention_layer["mode"] == "bert":
        _set_bert_MHA(
            attention_layer,
            norm_layer0,
            pos_encoder,
            embedding_dim,
            ff_transposed,
            data_shape,
            sequence_token_weight,
            imprint_sentence_position,  # This position will be imprinted onto the sentence via attention
            softmax_skew,
            v_length,
        )
    else:
        raise ValueError(f"Invalid MHA mode {attention_layer['mode']} given.")


def _set_bert_MHA(
    attention_layer,
    norm_layer0,
    pos_encoder,
    embedding_dim,
    ff_transposed,
    data_shape,
    sequence_token_weight=1,
    imprint_sentence_position=0,  # This position will be imprinted onto the sentence via attention
    softmax_skew=1000000,
    v_length=6,
):
    # Let's set the query matrix to produce just the first positional encoding (or could be any index - might want last index)
    if ff_transposed:
        qkv_shape = attention_layer["query"].weight.data.shape[1]
        log.info(f"Found attention of shape {attention_layer['query'].weight.T.shape}.")
    else:
        qkv_shape = attention_layer["query"].weight.data.shape[0]
        log.info(f"Found attention of shape {attention_layer['query'].weight.data.shape}.")

    # These are the positional embeddings after layer normalization:
    dummy_data = torch.zeros([1, *data_shape, embedding_dim])
    just_positions = norm_layer0(pos_encoder(dummy_data)).cpu()
    # Q matrix setup
    # We make the weight 0, and the bias some (large multiple of) positional encoding
    # Only coded here for one MHA layer at the beginning of the model...
    # Make the position super super large to skew softmax
    attention_layer["query"].bias.data.zero_()
    attention_layer["key"].bias.data.zero_()
    attention_layer["value"].bias.data.zero_()
    position_comp = just_positions[0, imprint_sentence_position, :][v_length : 2 * v_length]
    attention_layer["query"].bias.data[v_length : 2 * v_length] = softmax_skew * position_comp

    attention_layer["query"].weight.data = torch.zeros((qkv_shape, qkv_shape))
    # Set V_bias to subtract positional encoding
    v_bias = torch.zeros(qkv_shape)
    v_bias[imprint_sentence_position : (imprint_sentence_position + v_length)] = -just_positions[
        0, imprint_sentence_position, v_length : (2 * v_length)
    ]
    attention_layer["value"].bias.data = v_bias

    # K matrix setup (identity)
    if ff_transposed:
        attention_layer["key"].weight.data = torch.eye(qkv_shape)
    else:
        attention_layer["key"].weight.data = torch.eye(qkv_shape)

    # V matrix setup (truncated shifted identity block)
    v_data = torch.zeros((qkv_shape, qkv_shape))
    v_data[:v_length, v_length : (2 * v_length)] = torch.eye(v_length)

    if ff_transposed:
        attention_layer["value"].weight.data = v_data.T.contiguous()
    else:
        attention_layer["value"].weight.data = v_data
    # So, (QK^T)V just adds the same vector (first word embedding) to each word in the sequence.

    # Linear layer at the end of MHA - optionally can be set to small value to not 'skew' embeddings too much
    attention_layer["output"].weight.data = sequence_token_weight * torch.eye(qkv_shape)
    attention_layer["output"].bias.data.zero_()


def _set_default_MHA(
    attention_layer,
    norm_layer0,
    pos_encoder,
    embedding_dim,
    ff_transposed,
    data_shape,
    sequence_token_weight=1,
    imprint_sentence_position=0,  # This position will be imprinted onto the sentence via attention
    softmax_skew=1000000,
    v_length=6,
):
    # Let's set the query matrix to produce just the first positional encoding (or could be any index - might want last index)
    if ff_transposed:
        qkv_shape = attention_layer["in_proj_weight"].data.shape[1]
        log.info(f"Found attention of shape {attention_layer['in_proj_weight'].T.shape}.")
    else:
        qkv_shape = attention_layer["in_proj_weight"].data.shape[0]
        log.info(f"Found attention of shape {attention_layer['in_proj_weight'].data.shape}.")

    # These are the positional embeddings after layer normalization:
    dummy_data = torch.zeros([1, *data_shape, embedding_dim])
    just_positions = norm_layer0(pos_encoder(dummy_data)).cpu()
    # Q matrix setup
    # We make the weight 0, and the bias some (large multiple of) positional encoding
    # Only coded here for one MHA layer at the beginning of the model...
    # Make the position super super large to skew softmax
    attention_layer["in_proj_bias"].data.zero_()
    position_comp = just_positions[0, imprint_sentence_position, :][v_length : 2 * v_length]
    attention_layer["in_proj_bias"].data[: qkv_shape // 3][v_length : 2 * v_length] = softmax_skew * position_comp

    if ff_transposed:
        attention_layer["in_proj_weight"].data[:, : qkv_shape // 3] = torch.zeros((qkv_shape // 3, qkv_shape // 3))
    else:
        attention_layer["in_proj_weight"].data[: qkv_shape // 3] = torch.zeros((qkv_shape // 3, qkv_shape // 3))

    # Set V_bias to subtract positional encoding
    v_bias = torch.zeros(qkv_shape // 3)
    v_bias[imprint_sentence_position : (imprint_sentence_position + v_length)] = -just_positions[
        0, imprint_sentence_position, v_length : (2 * v_length)
    ]
    attention_layer["in_proj_bias"].data[2 * (qkv_shape // 3) :] = v_bias

    # K matrix setup (identity)
    if ff_transposed:
        attention_layer["in_proj_weight"].data[:, qkv_shape // 3 : 2 * (qkv_shape // 3)] = torch.eye(qkv_shape // 3)
    else:
        attention_layer["in_proj_weight"].data[qkv_shape // 3 : 2 * (qkv_shape // 3)] = torch.eye(qkv_shape // 3)

    # V matrix setup (truncated shifted identity block)
    v_data = torch.zeros((qkv_shape // 3, qkv_shape // 3))
    v_data[:v_length, v_length : (2 * v_length)] = torch.eye(v_length)

    if ff_transposed:
        attention_layer["in_proj_weight"].data[:, 2 * (qkv_shape // 3) :] = v_data.T.contiguous()
    else:
        attention_layer["in_proj_weight"].data[2 * (qkv_shape // 3) :] = v_data
    # So, (QK^T)V just adds the same vector (first word embedding) to each word in the sequence.

    # Linear layer at the end of MHA - optionally can be set to small value to not 'skew' embeddings too much
    attention_layer["out_proj_weight"].data = sequence_token_weight * torch.eye(qkv_shape // 3)
    attention_layer["out_proj_bias"].data.zero_()


def set_flow_backward_layer(second_layers, ff_transposed=False, eps=1e-4):
    """
    here we set the second linear layer in the ff block to accumulate everything
    from the first linear layer into one entry, thus allowing gradients to flow
    backward, but not 'shifting' the embeddings.
    """

    for layer in second_layers:
        layer.weight.data.zero_()
        if ff_transposed:
            layer.weight.data[:, -1] = eps / layer.weight.data.shape[0]
        else:
            layer.weight.data[-1] = eps / layer.weight.data.shape[1]
        layer.bias.data.zero_()


def disable_mha_layers(layers):
    """
    Here we set all MHA out_proj_weights to 0 except for the first one
    where we encode the sequence
    """

    for layer_out_proj in layers:
        layer_out_proj.weight.data.zero_()
        layer_out_proj.bias.data.zero_()


def equalize_mha_layer(
    attention_layer,
    ff_transposed,
    equalize_token_weight=0.001,
    v_length=6,
):

    if attention_layer["mode"] == "default":
        if ff_transposed:
            qkv_shape = attention_layer["in_proj_weight"].data.shape[1]
            log.info(f"Found attention of shape {attention_layer['in_proj_weight'].T.shape}.")
        else:
            qkv_shape = attention_layer["in_proj_weight"].data.shape[0]
        attention_layer["in_proj_bias"].data.zero_()
        # Q to zero:
        attention_layer["in_proj_weight"].data.zero_()
        # K matrix setup (identity)
        if ff_transposed:
            attention_layer["in_proj_weight"].data[:, qkv_shape // 3 : 2 * (qkv_shape // 3)] = torch.eye(qkv_shape // 3)
        else:
            attention_layer["in_proj_weight"].data[qkv_shape // 3 : 2 * (qkv_shape // 3)] = torch.eye(qkv_shape // 3)
        # V matrix setup ( identity block)
        if ff_transposed:
            attention_layer["in_proj_weight"].data[:, 2 * (qkv_shape // 3) :] = torch.eye(qkv_shape // 3)
        else:
            attention_layer["in_proj_weight"].data[2 * (qkv_shape // 3) :] = torch.eye(qkv_shape // 3)

        attention_layer["out_proj_weight"].data = equalize_token_weight * torch.eye(qkv_shape // 3)
        attention_layer["out_proj_bias"].data.zero_()
    else:
        if ff_transposed:
            qkv_shape = attention_layer["query"].weight.data.shape[1]
            log.info(f"Found attention of shape {attention_layer['query'].weight.T.shape}.")
        else:
            qkv_shape = attention_layer["query"].weight.data.shape[0]
            log.info(f"Found attention of shape {attention_layer['query'].weight.data.shape}.")
        attention_layer["query"].weight.data = torch.zeros((qkv_shape, qkv_shape))
        attention_layer["query"].bias.data.zero_()
        # K matrix setup (identity)
        attention_layer["key"].weight.data = torch.eye(qkv_shape)
        attention_layer["key"].bias.data.zero_()
        # Set V to identity
        attention_layer["value"].weight.data = torch.eye(qkv_shape)
        attention_layer["value"].bias.data.zero_()

        # Linear layer at the end of MHA - optionally can be set to small value to not 'skew' embeddings too much
        attention_layer["output"].weight.data = equalize_token_weight * torch.eye(qkv_shape)
        attention_layer["output"].bias.data.zero_()


def make_imprint_layer(first_layers, measurement, mean, std, hidden_dim, embedding_dim, ff_transposed=False):
    """
    measurement is the Gaussian vector we take inner product w.r.t.
    mean, std = mean, std of features from feature_distribution
    """

    def _get_bins(mean, std, num_bins):
        bins = []
        mass_per_bin = 1 / (num_bins)
        bins.append(-10)  # -Inf is not great here, but NormalDist(mu=0, sigma=1).cdf(10) approx 1
        for i in range(1, num_bins):
            bins.append(NormalDist().inv_cdf(i * mass_per_bin) * std + mean)
        return bins

    def _make_biases(bias_layer, bins):
        new_biases = torch.zeros_like(bias_layer.data)
        for i in range(new_biases.shape[0]):
            new_biases[i] = -bins[i]
        return new_biases

    bins = _get_bins(mean, std, hidden_dim * len(first_layers))
    bins_per_layer = len(bins) // len(first_layers)

    for i, layer in enumerate(first_layers):
        if ff_transposed:
            layer.weight.data = measurement.repeat(hidden_dim, 1).T
        else:
            layer.weight.data = measurement.repeat(hidden_dim, 1)
        layer.bias.data = _make_biases(layer.bias, bins[(i * bins_per_layer) : ((i + 1) * bins_per_layer)])
