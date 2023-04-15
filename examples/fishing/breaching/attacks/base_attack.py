"""Implementation for base attacker class.

Inherit from this class for a consistent interface with attack cases."""

import torch
from collections import defaultdict
import copy

from .auxiliaries.common import optimizer_lookup
from ..cases.models.transformer_dictionary import lookup_grad_indices

import logging

log = logging.getLogger(__name__)
embedding_layer_names = ["encoder.weight", "word_embeddings.weight", "transformer.wte"]


class _BaseAttacker:
    """This is a template class for an attack.

    A basic assumption for this attacker is that user data is fixed over multiple queries.
    """

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        self.cfg = cfg_attack
        self.memory_format = torch.channels_last if cfg_attack.impl.mixed_precision else torch.contiguous_format
        self.setup = dict(device=setup["device"], dtype=getattr(torch, cfg_attack.impl.dtype))
        self.model_template = copy.deepcopy(model)
        self.loss_fn = copy.deepcopy(loss_fn)

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """Overwrite this function to implement a new attack."""
        # Implement the attack here
        # The attack should consume the shared_data and server payloads and reconstruct into a dict
        # with key data, labels
        raise NotImplementedError()

        return reconstructed_data, stats

    def __repr__(self):
        raise NotImplementedError()

    def prepare_attack(self, server_payload, shared_data):
        """Basic startup common to many reconstruction methods."""
        stats = defaultdict(list)

        shared_data = shared_data.copy()  # Shallow copy is enough
        server_payload = server_payload.copy()

        # Load preprocessing constants:
        metadata = server_payload[0]["metadata"]
        self.data_shape = metadata.shape
        if hasattr(metadata, "mean"):
            self.dm = torch.as_tensor(metadata.mean, **self.setup)[None, :, None, None]
            self.ds = torch.as_tensor(metadata.std, **self.setup)[None, :, None, None]
        else:
            self.dm, self.ds = torch.tensor(0, **self.setup), torch.tensor(1, **self.setup)

        # Load server_payload into state:
        rec_models = self._construct_models_from_payload_and_buffers(server_payload, shared_data)
        shared_data = self._cast_shared_data(shared_data)
        if metadata.modality == "text":
            rec_models, shared_data = self._prepare_for_text_data(shared_data, rec_models)
        self._rec_models = rec_models
        # Consider label information
        if shared_data[0]["metadata"]["labels"] is None:
            labels = self._recover_label_information(shared_data, server_payload, rec_models)
        else:
            labels = shared_data[0]["metadata"]["labels"].clone()

        # Condition gradients?
        if self.cfg.normalize_gradients:
            shared_data = self._normalize_gradients(shared_data)
        return rec_models, labels, stats

    def _prepare_for_text_data(self, shared_data, rec_models):
        """Reconstruct the output of the Embedding Layer?"""
        # _circumvent_embedding_layer
        if "run-embedding" == self.cfg.text_strategy:
            # 1) Basic trick: Optimize in embedding space
            # Cut off embeddings:
            self.embeddings = []
            for model, data in zip(rec_models, shared_data):
                name_to_idx = dict(zip([n for n, _ in model.named_parameters()], range(len(data["gradients"]))))

                for name in embedding_layer_names:
                    for key in name_to_idx.keys():
                        if name in key:
                            embedding_position = name_to_idx[key]  # todo: generalize this in a robuster way

                self.embeddings.append(
                    dict(
                        weight=list(model.parameters())[embedding_position],
                        grads=data["gradients"].pop(embedding_position),
                    )
                )
                # Recur through model to find the matching module and disable it:

                def replace(model):
                    for child_name, child in model.named_children():
                        if isinstance(child, torch.nn.Embedding):
                            if child.weight is self.embeddings[-1]["weight"]:
                                setattr(model, child_name, torch.nn.Identity())
                        else:
                            replace(child)

                replace(model)

            # Adjust data shape
            _, token_embedding_dim = self.embeddings[0]["weight"].shape
            self.data_shape = [*self.data_shape, token_embedding_dim]
        elif self.cfg.text_strategy == "no-preprocessing":
            pass
        else:
            raise ValueError(f"Invalid text strategy {self.cfg.text_strategy} given.")
        # To try later:
        # Assuming sequence_length is known, and all tokens are leaked from the Embedding Layer
        # We should find the input by optimizing a "segmentation map" of these tokens
        # This can be relaxed to [0,1] constraints and solved with convex programming tricks
        # The Relaxation can be constructed over the Subset of tokens with non-zero gradients
        # Basic model (as also seen in DLG): Optimize in embedding space
        return rec_models, shared_data

    def _postprocess_text_data(self, reconstructed_user_data, models=None):
        """Post-process text data to recover tokens."""

        def _max_similarity(recovered_embeddings, true_embeddings):
            recovered_embeddings = recovered_embeddings - recovered_embeddings.mean(dim=-1, keepdim=True)
            true_embeddings = true_embeddings - true_embeddings.mean(dim=-1, keepdim=True)
            norm_rec = recovered_embeddings.pow(2).sum(dim=-1)
            norm_true = true_embeddings.pow(2).sum(dim=-1)
            cosim = recovered_embeddings.matmul(true_embeddings.T) / norm_rec[:, None] / norm_true[None, :]
            return cosim.argmax(dim=1)

        if hasattr(self, "embeddings"):
            # Use extracted embeddings:
            embedding_weight = self.embeddings[0]["weight"]
        else:
            # or lazily import lookup table
            from ..cases.models.transformer_dictionary import lookup_module_names

            embedding_weight = lookup_module_names(models[0].name, models[0])["embedding"].weight

        if self.cfg.token_recovery == "from-embedding":
            # This is the DLG strategy. Look up all inputs in embedding space.
            recovered_embeddings = reconstructed_user_data["data"]
            base_shape = recovered_embeddings.shape[0:2]
            recovered_embeddings = recovered_embeddings.view(-1, recovered_embeddings.shape[-1])
            true_embeddings = embedding_weight

            recovered_tokens = _max_similarity(recovered_embeddings, true_embeddings).view(*base_shape)

        elif self.cfg.token_recovery == "from-labels":
            # Only works well in some causal-lm?
            recovered_tokens = reconstructed_user_data["labels"]
        elif self.cfg.token_recovery == "from-limited-embedding":
            # Retrieve possible embeddings from gradient data
            recovered_embeddings = reconstructed_user_data["data"]
            base_shape = recovered_embeddings.shape[0:2]
            recovered_embeddings = recovered_embeddings.view(-1, recovered_embeddings.shape[-1])
            active_embedding_ids = reconstructed_user_data["labels"].unique()
            true_embeddings = embedding_weight[active_embedding_ids, :]
            matches = _max_similarity(recovered_embeddings, true_embeddings)
            recovered_tokens = active_embedding_ids[matches].view(*base_shape)

        reconstructed_user_data["data"] = recovered_tokens
        return reconstructed_user_data

    def _construct_models_from_payload_and_buffers(self, server_payload, shared_data):
        """Construct the model (or multiple) that is sent by the server and include user buffers if any."""

        # Load states into multiple models if necessary
        models = []
        for idx, payload in enumerate(server_payload):

            new_model = copy.deepcopy(self.model_template)
            new_model.to(**self.setup, memory_format=self.memory_format)

            # Load parameters
            parameters = payload["parameters"]
            if shared_data[idx]["buffers"] is not None:
                # User sends buffers. These should be used!
                buffers = shared_data[idx]["buffers"]
                new_model.eval()
            elif payload["buffers"] is not None:
                # The server has public buffers in any case
                buffers = payload["buffers"]
                new_model.eval()
            else:
                # The user sends no buffers and there are no public bufers
                # (i.e. the user in in training mode and does not send updates)
                new_model.train()
                for module in new_model.modules():
                    if hasattr(module, "track_running_stats"):
                        module.reset_parameters()
                        module.track_running_stats = False
                buffers = []

            with torch.no_grad():
                for param, server_state in zip(new_model.parameters(), parameters):
                    param.copy_(server_state.to(**self.setup))
                for buffer, server_state in zip(new_model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))

            if self.cfg.impl.JIT == "script":
                example_inputs = self._initialize_data((1, *self.data_shape))
                new_model = torch.jit.script(new_model, example_inputs=[(example_inputs,)])
            elif self.cfg.impl.JIT == "trace":
                example_inputs = self._initialize_data((1, *self.data_shape))
                new_model = torch.jit.trace(new_model, example_inputs=example_inputs)
            models.append(new_model)
        return models

    def _cast_shared_data(self, shared_data):
        """Cast user data to reconstruction data type."""
        for data in shared_data:
            data["gradients"] = [g.to(dtype=self.setup["dtype"]) for g in data["gradients"]]
            if data["buffers"] is not None:
                data["buffers"] = [b.to(dtype=self.setup["dtype"]) for b in data["buffers"]]
        return shared_data

    def _initialize_data(self, data_shape):
        """Note that data is initialized "inside" the network normalization."""
        init_type = self.cfg.init
        if init_type == "randn":
            candidate = torch.randn(data_shape, **self.setup)
        elif init_type == "randn-trunc":
            candidate = (torch.randn(data_shape, **self.setup) * 0.1).clamp(-0.1, 0.1)
        elif init_type == "rand":
            candidate = (torch.rand(data_shape, **self.setup) * 2) - 1.0
        elif init_type == "zeros":
            candidate = torch.zeros(data_shape, **self.setup)
        # Initializations from Wei et al, "A Framework for Evaluating Gradient Leakage
        #                                  Attacks in Federated Learning"
        elif any(c in init_type for c in ["red", "green", "blue", "dark", "light"]):  # init_types like 'red-true'
            candidate = torch.zeros(data_shape, **self.setup)
            if "light" in init_type:
                candidate = torch.ones(data_shape, **self.setup)
            else:
                nonzero_channel = 0 if "red" in init_type else 1 if "green" in init_type else 2
                candidate[:, nonzero_channel, :, :] = 1
            if "-true" in init_type:
                # Shift to be truly RGB, not just normalized RGB
                candidate = (candidate - self.dm) / self.ds
        elif "patterned" in init_type:  # Look for init_type=rand-patterned-4
            pattern_width = int("".join(filter(str.isdigit, init_type)))
            if "randn" in init_type:
                seed = torch.randn([data_shape[0], 3, pattern_width, pattern_width], **self.setup)
            elif "rand" in init_type:
                seed = (torch.rand([data_shape[0], 3, pattern_width, pattern_width], **self.setup) * 2) - 1
            else:  # default is also randn
                seed = torch.randn([data_shape[0], 3, pattern_width, pattern_width], **self.setup)
            # Shape expansion:
            x_factor, y_factor = (
                torch.as_tensor(data_shape[2] / pattern_width).ceil(),
                torch.as_tensor(data_shape[3] / pattern_width).ceil(),
            )
            candidate = (
                torch.tile(seed, (1, 1, int(x_factor), int(y_factor)))[:, :, : data_shape[2], : data_shape[3]]
                .contiguous()
                .clone()
            )
        elif "wei" in init_type:  # Look for init_type=rand-wei-4
            pattern_width = int("".join(filter(str.isdigit, init_type)))
            if "rand" in init_type:
                seed = (torch.rand([data_shape[0], 3, pattern_width, pattern_width], **self.setup) * 2) - 1
            else:
                seed = torch.randn([data_shape[0], 3, pattern_width, pattern_width], **self.setup)
            # Shape expansion:
            x_factor, y_factor = (
                torch.as_tensor(data_shape[2] / pattern_width).ceil(),
                torch.as_tensor(data_shape[3] / pattern_width).ceil(),
            )
            candidate = (
                torch.tile(seed, (1, 1, int(x_factor), int(y_factor)))[:, :, : data_shape[2], : data_shape[3]]
                .contiguous()
                .clone()
            )
        else:
            raise ValueError(f"Unknown initialization scheme {init_type} given.")

        candidate.to(memory_format=self.memory_format)
        candidate.requires_grad = True
        candidate.grad = torch.zeros_like(candidate)
        return candidate

    def _init_optimizer(self, candidate):
        optimizer, scheduler = optimizer_lookup(
            candidate,
            self.cfg.optim.optimizer,
            self.cfg.optim.step_size,
            scheduler=self.cfg.optim.step_size_decay,
            warmup=self.cfg.optim.warmup,
            max_iterations=self.cfg.optim.max_iterations,
        )
        return optimizer, scheduler

    def _normalize_gradients(self, shared_data, fudge_factor=1e-6):
        """Normalize gradients to have norm of 1. No guarantees that this would be a good idea for FL updates."""
        for data in shared_data:
            grad_norm = torch.stack([g.pow(2).sum() for g in data["gradients"]]).sum().sqrt()
            torch._foreach_div_(data["gradients"], max(grad_norm, fudge_factor))
        return shared_data

    def _recover_label_information(self, user_data, server_payload, rec_models):
        """Recover label information.

        This method runs under the assumption that the last two entries in the gradient vector
        correpond to the weight and bias of the last layer (mapping to num_classes).
        For non-classification tasks this has to be modified.

        The behavior with respect to multiple queries is work in progress and subject of debate.
        """
        num_data_points = user_data[0]["metadata"]["num_data_points"]
        num_classes = user_data[0]["gradients"][-1].shape[0]
        num_queries = len(user_data)

        if self.cfg.label_strategy is None:
            return None
        elif self.cfg.label_strategy == "iDLG":
            # In the simplest case, the label can just be inferred from the last layer
            # This was popularized in "iDLG" by Zhao et al., 2020
            # assert num_data_points == 1
            label_list = []
            for query_id, shared_data in enumerate(user_data):
                last_weight_min = torch.argmin(torch.sum(shared_data["gradients"][-2], dim=-1), dim=-1)
                label_list += [last_weight_min.detach()]
            labels = torch.stack(label_list).unique()
        elif self.cfg.label_strategy == "analytic":
            # Analytic recovery simply works as long as all labels are unique.
            label_list = []
            for query_id, shared_data in enumerate(user_data):
                valid_classes = (shared_data["gradients"][-1] < 0).nonzero()
                label_list += [valid_classes]
            labels = torch.stack(label_list).unique()[:num_data_points]
        elif self.cfg.label_strategy == "yin":
            # As seen in Yin et al. 2021, "See Through Gradients: Image Batch Recovery via GradInversion"
            # This additionally assumes that there is a nonlinearity with positive output (like ReLU) in front of the
            # last classification layer.
            # This scheme also works best if all labels are unique
            # Otherwise this is an extension of iDLG to multiple labels:
            total_min_vals = 0
            for query_id, shared_data in enumerate(user_data):
                total_min_vals += shared_data["gradients"][-2].min(dim=-1)[0]
            labels = total_min_vals.argsort()[:num_data_points]

        elif "wainakh" in self.cfg.label_strategy:

            if self.cfg.label_strategy == "wainakh-simple":
                # As seen in Weinakh et al., "User Label Leakage from Gradients in Federated Learning"
                m_impact = 0
                for query_id, shared_data in enumerate(user_data):
                    g_i = shared_data["gradients"][-2].sum(dim=1)
                    m_query = (
                        torch.where(g_i < 0, g_i, torch.zeros_like(g_i)).sum() * (1 + 1 / num_classes) / num_data_points
                    )
                    s_offset = 0
                    m_impact += m_query / num_queries
            elif self.cfg.label_strategy == "wainakh-whitebox":
                # Augment previous strategy with measurements of label impact for dummy data.
                m_impact = 0
                s_offset = torch.zeros(num_classes, **self.setup)

                print("Starting a white-box search for optimal labels. This will take some time.")
                for query_id, model in enumerate(rec_models):
                    # Estimate m:
                    weight_params = (list(rec_models[0].parameters())[-2],)
                    for class_idx in range(num_classes):
                        fake_data = torch.randn([num_data_points, *self.data_shape], **self.setup)
                        fake_labels = torch.as_tensor([class_idx] * num_data_points, **self.setup)
                        with torch.autocast(self.setup["device"].type, enabled=self.cfg.impl.mixed_precision):
                            loss = self.loss_fn(model(fake_data), fake_labels)
                        (W_cls,) = torch.autograd.grad(loss, weight_params)
                        g_i = W_cls.sum(dim=1)
                        m_impact += g_i.sum() * (1 + 1 / num_classes) / num_data_points / num_classes / num_queries

                    # Estimate s:
                    T = num_classes - 1
                    for class_idx in range(num_classes):
                        fake_data = torch.randn([T, *self.data_shape], **self.setup)
                        fake_labels = torch.arange(num_classes, **self.setup)
                        fake_labels = fake_labels[fake_labels != class_idx]
                        with torch.autocast(self.setup["device"].type, enabled=self.cfg.impl.mixed_precision):
                            loss = self.loss_fn(model(fake_data), fake_labels)
                        (W_cls,) = torch.autograd.grad(loss, (weight_params[0][class_idx],))
                        s_offset[class_idx] += W_cls.sum() / T / num_queries

            else:
                raise ValueError(f"Invalid Wainakh strategy {self.cfg.label_strategy}.")

            # After determining impact and offset, run the actual recovery algorithm
            label_list = []
            g_per_query = [shared_data["gradients"][-2].sum(dim=1) for shared_data in user_data]
            g_i = torch.stack(g_per_query).mean(dim=0)
            # Stage 1:
            for idx in range(num_classes):
                if g_i[idx] < 0:
                    label_list.append(torch.as_tensor(idx, device=self.setup["device"]))
                    g_i[idx] -= m_impact
            # Stage 2:
            g_i = g_i - s_offset
            while len(label_list) < num_data_points:
                selected_idx = g_i.argmin()
                label_list.append(torch.as_tensor(selected_idx, device=self.setup["device"]))
                g_i[idx] -= m_impact
            # Finalize labels:
            labels = torch.stack(label_list)

        elif self.cfg.label_strategy == "bias-corrected":  # WIP
            # This is slightly modified analytic label recovery in the style of Wainakh
            bias_per_query = [shared_data["gradients"][-1] for shared_data in user_data]
            label_list = []
            # Stage 1
            average_bias = torch.stack(bias_per_query).mean(dim=0)
            valid_classes = (average_bias < 0).nonzero()
            label_list += [*valid_classes.squeeze(dim=-1)]
            m_impact = average_bias_correct_label = average_bias[valid_classes].sum() / num_data_points

            average_bias[valid_classes] = average_bias[valid_classes] - m_impact
            # Stage 2
            while len(label_list) < num_data_points:
                selected_idx = average_bias.argmin()
                label_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact
            labels = torch.stack(label_list)
        elif self.cfg.label_strategy == "bias-text":  # WIP
            num_missing_labels = num_data_points * self.data_shape[0]
            # This is slightly modified analytic label recovery in the style of Wainakh
            bias_per_query = [shared_data["gradients"][-1] for shared_data in user_data]
            label_list = []
            # Stage 1
            average_bias = torch.stack(bias_per_query).mean(dim=0)
            valid_classes = (average_bias < 0).nonzero()
            label_list += [*valid_classes.squeeze(dim=-1)]
            tokens_in_input = embeddings[0]["grads"].norm(dim=-1).nonzero().squeeze(dim=-1)
            for token in tokens_in_input:
                if token not in label_list:
                    label_list.append(token)

            m_impact = average_bias_correct_label = average_bias[valid_classes].sum() / num_missing_labels

            average_bias[valid_classes] = average_bias[valid_classes] - m_impact
            # Stage 2
            while len(label_list) < num_missing_labels:
                selected_idx = average_bias.argmin()
                label_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact
            labels = torch.stack(label_list).view(num_data_points, self.data_shape[0])

        elif self.cfg.label_strategy == "random":
            # A random baseline
            labels = torch.randint(0, num_classes, (num_data_points,), device=self.setup["device"])
        elif self.cfg.label_strategy == "exhaustive":
            # Exhaustive search is possible in principle
            combinations = num_classes**num_data_points
            raise ValueError(
                f"Exhaustive label searching not implemented. Nothing stops you though from running your"
                f"attack algorithm for any possible combination of labels, except computational effort."
                f"In the given setting, a naive exhaustive strategy would attack {combinations} label vectors."
            )
            # Although this is arguably a worst-case estimate, you might be able to get "close enough" to the actual
            # label vector in much fewer queries, depending on which notion of close-enough makes sense for a given attack.
        else:
            raise ValueError(f"Invalid label recovery strategy {self.cfg.label_strategy} given.")

        # Pad with random labels if too few were produced:
        if len(labels) < num_data_points:
            labels = torch.cat(
                [labels, torch.randint(0, num_classes, (num_data_points - len(labels),), device=self.setup["device"])]
            )

        # Always sort, order does not matter here:
        labels = labels.sort()[0]
        log.info(f"Recovered labels {labels.tolist()} through strategy {self.cfg.label_strategy}.")
        return labels

    def recover_token_information(self, user_data, server_payload, model_name):
        """Recover token information. This is a variation of previous attacks on label recovery, but can abuse
        the embeddings layer in addition to the decoder layer.

        The behavior with respect to multiple queries is work in progress and subject of debate.
        """
        if self.cfg.token_strategy is None:
            return None
        embedding_parameter_idx, decoder_bias_parameter_idx = lookup_grad_indices(model_name)
        num_data_points = user_data[0]["metadata"]["num_data_points"]
        num_queries = len(user_data)
        token_cutoff = getattr(self.cfg, "token_cutoff", 3.5)

        # have to assert that this is the real decoder bias and embedding
        if decoder_bias_parameter_idx is not None:
            bias_per_query = [shared_data["gradients"][decoder_bias_parameter_idx] for shared_data in user_data]
            assert len(bias_per_query[0]) == server_payload[0]["metadata"]["vocab_size"]

        wte_per_query = [shared_data["gradients"][embedding_parameter_idx] for shared_data in user_data]
        assert wte_per_query[0].shape[0] == server_payload[0]["metadata"]["vocab_size"]

        num_missing_tokens = num_data_points * self.data_shape[0]

        if self.cfg.token_strategy == "decoder-bias":
            if decoder_bias_parameter_idx is None:
                raise ValueError("Cannot use this strategy on a model without decoder bias.")
            # works super well for normal stuff like transformer3 without tying

            # This is slightly modified analytic label recovery in the style of Wainakh

            token_list = []
            # Stage 1
            average_bias = torch.stack(bias_per_query).mean(dim=0)
            average_wte_norm = torch.stack(wte_per_query).mean(dim=0).norm(dim=1)
            valid_classes = (average_bias < 0).nonzero().squeeze(dim=-1)
            if len(valid_classes) > num_missing_tokens:
                # This should only happen due to numerical errors for bias or gradient noise:
                valid_classes = average_bias.topk(k=num_missing_tokens - 1, largest=False).indices
            token_list += [*valid_classes]
            # Supplement with missing tokens from input:
            std, mean = torch.std_mean(average_wte_norm.log())
            cutoff = mean + token_cutoff * std
            if not cutoff.isfinite():  # untied weights
                tokens_in_input = average_wte_norm.nonzero().squeeze(dim=-1)
            else:  # tied weights
                tokens_in_input = (average_wte_norm.log() > cutoff).nonzero().squeeze(dim=-1)
            for token in tokens_in_input:
                if token not in token_list:
                    token_list.append(token)

            m_impact = average_bias_correct_label = average_bias[valid_classes].sum() / num_missing_tokens

            average_bias[valid_classes] = average_bias[valid_classes] - m_impact
            # Stage 2
            while len(token_list) < num_missing_tokens:
                selected_idx = average_bias.argmin()
                token_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact
            tokens = torch.stack(token_list).view(num_data_points, self.data_shape[0])

        elif self.cfg.token_strategy == "embedding-norm":
            # This works decently well for GPT which has no decoder bias
            token_list = []
            # Stage 1
            average_wte_norm = torch.stack(wte_per_query).mean(dim=0).norm(dim=1)
            std, mean = torch.std_mean(average_wte_norm.log())

            valid_classes = []
            while len(valid_classes) == 0:  # Loop is usually unnecessary, but can recover from a bad cutoff
                cutoff = mean + token_cutoff * std
                if not cutoff.isfinite():  # untied weights
                    valid_classes = average_wte_norm.nonzero().squeeze(dim=-1)
                else:  # tied weights
                    valid_classes = (average_wte_norm.log() > cutoff).nonzero().squeeze(dim=-1)
                token_cutoff *= 0.8
            if cutoff.isfinite():
                log.info(f"Proceeded to cut estimated token distribution at {token_cutoff / 0.8:2.2f}.")

            if len(valid_classes) > num_missing_tokens:  # Cutoff overshoot
                valid_classes = average_wte_norm.topk(k=num_missing_tokens).indices
            token_list += [*valid_classes]

            # top2-log rule is decent:
            # top2 = average_wte_norm.log().topk(k=2).values  # log here is not an accident!
            # m_impact = top2[0] - top2[1]
            # but the sum is simpler:
            m_impact = average_wte_norm[valid_classes].sum() / num_missing_tokens

            average_wte_norm[valid_classes] = average_wte_norm[valid_classes] - m_impact
            # Stage 2
            while len(token_list) < num_missing_tokens:
                selected_idx = valid_classes[average_wte_norm[valid_classes].argmax()]
                token_list.append(selected_idx)
                average_wte_norm[selected_idx] -= m_impact
            tokens = torch.stack(token_list)

        elif self.cfg.token_strategy == "embedding-log":
            # Small variation of embedding-norm
            token_list = []
            # Stage 1
            average_wte_norm = torch.stack(wte_per_query).mean(dim=0).norm(dim=1)
            std, mean = torch.std_mean(average_wte_norm.log())
            valid_classes = []
            while len(valid_classes) == 0:
                cutoff = mean + token_cutoff * std
                if not cutoff.isfinite():  # untied weights
                    valid_classes = average_wte_norm.nonzero().squeeze(dim=-1)
                else:  # tied weights
                    valid_classes = (average_wte_norm.log() > cutoff).nonzero().squeeze(dim=-1)
                token_cutoff *= 0.8
            if cutoff.isfinite():
                log.info(f"Proceeded to cut estimated token distribution at {token_cutoff / 0.8:2.2f}.")
            if len(valid_classes) > num_missing_tokens:  # Cutoff overshoot
                valid_classes = average_wte_norm.topk(k=num_missing_tokens).indices
            token_list += [*valid_classes]

            average_wte_norm_log = average_wte_norm.log()
            m_impact = average_wte_norm_log[valid_classes].max() / torch.as_tensor(num_data_points).sqrt()
            # Stage 2
            while len(token_list) < num_missing_tokens:
                selected_idx = valid_classes[average_wte_norm_log[valid_classes].argmax()].squeeze()
                token_list.append(selected_idx)
                average_wte_norm_log[selected_idx] -= m_impact
            tokens = torch.stack(token_list)

        elif self.cfg.token_strategy == "mixed":
            # Can improve performance for tied embeddings over just decoder-bias
            # as unique token extraction is slightly more exact from the embedding layer

            token_list = []
            # Stage 1
            average_bias = torch.stack(bias_per_query).mean(dim=0)
            average_wte_norm = torch.stack(wte_per_query).mean(dim=0).norm(dim=1)
            std, mean = torch.std_mean(average_wte_norm.log())
            valid_classes = []
            while len(valid_classes) == 0:
                cutoff = mean + token_cutoff * std
                if not cutoff.isfinite():  # untied weights
                    valid_classes = average_wte_norm.nonzero().squeeze(dim=-1)
                else:  # tied weights
                    valid_classes = (average_wte_norm.log() > cutoff).nonzero().squeeze(dim=-1)
                token_cutoff *= 0.8
            if cutoff.isfinite():
                log.info(f"Proceeded to cut estimated token distribution at {token_cutoff / 0.8:2.2f}.")
            token_list += [*valid_classes]

            m_impact = average_bias[valid_classes].sum() / num_missing_tokens
            average_bias[valid_classes] = average_bias[valid_classes] - m_impact
            # Stage 2
            while len(token_list) < num_missing_tokens:
                selected_idx = valid_classes[average_bias[valid_classes].argmin()]
                token_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact
            tokens = torch.stack(token_list)

        elif self.cfg.token_strategy == "greedy-embedding":
            # Sanity check without unique token selection
            token_list = []
            # Stage 1
            average_wte_norm = torch.stack(wte_per_query).mean(dim=0).norm(dim=1)
            m_impact = average_wte_norm.sum() / num_missing_tokens
            # Stage 2
            while len(token_list) < num_missing_tokens:
                selected_idx = average_wte_norm.argmin()
                token_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact
            tokens = torch.stack(token_list)
        elif self.cfg.token_strategy == "greedy-bias":
            # Sanity check without unique token selection
            token_list = []
            # Stage 1
            average_bias = torch.stack(bias_per_query).mean(dim=0)
            m_impact = average_bias.sum() / num_missing_tokens
            # Stage 2
            while len(token_list) < num_missing_tokens:
                selected_idx = average_bias.argmin()
                token_list.append(selected_idx)
                average_bias[selected_idx] -= m_impact
            tokens = torch.stack(token_list)
        else:
            raise ValueError(f"Invalid strategy {self.cfg.token_strategy} for token recovery before attack.")

        # Always sort, order does not matter here:
        tokens = tokens.sort()[0]
        log.info(f"Recovered tokens {tokens} through strategy {self.cfg.token_strategy}.")
        return tokens
