"""Implement server code. This will be short, if the server is honest, but a lot can happen for the malicious variants."""

import torch
import numpy as np
from scipy import stats
import copy

from .malicious_modifications import ImprintBlock, SparseImprintBlock, OneShotBlock, CuriousAbandonHonesty
from .malicious_modifications.parameter_utils import introspect_model, replace_module_by_instance
from .malicious_modifications.analytic_transformer_utils import (
    compute_feature_distribution,
    partially_disable_embedding,
    set_MHA,
    set_flow_backward_layer,
    disable_mha_layers,
    equalize_mha_layer,
    partially_norm_position,
    make_imprint_layer,
)
from .models.transformer_dictionary import lookup_module_names
from .models.language_models import LearnablePositionalEmbedding, PositionalEmbedding

from .aux_training import train_encoder_decoder
from .malicious_modifications.feat_decoders import generate_decoder

from .malicious_modifications.classattack_utils import (
    check_with_tolerance,
    reconstruct_feature,
    find_best_feat,
    estimate_gt_stats,
    wrap_indices,
)

from .data import construct_dataloader
import logging

log = logging.getLogger(__name__)


def construct_server(
    model, loss_fn, cfg_case, setup=dict(device=torch.device("cpu"), dtype=torch.float)
):
    """Interface function."""
    # # not (external_dataloader is None and cfg_case.server.has_external_data)
    # dataloader = None   # external_dataloader
    # cfg_case.server.name == "class_malicious_parameters":
    server = MaliciousClassParameterServer(model, loss_fn, cfg_case, setup, external_dataloader=None)
    return server


class HonestServer:
    """Implement an honest server protocol.

    This class loads and selects the initial model and then sends this model to the (simulated) user.
    If multiple queries are possible, then these have to loop externally over muliple rounds via .run_protocol

    Central output: self.distribute_payload -> Dict[parameters=parameters, buffers=buffers, metadata=DataHyperparams]
    """

    THREAT = "Honest-but-curious"

    def __init__(
        self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None
    ):
        """Inialize the server settings."""
        self.model = model
        self.model.eval()

        self.loss = loss
        self.setup = setup

        self.num_queries = 1  # cfg_case.server.num_queries

        # Data configuration has to be shared across all parties to keep preprocessing consistent:
        self.cfg_data = cfg_case.data
        self.cfg_server = cfg_case.server

        self.external_dataloader = external_dataloader

        self.secrets = dict()  # Should be nothing in here

    def __repr__(self):
        return f"""Server (of type {self.__class__.__name__}) with settings:
    Threat model: {self.THREAT}
    Number of planned queries: {self.num_queries}
    Has external/public data: {self.cfg_server.has_external_data}

    Model:
        model specification: {str(self.model.name)}
        model state: {self.cfg_server.model_state}
        {f'public buffers: {self.cfg_server.provide_public_buffers}' if len(list(self.model.buffers())) > 0 else ''}

    Secrets: {self.secrets}
    """

    def reconfigure_model(self, model_state, query_id=0):
        """Reinitialize, continue training or otherwise modify model parameters in a benign way."""
        self.model.cpu()  # References might have been used on GPU later on. Return to normal first.
        for name, module in self.model.named_modules():
            if model_state == "untrained":
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
            elif model_state == "trained":
                pass  # model was already loaded as pretrained model
            elif model_state == "linearized":
                with torch.no_grad():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        module.weight.data = module.running_var.data.clone()
                        module.bias.data = module.running_mean.data.clone() + 10
                    if isinstance(module, torch.nn.Conv2d) and hasattr(module, "bias"):
                        module.bias.data += 10
            elif model_state == "orthogonal":
                # reinit model with orthogonal parameters:
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
                if "conv" in name or "linear" in name:
                    torch.nn.init.orthogonal_(module.weight, gain=1)
            elif model_state == "unchanged":
                # Disregard potential to update this model
                pass

    def reset_model(self):
        pass

    def distribute_payload(self, query_id=0):
        """Server payload to send to users. These are only references to simplfiy the simulation."""

        self.reconfigure_model(self.cfg_server.model_state, query_id)
        honest_model_parameters = [p for p in self.model.parameters()]  # do not send only the generators
        if self.cfg_server.provide_public_buffers:
            # NOTE(dchu): buffers are tensors that don't need to store gradient
            # information (e.g. batch norm avg and stddev)
            honest_model_buffers = [b for b in self.model.buffers()]
        else:
            honest_model_buffers = None
        return dict(parameters=honest_model_parameters, buffers=honest_model_buffers, metadata=self.cfg_data)

    def vet_model(self, model):
        """This server is honest."""
        model = self.model  # Re-reference this everywhere
        return self.model

    def queries(self):
        return range(self.num_queries)

    def run_protocol(self, user):
        """Helper function to simulate multiple queries given a user object."""
        # Simulate a simple FL protocol
        shared_user_data = []
        payloads = []
        for query_id in self.queries():
            server_payload = self.distribute_payload(query_id)  # A malicious server can return something "fun" here
            shared_data_per_round, true_user_data = user.compute_local_updates(server_payload)
            # true_data can only be used for analysis
            payloads += [server_payload]
            shared_user_data += [shared_data_per_round]
        return shared_user_data, payloads, true_user_data


class MaliciousModelServer(HonestServer):
    """Implement a malicious server protocol.

    This server is now also able to modify the model maliciously, before sending out payloads.
    Architectural changes (via self.prepare_model) are triggered before instantation of user objects.
    These architectural changes can also be understood as a 'malicious analyst' and happen first.
    """

    THREAT = "Malicious (Analyst)"

    CANDIDATE_FIRST_LAYERS = (
        torch.nn.Linear,
        torch.nn.Flatten,
        torch.nn.Conv2d,
        LearnablePositionalEmbedding,
        PositionalEmbedding,
        # Token Embeddings are not valid "first" layers and hencec not included here
    )

    def __init__(
        self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None
    ):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_case, setup, external_dataloader)
        self.model_state = "custom"  # Do not mess with model parameters no matter what init is agreed upon
        self.secrets = dict()

    def vet_model(self, model):
        """This server is not honest :>"""

        modified_model = self.model
        if self.cfg_server.model_modification.type == "ImprintBlock":
            block_fn = ImprintBlock
        elif self.cfg_server.model_modification.type == "SparseImprintBlock":
            block_fn = SparseImprintBlock
        elif self.cfg_server.model_modification.type == "OneShotBlock":
            block_fn = OneShotBlock
        elif self.cfg_server.model_modification.type == "CuriousAbandonHonesty":
            block_fn = CuriousAbandonHonesty
        else:
            raise ValueError("Unknown modification")

        modified_model, secrets = self._place_malicious_block(
            modified_model, block_fn, **self.cfg_server.model_modification
        )
        self.secrets["ImprintBlock"] = secrets

        if self.cfg_server.model_modification.position is not None:
            if self.cfg_server.model_modification.type == "SparseImprintBlock":
                block_fn = type(None)  # Linearize the full model for SparseImprint
            if self.cfg_server.model_modification.handle_preceding_layers == "identity":
                self._linearize_up_to_imprint(modified_model, block_fn)
            elif self.cfg_server.model_modification.handle_preceding_layers == "VAE":
                # Train preceding layers to be a VAE up to the target dimension
                modified_model, decoder = self.train_encoder_decoder(modified_model, block_fn)
                self.secrets["ImprintBlock"]["decoder"] = decoder
            else:
                # Otherwise do not modify the preceding layers. The attack then returns the layer input at this position directly
                pass

        # Reduce failures in later layers:
        # Note that this clashes with the VAE option!
        self._normalize_throughput(
            modified_model, gain=self.cfg_server.model_gain, trials=self.cfg_server.normalize_rounds
        )
        self.model = modified_model
        model = modified_model
        return self.model

    def _place_malicious_block(
        self, modified_model, block_fn, type, position=None, handle_preceding_layers=None, **kwargs
    ):
        """The block is placed directly before the named module given by "position".
        If none is given, the block is placed before the first layer.
        """
        if position is None:
            all_module_layers = {name: module for name, module in modified_model.named_modules()}
            for name, module in modified_model.named_modules():
                if isinstance(module, self.CANDIDATE_FIRST_LAYERS):
                    log.info(f"First layer determined to be {name}")
                    position = name
                    break

        block_found = False
        for name, module in modified_model.named_modules():
            if position in name:  # give some leeway for additional containers.
                feature_shapes = introspect_model(modified_model, tuple(self.cfg_data.shape), self.cfg_data.modality)
                data_shape = feature_shapes[name]["shape"][1:]
                print(f"Block inserted at feature shape {data_shape}.")
                module_to_be_modified = module
                block_found = True
                break

        if not block_found:
            raise ValueError(f"Could not find module {position} in model to insert layer.")

        # Insert malicious block:
        block = block_fn(data_shape, **kwargs)
        replacement = torch.nn.Sequential(block, module_to_be_modified)
        replace_module_by_instance(modified_model, module_to_be_modified, replacement)
        for idx, param in enumerate(modified_model.parameters()):
            if param is block.linear0.weight:
                weight_idx = idx
            if param is block.linear0.bias:
                bias_idx = idx
        secrets = dict(weight_idx=weight_idx, bias_idx=bias_idx, shape=data_shape, structure=block.structure)

        return modified_model, secrets

    def _linearize_up_to_imprint(self, model, block_fn):
        """This linearization option only works for a ResNet architecture."""
        first_conv_set = False  # todo: make this nice
        for name, module in self.model.named_modules():
            if isinstance(module, block_fn):
                break
            with torch.no_grad():
                if isinstance(module, torch.nn.BatchNorm2d):
                    # module.weight.data = (module.running_var.data.clone() + module.eps).sqrt()
                    # module.bias.data = module.running_mean.data.clone()
                    torch.nn.init.ones_(module.running_var)
                    torch.nn.init.ones_(module.weight)
                    torch.nn.init.zeros_(module.running_mean)
                    torch.nn.init.zeros_(module.bias)
                if isinstance(module, torch.nn.Conv2d):
                    if not first_conv_set:
                        torch.nn.init.dirac_(module.weight)
                        num_groups = module.out_channels // 3
                        module.weight.data[: num_groups * 3] = torch.cat(
                            [module.weight.data[:3, :3, :, :]] * num_groups
                        )
                        first_conv_set = True
                    else:
                        torch.nn.init.zeros_(module.weight)  # this is the resnet rule
                if "downsample.0" in name:
                    torch.nn.init.dirac_(module.weight)
                    num_groups = module.out_channels // module.in_channels
                    concat = torch.cat(
                        [module.weight.data[: module.in_channels, : module.in_channels, :, :]] * num_groups
                    )
                    module.weight.data[: num_groups * module.in_channels] = concat
                if isinstance(module, torch.nn.ReLU):
                    replace_module_by_instance(model, module, torch.nn.Identity())

    @torch.inference_mode()
    def _normalize_throughput(self, model, gain=1, trials=1, bn_modeset=False):
        """Reset throughput to be within standard mean and gain-times standard deviation."""
        features = dict()

        def named_hook(name):
            def hook_fn(module, input, output):
                features[name] = output

            return hook_fn

        if trials > 0:
            log.info(f"Normalizing model throughput with gain {gain}...")
            model.to(**self.setup)
        for round in range(trials):
            if not bn_modeset:
                for name, module in model.named_modules():
                    if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                        if isinstance(module, torch.nn.Conv2d) and module.bias is None:
                            if "downsample.0" in name:
                                module.weight.data.zero_()
                                log.info(f"Reset weight in downsample {name} to zero.")
                            continue

                        if "downsample.1" in name:
                            continue
                        hook = module.register_forward_hook(named_hook(name))
                        if self.external_dataloader is not None:
                            random_data_sample = next(iter(self.external_dataloader))[0].to(**self.setup)
                        else:
                            random_data_sample = torch.randn(
                                self.cfg_data.batch_size, *self.cfg_data.shape, **self.setup
                            )

                        model(random_data_sample)
                        std, mu = torch.std_mean(features[name])
                        log.info(f"Current mean of layer {name} is {mu.item()}, std is {std.item()} in round {round}.")

                        with torch.no_grad():
                            module.weight.data /= std / gain + 1e-8
                            module.bias.data -= mu / (std / gain + 1e-8)
                        hook.remove()
                        del features[name]
            else:
                model.train()
                if self.external_dataloader is not None:
                    random_data_sample = next(iter(self.external_dataloader))[0].to(**self.setup)
                else:
                    random_data_sample = torch.randn(self.cfg_data.batch_size, *self.cfg_data.shape, **self.setup)
                model(random_data_sample)
                model.eval()
        # Free up GPU:
        model.to(device=torch.device("cpu"))

    def train_encoder_decoder(self, modified_model, block_fn):
        """Train a compressed code (with VAE) that will then be found by the attacker."""
        if self.external_dataloader is None:
            raise ValueError("External data is necessary to train an optimal encoder/decoder structure.")

        # Unroll model up to imprint block
        # For now only the last position is allowed:
        layer_cake = list(modified_model.children())
        encoder = torch.nn.Sequential(*(layer_cake[:-1]), torch.nn.Flatten())
        decoder = generate_decoder(modified_model)
        log.info(encoder)
        log.info(decoder)
        stats = train_encoder_decoder(encoder, decoder, self.external_dataloader, self.setup)
        return modified_model, decoder


class MaliciousClassParameterServer(HonestServer):
    """Modify parameters for the "class attack" which can pick out a subset of image data from a larger batch."""

    THREAT = "Malicious (Parameters)"

    def __init__(
        self, model, loss, cfg_case, setup=dict(dtype=torch.float, device=torch.device("cpu")), external_dataloader=None
    ):
        """Inialize the server settings."""
        super().__init__(model, loss, cfg_case, setup, external_dataloader)
        self.model_state = "custom"  # Do not mess with model parameters no matter what init is agreed upon
        self.secrets = dict()
        self.original_model = copy.deepcopy(model)

    def reset_model(self):
        self.model = copy.deepcopy(self.original_model)

    def vet_model(self, model):
        """This server is not honest, but the model architecture stays normal."""
        model = self.model  # Re-reference this everywhere
        return self.model

    def run_protocol(self, user, additional_users=None, run_honest_protocol=False):
        """This server is allowed to run malicious protocols."""
        if run_honest_protocol:
            return super().run_protocol(user)
        else:
            if additional_users is None:
                return self.run_protocol_binary_attack(user) # This is what is run
            else:
                return self.run_protocol_feature_estimation(user, additional_users)

    def run_protocol_binary_attack(self, user):
        """Helper function for modified protocols, this is a binary attack that will repeatedly query a user
        with malicious server states."""
        # get class info first (this could be skipped and replaced by an attack on all/random labels)
        server_payload = self.distribute_payload()

        if self.cfg_server.query_once_for_labels:
            shared_data, true_user_data = user.compute_local_updates(server_payload)
            # This first query is not strictly necessary, you could also attack for a random class.
            t_labels = shared_data["metadata"]["labels"].detach().cpu().numpy()
            log.info(f"Found labels {t_labels} in first query.")
        else:
            # Choose random test labels to attack
            t_labels = np.random.choice(np.arange(0, self.cfg_data.classes), user.num_data_points)
            log.info(f"Randomly attacking labels {t_labels}.")
            shared_data = dict(gradients=None, buffers=None, metadata=dict())

        if self.cfg_server.opt_on_avg_grad:
            # optimize on averaged gradient with cls attack
            log.info("Optimize on averaged gradient with cls attack.")

            # cls attack on all labels in the batch
            self.reconfigure_for_class_attack(target_classes=t_labels)
            server_payload = self.distribute_payload()
            shared_data, true_user_data = user.compute_local_updates(server_payload)
            final_shared_data = [shared_data]
            final_payload = [server_payload]
        else:
            # attack cls by cls
            target_cls = np.unique(t_labels)[self.cfg_server.target_cls_idx]  # Could be any class
            target_indx = np.where(t_labels == target_cls)[0]
            reduced_shared_data = copy.deepcopy(shared_data)
            reduced_shared_data["metadata"]["num_data_points"] = len(target_indx)
            reduced_shared_data["metadata"]["labels"] = shared_data["metadata"]["labels"][target_indx]

            if len(target_indx) == 1:
                # simple cls attack if there is no cls collision
                log.info(f"Attacking label {reduced_shared_data['metadata']['labels'].item()} with cls attack.")
                cls_to_obtain = int(reduced_shared_data["metadata"]["labels"][0])

                # This is where the parameter modification happens!
                self.reconfigure_for_class_attack(target_classes=cls_to_obtain)

                server_payload = self.distribute_payload()
                tmp_shared_data, true_user_data = user.compute_local_updates(server_payload)
                reduced_shared_data["gradients"] = tmp_shared_data["gradients"]
                final_shared_data = [reduced_shared_data]
                final_payload = [server_payload]

                self.secrets["ClassAttack"] = dict(
                    num_data=1,
                    target_indx=target_indx,
                    true_num_data=shared_data["metadata"]["num_data_points"],
                    all_labels=shared_data["metadata"]["labels"],
                )
            else:
                # send several queries because of cls collision
                log.info(f"Attacking label {reduced_shared_data['metadata']['labels'][0].item()} with binary attack.")
                cls_to_obtain = int(shared_data["metadata"]["labels"][0])
                num_collisions = (shared_data["metadata"]["labels"] == int(cls_to_obtain)).sum()
                log.info(f"There are in total {num_collisions.item()} datapoints with label {cls_to_obtain}.")

                # find the starting point and the feature entry gives the max avg value
                self.reconfigure_for_class_attack(target_classes=cls_to_obtain)
                server_payload = self.distribute_payload()
                tmp_shared_data, true_user_data = user.compute_local_updates(server_payload)
                avg_feature = torch.flatten(reconstruct_feature(tmp_shared_data, cls_to_obtain))

                single_gradient_recovered = False

                while not single_gradient_recovered:
                    feature_loc = int(torch.argmax(avg_feature))
                    feature_val = float(avg_feature[feature_loc])
                    attack_state = dict(feature_loc=feature_loc, feature_val=feature_val)

                    # binary attack to recover all single gradients
                    attack_state["num_target_data"] = int(
                        torch.count_nonzero((reduced_shared_data["metadata"]["labels"] == int(cls_to_obtain)).to(int))
                    )
                    attack_state["num_data_points"] = shared_data["metadata"]["num_data_points"]

                    if self.cfg_server.one_shot_binary_attack:
                        recovered_single_gradients = self.one_shot_binary_attack(user, cls_to_obtain, attack_state)
                    else:
                        recovered_single_gradients = self.binary_attack(user, cls_to_obtain, attack_state)

                    if recovered_single_gradients is not None:
                        single_gradient_recovered = True
                    else:
                        avg_feature[feature_loc] = -1000

                    if not single_gradient_recovered:
                        log.info(f"Spent {user.counted_queries} user queries so far.")

                # return to the model with multiplier=1, (better with larger multiplier, but not optimizable if it is too large)
                self.reconfigure_for_feature_attack(
                    feature_val, feature_loc, target_classes=cls_to_obtain, allow_reset_param_weights=True
                )
                server_payload = self.distribute_payload()

                # recover image by image
                # add reversed() because the ith is always more confident than i-1th
                grad_i = list(reversed(recovered_single_gradients))[self.cfg_server.grad_idx]
                log.info(
                    f"Start recovering datapoint {self.cfg_server.grad_idx} of label "
                    f"{reduced_shared_data['metadata']['labels'][0].item()}."
                )

                final_shared_data = copy.deepcopy(reduced_shared_data)
                final_shared_data["metadata"]["num_data_points"] = 1
                final_shared_data["metadata"]["labels"] = reduced_shared_data["metadata"]["labels"][0:1]
                final_shared_data["gradients"] = grad_i

                final_shared_data = [final_shared_data]
                final_payload = [server_payload]

                self.secrets["ClassAttack"] = dict(
                    num_data=1,
                    target_indx=target_indx[self.cfg_server.grad_idx],
                    true_num_data=shared_data["metadata"]["num_data_points"],
                    all_labels=shared_data["metadata"]["labels"],
                )

        log.info(f"User {user.user_idx} was queried {user.counted_queries} times.")
        return final_shared_data, final_payload, true_user_data

    def run_protocol_feature_estimation(self, target_user, additional_users):
        """Estimate feature based on queries to additional_users to finally attack the target_user."""

        log.info(f"Estimating feature distribution based on {len(additional_users)} given additional users.")

        self.reconfigure_for_class_attack()
        est_features, est_sample_sizes = self.estimate_feat(additional_users)
        feature_loc = find_best_feat(est_features, est_sample_sizes, method="kstest")

        est_mean, est_std = estimate_gt_stats(est_features, est_sample_sizes, indx=feature_loc)

        expected_data_points = np.sum(est_sample_sizes) / len(additional_users)
        if expected_data_points == 1:  # no collisions expected?
            feature_val = self.cfg_server.class_multiplier
        else:
            expected_quantile = 1 / expected_data_points * self.cfg_server.reweight_collisions
            feature_val = stats.norm.ppf(expected_quantile, est_mean, est_std)
        log.info(
            f"Feature {feature_loc} with est. distribution mu={est_mean:2.4f},std={est_std:2.4f} "
            f"cut off with value {feature_val} due to {expected_data_points} expected data points."
        )
        self.reconfigure_for_feature_attack(feature_val, feature_loc)

        log.info("Commencing with update on target user.")
        server_payload = self.distribute_payload()
        shared_data, true_user_data = target_user.compute_local_updates(server_payload)

        self.reconfigure_for_feature_attack(feature_val, feature_loc, allow_reset_param_weights=True)
        true_user_data["distribution"] = est_features[feature_loc]

        return [shared_data], [server_payload], true_user_data

    def one_shot_binary_attack(self, user, cls_to_obtain, attack_state):
        feature_loc = attack_state["feature_loc"]
        feature_val = attack_state["feature_val"]
        num_data_points = attack_state["num_data_points"]
        all_feature_val = []

        feature_within_tolerance = False
        while not feature_within_tolerance:
            all_feature_val.append(feature_val)
            log.info(f"Querying feature {feature_loc} with feature val {feature_val}.")
            self.reconfigure_for_feature_attack(feature_val, feature_loc, target_classes=cls_to_obtain)
            server_payload = self.distribute_payload()
            shared_data, _ = user.compute_local_updates(server_payload)
            avg_feature = torch.flatten(reconstruct_feature(shared_data, cls_to_obtain))
            feature_val = float(avg_feature[feature_loc])
            log.info(f"And found avg feature val {feature_val}.")
            if check_with_tolerance(feature_val, all_feature_val, threshold=self.cfg_server.feat_threshold):
                curr_grad = list(shared_data["gradients"])
                feature_within_tolerance = True

        curr_grad[-1] = curr_grad[-1] * num_data_points
        curr_grad[:-1] = [grad_ii * num_data_points / self.cfg_server.feat_multiplier for grad_ii in curr_grad[:-1]]

        return [curr_grad]

    def binary_attack(self, user, cls_to_obtain, attack_state):
        feature_val = attack_state["feature_val"]
        num_target_data = attack_state["num_target_data"]
        num_data_points = attack_state["num_data_points"]

        # get filter feature points first
        all_feature_val = []
        attack_state["feat_grad"] = []
        attack_state["visited"] = []
        attack_state["counter"] = 0
        retval = self.binary_attack_recursion(user, cls_to_obtain, attack_state, [feature_val], all_feature_val)
        if retval == 0:  # Stop early after too many attempts in binary search:
            return None
        all_feature_val = np.array(all_feature_val)
        sorted_inds = np.argsort(all_feature_val)
        sorted_feat_grad = []
        all_feature_val = all_feature_val[sorted_inds]
        for i in sorted_inds:
            sorted_feat_grad.append(attack_state["feat_grad"][i])
        attack_state["feat_grad"] = sorted_feat_grad

        # recover gradients
        curr_grad = copy.deepcopy(list(attack_state["feat_grad"][0]))
        curr_grad[-1] = curr_grad[-1] * num_data_points
        curr_grad[:-1] = [grad_ii * num_data_points / self.cfg_server.feat_multiplier for grad_ii in curr_grad[:-1]]
        prev_grad = copy.deepcopy(curr_grad)
        single_gradients = [curr_grad]
        for i in range(1, len(all_feature_val)):
            curr_grad = copy.deepcopy(list(attack_state["feat_grad"][i]))
            curr_grad[-1] = curr_grad[-1] * num_data_points
            curr_grad[:-1] = [grad_ii * num_data_points / self.cfg_server.feat_multiplier for grad_ii in curr_grad[:-1]]
            grad_i = [grad_ii - grad_jj for grad_ii, grad_jj in zip(curr_grad, prev_grad)]
            single_gradients.append(grad_i)
            prev_grad = copy.deepcopy(curr_grad)

        return single_gradients

    def binary_attack_recursion(self, user, cls_to_obtain, attack_state, feat_01_values, all_feature_val):

        if len(all_feature_val) >= attack_state["num_target_data"]:
            return 1
        if attack_state["counter"] >= attack_state["num_target_data"] ** 2:
            log.info(f"Too many attempts ({attack_state['counter']}) on this feature!")
            return 0

        new_feat_01_values = []

        # get left and right mid point
        feature_loc = attack_state["feature_loc"]

        for feat_01_value in feat_01_values:
            attack_state["feature_val"] = feat_01_value
            self.reconfigure_for_feature_attack(feat_01_value, feature_loc, target_classes=cls_to_obtain)
            server_payload = self.distribute_payload()
            shared_data, _ = user.compute_local_updates(server_payload)
            feat_0 = torch.flatten(reconstruct_feature(shared_data, cls_to_obtain))
            feat_0_value = float(feat_0[feature_loc])  # the middle includes left hand side
            feat_1_value = 2 * feat_01_value - feat_0_value
            attack_state["counter"] += 1

            feat_candidates = [feat_0_value]

            for feat_cand in feat_candidates:
                if check_with_tolerance(feat_cand, attack_state["visited"], self.cfg_server.feat_threshold):
                    pass
                else:
                    if not check_with_tolerance(feat_01_value, all_feature_val, self.cfg_server.feat_threshold):
                        all_feature_val.append(feat_01_value)
                        attack_state["feat_grad"].append(list(shared_data["gradients"]))
                    new_feat_01_values.append(feat_cand)
                    attack_state["visited"].append(feat_cand)

                if len(all_feature_val) >= attack_state["num_target_data"]:
                    return

                if attack_state["counter"] >= attack_state["num_target_data"] ** 2:
                    log.info(f"Too many attempts ({attack_state['counter']}) on this feature!")
                    return 0

            feat_candidates = [feat_1_value, (feat_01_value + feat_1_value) / 2, (feat_01_value + feat_0_value) / 2]

            for feat_cand in feat_candidates:
                if not check_with_tolerance(feat_cand, attack_state["visited"], self.cfg_server.feat_threshold):
                    new_feat_01_values.append(feat_cand)

        return self.binary_attack_recursion(user, cls_to_obtain, attack_state, feat_01_values, all_feature_val)

    def estimate_feat(self, additional_users, target_class=None):
        """Estimate features from externally given additional users."""
        est_features = []
        sample_sizes = []
        if target_class is None:
            target_class = self.cfg_server.target_cls_idx

        for user in additional_users:
            server_payload = self.distribute_payload()
            shared_data, _ = user.compute_local_updates(server_payload)
            num_target = int(torch.count_nonzero((shared_data["metadata"]["labels"] == int(target_class)).to(int)))
            if num_target != 0:
                est_features.append(
                    torch.flatten(reconstruct_feature(shared_data, target_class)).detach().cpu().numpy()
                )
                sample_sizes.append(num_target)

        if len(est_features) == 0:
            raise ValueError(f"These additional users do not own images from the target class {target_class}.")
        else:
            est_features = np.vstack(est_features)
            sample_sizes = np.array(sample_sizes)

        return est_features.T, sample_sizes

    @torch.no_grad()
    def reconfigure_for_class_attack(self, target_classes=None):
        self.reset_model()
        if target_classes is None:
            target_classes = [self.cfg_server.target_cls_idx]
        cls_to_obtain = wrap_indices(target_classes)

        *_, l_w, l_b = self.model.parameters()

        # linear weight
        masked_weight = torch.zeros_like(l_w)
        masked_weight[cls_to_obtain] = self.cfg_server.class_multiplier
        l_w.copy_(masked_weight)

        # linear bias
        masked_bias = torch.ones_like(l_b) * self.cfg_server.bias_multiplier
        masked_bias[cls_to_obtain] = l_b[cls_to_obtain]
        l_b.copy_(masked_bias)

    @torch.no_grad()
    def reconfigure_for_feature_attack(
        self, feature_val, feature_loc, target_classes=None, allow_reset_param_weights=False
    ):
        self.reset_model()
        if target_classes is None:
            target_classes = [self.cfg_server.target_cls_idx]
        cls_to_obtain = wrap_indices(target_classes)
        feature_loc = wrap_indices(feature_loc)

        if allow_reset_param_weights and self.cfg_server.reset_param_weights:
            feat_multiplier = 1
        else:
            feat_multiplier = self.cfg_server.feat_multiplier
        *_, l_w, l_b = self.model.parameters()

        masked_weight = torch.zeros_like(l_w)
        masked_weight[cls_to_obtain, feature_loc] = feat_multiplier
        l_w.copy_(masked_weight)

        masked_bias = torch.ones_like(l_b) * self.cfg_server.bias_multiplier
        masked_bias[cls_to_obtain] = -feature_val * self.cfg_server.feat_multiplier
        l_b.copy_(masked_bias)
