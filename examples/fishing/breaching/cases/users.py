"""Implement user code."""

import torch
import copy
from itertools import chain

from .data import construct_dataloader
import logging

log = logging.getLogger(__name__)


def construct_user(model, loss_fn, cfg_case, setup):
    """Interface function."""
    # cfg_case.user.user_type == "multiuser_aggregate":  # NOTE(dchu): FISHING
    dataloaders, indices = [], []
    for idx in range(*cfg_case.user.user_range):
        dataloaders += [construct_dataloader(cfg_case.data, cfg_case.impl, user_idx=idx)]
        indices += [idx]
    user = MultiUserAggregate(model, loss_fn, dataloaders, setup, cfg_case.user, user_indices=indices)
    return user


class UserSingleStep(torch.nn.Module):  # NOTE(dchu): FISHING
    """A user who computes a single local update step."""

    def __init__(self, model, loss, dataloader, setup, idx, cfg_user):
        """Initialize from cfg_user dict which contains atleast all keys in the matching .yaml :>"""
        super().__init__()
        self.num_data_points = cfg_user.num_data_points

        self.provide_labels = cfg_user.provide_labels
        self.provide_num_data_points = cfg_user.provide_num_data_points
        self.provide_buffers = cfg_user.provide_buffers

        self.user_idx = idx
        self.setup = setup

        self.model = copy.deepcopy(model)
        self.model.to(**setup)

        self.defense_repr = []
        self._initialize_local_privacy_measures(cfg_user.local_diff_privacy)

        self.dataloader = dataloader
        self.loss = copy.deepcopy(loss)  # Just in case the loss contains state

        self.counted_queries = 0  # Count queries to this user

    def __repr__(self):
        n = "\n"
        return f"""User (of type {self.__class__.__name__}) with settings:
    Number of data points: {self.num_data_points}

    Threat model:
    User provides labels: {self.provide_labels}
    User provides buffers: {self.provide_buffers}
    User provides number of data points: {self.provide_num_data_points}

    Data:
    Dataset: {self.dataloader.name}
    user: {self.user_idx}
    {n.join(self.defense_repr)}
        """

    def _initialize_local_privacy_measures(self, local_diff_privacy):
        """Initialize generators for noise in either gradient or input."""
        if local_diff_privacy["gradient_noise"] > 0.0:
            loc = torch.as_tensor(0.0, **self.setup)
            scale = torch.as_tensor(local_diff_privacy["gradient_noise"], **self.setup)
            if local_diff_privacy["distribution"] == "gaussian":
                self.generator = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif local_diff_privacy["distribution"] == "laplacian":
                self.generator = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {local_diff_privacy["distribution"]} given.')
            self.defense_repr.append(
                f'Defense: Local {local_diff_privacy["distribution"]} gradient noise with strength {scale.item()}.'
            )
        else:
            self.generator = None
        if local_diff_privacy["input_noise"] > 0.0:
            loc = torch.as_tensor(0.0, **self.setup)
            scale = torch.as_tensor(local_diff_privacy["input_noise"], **self.setup)
            if local_diff_privacy["distribution"] == "gaussian":
                self.generator_input = torch.distributions.normal.Normal(loc=loc, scale=scale)
            elif local_diff_privacy["distribution"] == "laplacian":
                self.generator_input = torch.distributions.laplace.Laplace(loc=loc, scale=scale)
            else:
                raise ValueError(f'Invalid distribution {local_diff_privacy["distribution"]} given.')
            self.defense_repr.append(
                f'Defense: Local {local_diff_privacy["distribution"]} input noise with strength {scale.item()}.'
            )
        else:
            self.generator_input = None
        self.clip_value = local_diff_privacy.get("per_example_clipping", 0.0)
        if self.clip_value > 0:
            self.defense_repr.append(f"Defense: Gradient clipping to maximum of {self.clip_value}.")

    def compute_local_updates(self, server_payload, custom_data=None):
        """Compute local updates to the given model based on server payload.

        Batchnorm behavior:
        If public buffers are sent by the server, then the user will be set into evaluation mode
        Otherwise the user is in training mode and sends back buffer based on .provide_buffers.

        Shared labels are canonically sorted for simplicity.

        Optionally custom data can be directly inserted here, superseding actual user data.
        Use this behavior only for demonstrations.
        """
        self.counted_queries += 1
        if custom_data is None:
            data = self._load_data()
        else:
            data = custom_data
        B = data["labels"].shape[0]
        # Compute local updates
        shared_grads = []
        shared_buffers = []

        parameters = server_payload["parameters"]
        buffers = server_payload["buffers"]

        with torch.no_grad():
            for param, server_state in zip(self.model.parameters(), parameters):
                param.copy_(server_state.to(**self.setup))
            if buffers is not None:
                for buffer, server_state in zip(self.model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
                self.model.eval()
            else:
                for module in self.model.modules():
                    if hasattr(module, "momentum"):
                        module.momentum = None  # Force recovery without division
                self.model.train()
        log.info(
            f"Computing user update on user {self.user_idx} in model mode: {'training' if self.model.training else 'eval'}."
        )

        def _compute_batch_gradient(data):
            data[self.data_key] = (
                data[self.data_key] + self.generator_input.sample(data[self.data_key].shape)
                if self.generator_input is not None
                else data[self.data_key]
            )
            outputs = self.model(**data)
            loss = self.loss(outputs, data["labels"])
            return torch.autograd.grad(loss, self.model.parameters())

        if self.clip_value > 0:  # Compute per-example gradients and clip them in this case
            shared_grads = [torch.zeros_like(p) for p in self.model.parameters()]
            for data_idx in range(B):
                data_point = {key: val[data_idx : data_idx + 1] for key, val in data.items()}
                per_example_grads = _compute_batch_gradient(data_point)
                self._clip_list_of_grad_(per_example_grads)
                torch._foreach_add_(shared_grads, per_example_grads)
            torch._foreach_div_(shared_grads, B)
        else:
            # Compute the forward pass
            shared_grads = _compute_batch_gradient(data)
        self._apply_differential_noise(shared_grads)

        if buffers is not None:
            shared_buffers = None
        else:
            shared_buffers = [b.clone().detach() for b in self.model.buffers()]

        metadata = dict(
            num_data_points=self.num_data_points if self.provide_num_data_points else None,
            labels=data["labels"].sort()[0] if self.provide_labels else None,
            local_hyperparams=None,
        )
        shared_data = dict(
            gradients=shared_grads, buffers=shared_buffers if self.provide_buffers else None, metadata=metadata
        )
        true_user_data = dict(data=data[self.data_key], labels=data["labels"], buffers=shared_buffers)

        return shared_data, true_user_data

    def _clip_list_of_grad_(self, grads):
        """Apply differential privacy component per-example clipping."""
        grad_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)
        if grad_norm > self.clip_value:
            [g.mul_(self.clip_value / (grad_norm + 1e-6)) for g in grads]

    def _apply_differential_noise(self, grads):
        """Apply differential privacy component gradient noise."""
        if self.generator is not None:
            for grad in grads:
                grad += self.generator.sample(grad.shape)

    def _load_data(self, setup=None):
        """Generate data from dataloader, truncated by self.num_data_points"""
        # Select data
        data_blocks = []
        num_samples = 0

        if setup is None:
            setup = self.setup

        for idx, data_block in enumerate(self.dataloader):
            data_blocks += [data_block]
            num_samples += data_block["labels"].shape[0]
            if num_samples > self.num_data_points:
                break

        if num_samples < self.num_data_points:
            raise ValueError(
                f"This user does not have the requested {self.num_data_points} samples,"
                f"they only own {num_samples} samples."
            )

        data = dict()
        for key in data_blocks[0]:
            data[key] = torch.cat([d[key] for d in data_blocks], dim=0)[: self.num_data_points].to(
                device=setup["device"]
            )
        self.data_key = "input_ids" if "input_ids" in data.keys() else "inputs"
        return data

    def print(self, user_data, **kwargs):
        """Print decoded user data to output."""
        tokenizer = self.dataloader.dataset.tokenizer
        decoded_tokens = tokenizer.batch_decode(user_data["data"], clean_up_tokenization_spaces=True)
        for line in decoded_tokens:
            print(line)

    def print_with_confidence(self, user_data, **kwargs):
        """Print decoded user data to output."""
        tokenizer = self.dataloader.dataset.tokenizer
        colors = [160, 166, 172, 178, 184, 190]
        thresholds = torch.as_tensor([0, 0.5, 0.75, 0.95, 0.99, 0.9999])

        def bg_color(text, confidence_score):
            threshold = ((confidence_score > thresholds) + torch.arange(0, len(colors)) / 100).argmax()
            return "\33[48;5;" + str(colors[threshold]) + "m" + text + "\33[0m"

        for sequence, sequence_confidence in zip(user_data["data"], user_data["confidence"]):
            for token, c in zip(sequence, sequence_confidence):
                decoded_token = tokenizer.decode(token)
                print(bg_color(decoded_token + " ", c), end="")
            print("\n")

    def print_and_mark_correct(self, user_data, true_user_data, **kwargs):
        """Print decoded user data to output."""
        tokenizer = self.dataloader.dataset.tokenizer

        def bg_color(text, correct):
            if correct:
                return "\33[48;5;190m" + text + "\33[0m"
            else:
                return "\33[48;5;160m" + text + "\33[0m"

        for sequence, gt_sequence in zip(user_data["data"], true_user_data["data"]):
            for token, gt_token in zip(sequence, gt_sequence):
                decoded_token = tokenizer.decode(token)
                print(bg_color(decoded_token + " ", token == gt_token), end="")
            print("\n")

    def plot(self, user_data, scale=False, print_labels=False):
        """Plot user data to output. Probably best called from a jupyter notebook."""
        import matplotlib.pyplot as plt  # lazily import this here

        dm = torch.as_tensor(self.dataloader.dataset.mean, **self.setup)[None, :, None, None]
        ds = torch.as_tensor(self.dataloader.dataset.std, **self.setup)[None, :, None, None]
        classes = self.dataloader.dataset.classes

        data = user_data["data"].clone().detach()
        labels = user_data["labels"].clone().detach() if user_data["labels"] is not None else None
        if labels is None:
            print_labels = False

        if scale:
            min_val, max_val = data.amin(dim=[2, 3], keepdim=True), data.amax(dim=[2, 3], keepdim=True)
            # print(f'min_val: {min_val} | max_val: {max_val}')
            data = (data - min_val) / (max_val - min_val)
        else:
            data.reshape((-1, 3, 224, 224)).mul_(ds).add_(dm).clamp_(0, 1)
        data = data.to(dtype=torch.float32).reshape((-1, 3, 224, 224))

        if data.shape[0] == 1:
            plt.axis("off")
            plt.imshow(data[0].permute(1, 2, 0).cpu())
            if print_labels:
                plt.title(f"Data with label {classes[labels]}")
        else:
            grid_shape = int(torch.as_tensor(data.shape[0]).sqrt().ceil())
            s = 24 if data.shape[3] > 150 else 6
            fig, axes = plt.subplots(grid_shape, grid_shape, figsize=(s, s))
            label_classes = []
            for i, (im, axis) in enumerate(zip(data, axes.flatten())):
                axis.imshow(im.permute(1, 2, 0).cpu())
                if labels is not None and print_labels:
                    label_classes.append(classes[labels[i]])
                axis.axis("off")
            if print_labels:
                print(label_classes)


class UserMultiStep(UserSingleStep):    # NOTE(dchu): FISHING
    """A user who computes multiple local update steps as in a FedAVG scenario."""

    def __init__(self, model, loss, dataloader, setup, idx, cfg_user):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__(model, loss, dataloader, setup, idx, cfg_user)

        self.num_local_updates = cfg_user.num_local_updates
        self.num_data_per_local_update_step = cfg_user.num_data_per_local_update_step
        self.local_learning_rate = cfg_user.local_learning_rate
        self.provide_local_hyperparams = cfg_user.provide_local_hyperparams

    def __repr__(self):
        n = "\n"
        return (
            super().__repr__()
            + n
            + f"""    Local FL Setup:
        Number of local update steps: {self.num_local_updates}
        Data per local update step: {self.num_data_per_local_update_step}
        Local learning rate: {self.local_learning_rate}

        Threat model:
        Share these hyperparams to server: {self.provide_local_hyperparams}

        """
        )

    def compute_local_updates(self, server_payload):
        """Compute local updates to the given model based on server payload."""
        self.counted_queries += 1
        user_data = self._load_data()

        # Compute local updates
        parameters = server_payload["parameters"]
        buffers = server_payload["buffers"]

        with torch.no_grad():
            for param, server_state in zip(self.model.parameters(), parameters):
                param.copy_(server_state.to(**self.setup))
            if buffers is not None:
                for buffer, server_state in zip(self.model.buffers(), buffers):
                    buffer.copy_(server_state.to(**self.setup))
                self.model.eval()
            else:
                self.model.train()
        log.info(
            f"Computing user update on user {self.user_idx} in model mode: {'training' if self.model.training else 'eval'}."
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate)
        seen_data_idx = 0
        label_list = []
        for step in range(self.num_local_updates):
            data = {
                k: v[seen_data_idx : seen_data_idx + self.num_data_per_local_update_step] for k, v in user_data.items()
            }
            seen_data_idx += self.num_data_per_local_update_step
            seen_data_idx = seen_data_idx % self.num_data_points
            label_list.append(data["labels"].sort()[0])

            optimizer.zero_grad()
            # Compute the forward pass
            data[self.data_key] = (
                data[self.data_key] + self.generator_input.sample(data[self.data_key].shape)
                if self.generator_input is not None
                else data[self.data_key]
            )
            outputs = self.model(**data)
            loss = self.loss(outputs, data["labels"])
            loss.backward()

            grads_ref = [p.grad for p in self.model.parameters()]
            if self.clip_value > 0:
                self._clip_list_of_grad_(grads_ref)
            self._apply_differential_noise(grads_ref)
            optimizer.step()

        # Share differential to server version:
        # This is equivalent to sending the new stuff and letting the server do it, but in line
        # with the gradients sent in UserSingleStep
        shared_grads = [
            (p_local - p_server.to(**self.setup)).clone().detach()
            for (p_local, p_server) in zip(self.model.parameters(), parameters)
        ]

        shared_buffers = [b.clone().detach() for b in self.model.buffers()]
        metadata = dict(
            num_data_points=self.num_data_points if self.provide_num_data_points else None,
            labels=user_data["labels"] if self.provide_labels else None,
            local_hyperparams=dict(
                lr=self.local_learning_rate,
                steps=self.num_local_updates,
                data_per_step=self.num_data_per_local_update_step,
                labels=label_list,
            )
            if self.provide_local_hyperparams
            else None,
            data_key=self.data_key,
        )
        shared_data = dict(
            gradients=shared_grads, buffers=shared_buffers if self.provide_buffers else None, metadata=metadata
        )
        true_user_data = dict(data=user_data[self.data_key], labels=user_data["labels"], buffers=shared_buffers)

        return shared_data, true_user_data


class ChainedDataloader:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders

    def __iter__(self):
        return chain(*self.dataloaders)

    def __len__(self):
        return torch.as_tensor([len(loader) for loader in self.dataloaders]).sum().item()

    def __getattr__(self, name):
        """Call this only if all attributes of Subset are exhausted."""
        return getattr(self.dataloaders[0], name)


class MultiUserAggregate(UserMultiStep):
    """A silo of users who compute local updates as in a fedSGD or fedAVG scenario and aggregate their results.

    For an unaggregated single silo refer to SingleUser classes as above.
    This aggregration is assumed to be safe (e.g. via secure aggregation) and the attacker and server only gain
    access to the aggregated local updates.

    self.dataloader of this class is actually quite unwieldy, due to its possible size.
    """

    def __init__(self, model, loss, dataloaders, setup, cfg_user, user_indices):
        """Initialize but do not propagate the cfg_case.user dict further."""
        super().__init__(model, loss, dataloaders, setup, None, cfg_user)

        self.num_users = len(dataloaders)

        self.users = []
        self.user_setup = dict(dtype=setup["dtype"], device=torch.device("cpu"))  # Simulate on CPU
        for idx in user_indices:
            if self.num_local_updates > 1:
                self.users.append(UserMultiStep(model, loss, dataloaders[idx], self.user_setup, idx, cfg_user))
            else:
                self.users.append(UserSingleStep(model, loss, dataloaders[idx], self.user_setup, idx, cfg_user))

        self.dataloader = ChainedDataloader(dataloaders)
        self.user_idx = f"{user_indices[0]}_{user_indices[-1]}"  # Only for printout identification

    def __repr__(self):
        n = "\n"
        return self.users[0].__repr__() + n + f"""    Number of aggregated users: {self.num_users}"""

    def compute_local_updates(self, server_payload, return_true_user_data=False):
        """Compute local updates to the given model based on server payload.

        Collecting and returning a tensor containing all input data (as for the other users) is disabled by default.
        (to save your RAM).
        """
        self.counted_queries += 1
        # Compute local updates

        server_payload["parameters"] = [p.to(**self.setup) for p in server_payload["parameters"]]
        if server_payload["buffers"] is not None:
            server_payload["buffers"] = [b.to(**self.setup) for b in server_payload["buffers"]]

        aggregate_updates = [torch.zeros_like(p) for p in server_payload["parameters"]]
        aggregate_buffers = [torch.zeros_like(b, dtype=torch.float) for b in server_payload["buffers"]]
        aggregate_labels = []
        aggregate_label_lists = []  # Only ever used in rare sanity checks. List of labels per local update step
        aggregate_true_user_data = []
        aggregate_true_user_labels = []

        for user in self.users:
            user.to(**self.setup)
            user.setup = self.setup
            user_data, true_user_data = user.compute_local_updates(server_payload)
            user.to(**self.user_setup)
            user.setup = self.user_setup

            if return_true_user_data:
                aggregate_true_user_data += [true_user_data["data"].cpu()]
                if true_user_data["labels"] is not None:
                    aggregate_true_user_labels += [true_user_data["labels"].cpu()]
            torch._foreach_sub_(user_data["gradients"], aggregate_updates)
            torch._foreach_add_(aggregate_updates, user_data["gradients"], alpha=1 / self.num_users)

            if user_data["buffers"] is not None:
                torch._foreach_sub_(user_data["buffers"], aggregate_buffers)
                torch._foreach_add_(aggregate_buffers, buffer_to_server, alpha=1 / self.num_users)
            if user_data["metadata"]["labels"] is not None:
                aggregate_labels.append(user_data["metadata"]["labels"])
            if params := user_data["metadata"]["local_hyperparams"] is not None:
                if params["labels"] is not None:
                    aggregate_label_lists += [l.cpu() for l in user_data["metadata"]["local_hyperparams"]["labels"]]

        shared_data = dict(
            gradients=aggregate_updates,
            buffers=aggregate_buffers if self.provide_buffers else None,
            metadata=dict(
                num_data_points=self.num_data_points * len(self.users) if self.provide_num_data_points else None,
                labels=torch.cat(aggregate_labels).sort()[0] if self.provide_labels else None,
                num_users=self.num_users,
                local_hyperparams=dict(
                    lr=self.local_learning_rate,
                    steps=self.num_local_updates,
                    data_per_step=self.num_data_per_local_update_step,
                    labels=aggregate_label_lists,
                )
                if self.provide_local_hyperparams
                else None,
            ),
        )

        if return_true_user_data:
            aggregate_true_user_data = torch.cat(aggregate_true_user_data, dim=0)
            aggregate_true_user_labels = (
                torch.cat(aggregate_true_user_labels) if len(aggregate_true_user_labels) > 0 else None
            )

            true_user_data = dict(
                data=aggregate_true_user_data, labels=aggregate_true_user_labels, buffers=aggregate_buffers
            )

        return shared_data, true_user_data
