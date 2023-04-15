"""Implementation for basic gradient inversion attacks.

This covers optimization-based reconstruction attacks as in Wang et al. "Beyond Infer-
ring Class Representatives: User-Level Privacy Leakage From Federated Learning."
and convers subsequent developments such as
* Zhu et al., "Deep Leakage from gradients",
* Geiping et al., "Inverting Gradients - How easy is it to break privacy in FL"
* ?
"""

import torch
import time

from .base_attack import _BaseAttacker
from .auxiliaries.regularizers import regularizer_lookup, TotalVariation
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup

import logging

log = logging.getLogger(__name__)


class OptimizationBasedAttacker(_BaseAttacker):
    """Implements a wide spectrum of optimization-based attacks."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)
        objective_fn = objective_lookup.get(self.cfg.objective.type)
        if objective_fn is None:
            raise ValueError(f"Unknown objective type {self.cfg.objective.type} given.")
        else:
            self.objective = objective_fn(**self.cfg.objective)
        self.regularizers = []
        try:
            for key in self.cfg.regularization.keys():
                if self.cfg.regularization[key].scale > 0:
                    self.regularizers += [regularizer_lookup[key](self.setup, **self.cfg.regularization[key])]
        except AttributeError:
            pass  # No regularizers selected.

        try:
            self.augmentations = []
            for key in self.cfg.augmentations.keys():
                self.augmentations += [augmentation_lookup[key](**self.cfg.augmentations[key])]
            self.augmentations = torch.nn.Sequential(*self.augmentations).to(**setup)
        except AttributeError:
            self.augmentations = torch.nn.Sequential()  # No augmentations selected.

    def __repr__(self):
        n = "\n"
        return f"""Attacker (of type {self.__class__.__name__}) with settings:
    Hyperparameter Template: {self.cfg.type}

    Objective: {repr(self.objective)}
    Regularizers: {(n + ' '*18).join([repr(r) for r in self.regularizers])}
    Augmentations: {(n + ' '*18).join([repr(r) for r in self.augmentations])}

    Optimization Setup:
        {(n + ' ' * 8).join([f'{key}: {val}' for key, val in self.cfg.optim.items()])}
        """

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions = []
        try:
            for trial in range(self.cfg.restarts.num_trials):
                candidate_solutions += [
                    self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun)
                ]
                scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass
        optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
        reconstructed_data = dict(data=optimal_solution, labels=labels)
        if server_payload[0]["metadata"].modality == "text":
            reconstructed_data = self._postprocess_text_data(reconstructed_data)
        if "ClassAttack" in server_secrets:
            # Only a subset of images was actually reconstructed:
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
        return reconstructed_data, stats

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False):
        """Run a single reconstruction trial."""

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])

        # Initialize candidate reconstruction data
        candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
        if initial_data is not None:
            candidate.data = initial_data.data.clone().to(**self.setup)

        best_candidate = candidate.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        # Initialize optimizers
        optimizer, scheduler = self._init_optimizer([candidate])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective(candidate, labels, rec_model, optimizer, shared_data, iteration)
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate.detach().clone()

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    log.info(
                        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                    )
                    current_wallclock = timestamp

                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        return best_candidate.detach()

    def _compute_objective(self, candidate, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()

            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):
                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels)
                total_objective += objective
                total_task_loss += task_loss
            for regularizer in self.regularizers:
                total_objective += regularizer(candidate_augmented)

            if total_objective.requires_grad:
                total_objective.backward(inputs=candidate, create_graph=False)
            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(candidate.grad)
                    candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                if self.cfg.optim.grad_clip is not None:
                    grad_norm = candidate.grad.norm()
                    if grad_norm > self.cfg.optim.grad_clip:
                        candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                if self.cfg.optim.signed is not None:
                    if self.cfg.optim.signed == "soft":
                        scaling_factor = (
                            1 - iteration / self.cfg.optim.max_iterations
                        )  # just a simple linear rule for now
                        candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif self.cfg.optim.signed == "hard":
                        candidate.grad.sign_()
                    else:
                        pass

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure

    def _score_trial(self, candidate, labels, rec_model, shared_data):
        """Score candidate solutions based on some criterion."""

        if self.cfg.restarts.scoring in ["euclidean", "cosine-similarity"]:
            objective = Euclidean() if self.cfg.restarts.scoring == "euclidean" else CosineSimilarity()
            objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
            score = 0
            for model, data in zip(rec_model, shared_data):
                score += objective(model, data["gradients"], candidate, labels)[0]
        elif self.cfg.restarts.scoring in ["TV", "total-variation"]:
            score = TotalVariation(scale=1.0)(candidate)
        else:
            raise ValueError(f"Scoring mechanism {self.cfg.scoring} not implemented.")
        return score if score.isfinite() else float("inf")

    def _select_optimal_reconstruction(self, candidate_solutions, scores, stats):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        stats["opt_value"] = optimal_val.item()
        if optimal_val.isfinite():
            log.info(f"Optimal candidate solution with rec. loss {optimal_val.item():2.4f} selected.")
            return optimal_solution
        else:
            log.info("No valid reconstruction could be found.")
            return torch.zeros_like(optimal_solution)
