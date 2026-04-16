"""FedDF-specific helpers for ensemble distillation on the server."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from feddf_utils import extract_batch_inputs, unwrap_model_outputs
from torch.utils.data import DataLoader, Dataset, TensorDataset

from plato.algorithms import fedavg


class Algorithm(fedavg.Algorithm):
    """Algorithm helpers for aggregating logits and distilling the student."""

    @staticmethod
    def aggregate_teacher_logits(
        updates,
        payloads: Sequence[Mapping[str, torch.Tensor]],
        *,
        weighting: str = "uniform",
    ) -> torch.Tensor:
        """Compute the ensembled teacher logits for AVGLOGITS distillation."""
        if not payloads:
            raise ValueError("FedDF requires at least one logits payload.")

        first_logits = payloads[0].get("logits")
        if not isinstance(first_logits, torch.Tensor):
            raise TypeError("FedDF payloads must include a 'logits' tensor.")

        weighting_name = weighting.strip().lower()
        if weighting_name not in {"uniform", "samples"}:
            raise ValueError(
                "FedDF teacher weighting must be either 'uniform' or 'samples'."
            )

        total_samples = sum(getattr(update.report, "num_samples", 0) for update in updates)
        use_uniform_average = weighting_name == "uniform" or total_samples <= 0

        aggregated = torch.zeros_like(first_logits, dtype=torch.float32)

        for update, payload in zip(updates, payloads):
            logits = payload.get("logits")
            if not isinstance(logits, torch.Tensor):
                raise TypeError("FedDF payloads must include a 'logits' tensor.")
            if logits.shape != first_logits.shape:
                raise ValueError(
                    "FedDF client logits must share the same proxy-set shape."
                )

            if use_uniform_average:
                weight = 1 / len(payloads)
            else:
                weight = getattr(update.report, "num_samples", 0) / total_samples

            aggregated += logits.detach().float() * weight

        return aggregated

    def distill_weights(
        self,
        baseline_weights: Mapping[str, torch.Tensor],
        teacher_logits: torch.Tensor,
        proxy_dataset: Dataset,
        *,
        temperature: float,
        distillation_epochs: int,
        distillation_batch_size: int,
        distillation_learning_rate: float,
        distillation_optimizer_name: str,
        use_cosine_annealing: bool,
        shuffle_batches: bool,
    ) -> OrderedDict[str, torch.Tensor]:
        """Distill the server model on proxy inputs using ensemble logits."""
        if len(proxy_dataset) != len(teacher_logits):
            raise ValueError(
                "FedDF proxy samples and teacher logits must have matching lengths."
            )

        trainer = self.require_trainer()
        model = self.require_model()
        device = torch.device(getattr(trainer, "device", "cpu"))

        self.load_weights(baseline_weights)

        inputs = []
        for example in proxy_dataset:
            inputs.append(extract_batch_inputs(example))

        proxy_inputs = torch.stack(inputs)
        distillation_dataset = TensorDataset(proxy_inputs, teacher_logits.detach().cpu())
        dataloader = DataLoader(
            distillation_dataset,
            batch_size=distillation_batch_size,
            shuffle=shuffle_batches,
        )

        was_training = model.training
        model.to(device)
        model.train()

        optimizer_name = distillation_optimizer_name.strip().lower()
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=distillation_learning_rate,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=distillation_learning_rate,
            )
        else:
            raise ValueError(
                "FedDF distillation optimizer must be either 'adam' or 'sgd'."
            )

        total_steps = max(distillation_epochs * len(dataloader), 1)
        scheduler = None
        if use_cosine_annealing:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
            )

        for _ in range(distillation_epochs):
            for batch_inputs, batch_logits in dataloader:
                batch_inputs = batch_inputs.to(device)
                batch_logits = batch_logits.to(device)
                teacher_probs = torch.softmax(batch_logits / temperature, dim=1)

                optimizer.zero_grad()
                student_logits = unwrap_model_outputs(model(batch_inputs))
                student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
                loss = (
                    F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                    * temperature
                    * temperature
                )
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        if not was_training:
            model.eval()

        return OrderedDict(
            (name, tensor.detach().cpu().clone())
            for name, tensor in model.state_dict().items()
        )
