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
    ) -> torch.Tensor:
        """Compute a sample-weighted ensemble of client logits."""
        if not payloads:
            raise ValueError("FedDF requires at least one logits payload.")

        first_logits = payloads[0].get("logits")
        if not isinstance(first_logits, torch.Tensor):
            raise TypeError("FedDF payloads must include a 'logits' tensor.")

        total_samples = sum(getattr(update.report, "num_samples", 0) for update in updates)
        if total_samples <= 0:
            total_samples = len(payloads)

        aggregated = torch.zeros_like(first_logits, dtype=torch.float32)

        for update, payload in zip(updates, payloads):
            logits = payload.get("logits")
            if not isinstance(logits, torch.Tensor):
                raise TypeError("FedDF payloads must include a 'logits' tensor.")
            if logits.shape != first_logits.shape:
                raise ValueError(
                    "FedDF client logits must share the same proxy-set shape."
                )

            weight = getattr(update.report, "num_samples", 0) / total_samples
            if total_samples == len(payloads):
                weight = 1 / len(payloads)

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
            shuffle=False,
        )

        was_training = model.training
        model.to(device)
        model.train()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=distillation_learning_rate,
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

        if not was_training:
            model.eval()

        return OrderedDict(
            (name, tensor.detach().cpu().clone())
            for name, tensor in model.state_dict().items()
        )
