"""
Testing strategies for model evaluation.

This module provides default and custom testing strategy implementations
for the composable trainer architecture.
"""

import logging
import os

import torch

from plato.trainers.strategies.base import TestingStrategy, TrainingContext


class DefaultTestingStrategy(TestingStrategy):
    """
    Default testing strategy for standard classification tasks.

    Uses standard accuracy computation with argmax over logits.
    """

    def test_model(self, model, config, testset, sampler, context):
        """
        Test the model using standard classification accuracy.

        Args:
            model: The model to test
            config: Testing configuration dictionary
            testset: Test dataset
            sampler: Optional data sampler for test set
            context: Training context

        Returns:
            Classification accuracy as a float
        """
        model.to(context.device)
        model.eval()

        # Create test data loader
        batch_size = config.get("batch_size", 32)

        # Handle different sampler types properly
        if sampler is not None:
            if isinstance(sampler, torch.utils.data.Sampler):
                # It's already a PyTorch Sampler object
                sampler_obj = sampler
            elif isinstance(sampler, (list, range)):
                # It's a list of indices, create SubsetRandomSampler
                sampler_obj = torch.utils.data.SubsetRandomSampler(sampler)
            elif hasattr(sampler, "get"):
                # It's a Plato Sampler, call get() to obtain PyTorch sampler
                sampler_obj = sampler.get()
            else:
                # Unknown type, try to use it directly
                sampler_obj = sampler
        else:
            sampler_obj = None

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, sampler=sampler_obj
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = (
                    examples.to(context.device),
                    labels.to(context.device),
                )

                outputs = model(examples)
                _, predicted = torch.max(outputs.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total if total > 0 else 0

        # Log results
        if context.client_id == 0:
            logging.info(
                "[Server #%d] Test accuracy: %.2f%%",
                os.getpid(),
                100 * accuracy,
            )
        else:
            logging.info(
                "[Client #%d] Test accuracy: %.2f%%",
                context.client_id,
                100 * accuracy,
            )

        return accuracy
