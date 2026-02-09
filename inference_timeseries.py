#!/usr/bin/env python3
"""
Time Series Inference Script for PatchTSMixer Models

Generates forecasts from trained checkpoints in two modes:
1. Future Forecast: Predict next time window after latest available data
2. Test Evaluation: Evaluate model performance on test set

Usage:
    # Future forecast
    uv run inference_timeseries.py -c <config.toml> --future

    # Test evaluation
    uv run inference_timeseries.py -c <config.toml> --num_samples 100
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch

# Add plato to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.models import registry as models_registry
from plato.serialization.safetensor import deserialize_tree

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Checkpoint Loading

def load_checkpoint(checkpoint_path, model):
    """Load model weights from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logging.info(f"Loading checkpoint from: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        serialized = f.read()

    state_dict_raw = deserialize_tree(serialized)
    if not isinstance(state_dict_raw, dict):
        raise TypeError("Deserialized state dict is not a mapping.")

    state_dict = OrderedDict(state_dict_raw.items())
    model.load_state_dict(state_dict, strict=True)

    logging.info("Checkpoint loaded successfully")
    return model


def find_latest_checkpoint(config):
    """Find the latest checkpoint file from the checkpoint directory."""
    checkpoint_dir = config.params.get("checkpoint_path")
    model_name = config.trainer.model_name

    if not os.path.exists(checkpoint_dir):
        return None

    # Look for checkpoint files: checkpoint_{model_name}_{round}.safetensors
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith(f"checkpoint_{model_name}_") and f.endswith(".safetensors")
    ]

    if not checkpoint_files:
        return None

    # Extract round numbers and find the latest
    checkpoint_rounds = []
    for f in checkpoint_files:
        try:
            round_num = int(f.replace(f"checkpoint_{model_name}_", "").replace(".safetensors", ""))
            checkpoint_rounds.append((round_num, f))
        except ValueError:
            continue

    if not checkpoint_rounds:
        return None

    checkpoint_rounds.sort(reverse=True)
    latest_round, latest_file = checkpoint_rounds[0]
    checkpoint_path = os.path.join(checkpoint_dir, latest_file)

    logging.info(f"Using latest checkpoint from round {latest_round}: {latest_file}")
    return checkpoint_path

# Future Forecast Mode
def run_future_forecast(model, datasource, device, config):
    """
    Generate forecast for the next time window after the latest available data.

    Returns:
        prediction: (prediction_length, num_channels)
        context: (context_length, num_channels)
    """
    model.to(device)
    model.eval()

    # Get full normalized time series data
    if hasattr(datasource, 'normalized_data'):
        full_data = torch.FloatTensor(datasource.normalized_data)
    elif hasattr(datasource, 'data'):
        full_data = torch.FloatTensor(datasource.data) if not torch.is_tensor(datasource.data) else datasource.data
    else:
        testset = datasource.get_test_set()
        if hasattr(testset, 'data'):
            full_data = testset.data
        else:
            raise RuntimeError("Cannot access time series data from datasource")

    context_length = config.trainer.context_length

    if len(full_data) < context_length:
        raise ValueError(f"Not enough data: need {context_length} timesteps, have {len(full_data)}")

    # Use the last context_length timesteps as input
    context = full_data[-context_length:]

    logging.info(f"Using latest {context_length} timesteps as context")
    logging.info(f"Context range: index {len(full_data) - context_length} to {len(full_data) - 1}")

    # Run inference
    with torch.no_grad():
        context_tensor = context.unsqueeze(0).to(device)
        outputs = model(past_values=context_tensor)

        # Extract predictions
        if hasattr(outputs, "prediction_outputs"):
            preds = outputs.prediction_outputs
        elif hasattr(outputs, "logits"):
            preds = outputs.logits
        else:
            preds = outputs[0]

    prediction = preds.cpu().numpy()[0]
    context_np = context.cpu().numpy()

    logging.info(f"Generated future forecast: shape {prediction.shape}")
    return prediction, context_np


def save_future_forecast(prediction, context, output_dir, config):
    """Save future forecast and context to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Get channel names
    all_channel_names = get_channel_names(config, prediction.shape[-1])

    prediction_channel_indices = getattr(config.trainer, "prediction_channel_indices", None)
    if prediction_channel_indices is not None:
        pred_channel_names = [all_channel_names[i] for i in prediction_channel_indices]
    else:
        pred_channel_names = all_channel_names[:prediction.shape[-1]]

    # Save forecast
    csv_data = {"timestep": list(range(prediction.shape[0]))}
    for idx, ch_name in enumerate(pred_channel_names):
        csv_data[ch_name] = prediction[:, idx]

    df = pd.DataFrame(csv_data)
    forecast_file = os.path.join(output_dir, "future_forecast.csv")
    df.to_csv(forecast_file, index=False)

    logging.info(f"Saved future forecast to {forecast_file}")
    logging.info(f"  Shape: {prediction.shape} (timesteps x channels)")
    logging.info(f"  Channels: {', '.join(pred_channel_names)}")

    # Save context
    csv_data_context = {"timestep": list(range(-context.shape[0], 0))}
    for idx, ch_name in enumerate(all_channel_names):
        if idx < context.shape[-1]:
            csv_data_context[ch_name] = context[:, idx]

    df_context = pd.DataFrame(csv_data_context)
    context_file = os.path.join(output_dir, "context_data.csv")
    df_context.to_csv(context_file, index=False)

    logging.info(f"Saved context data to {context_file}")
    logging.info(f"  Shape: {context.shape} (timesteps x channels)")


# Test Set Evaluation Mode
def run_test_evaluation(model, testset, device, num_samples=None):
    """
    Run inference on test set samples.

    Returns:
        predictions: (num_samples, prediction_length, num_channels)
        ground_truth: (num_samples, prediction_length, num_channels)
        past_values: (num_samples, context_length, num_channels)
    """
    model.to(device)
    model.eval()

    num_samples = min(num_samples, len(testset)) if num_samples else len(testset)

    predictions_list = []
    ground_truth_list = []
    past_values_list = []

    logging.info(f"Generating forecasts for {num_samples} test samples...")

    with torch.no_grad():
        for i in range(num_samples):
            sample = testset[i]
            past_values = sample["past_values"].unsqueeze(0).to(device)
            future_values = sample["future_values"].unsqueeze(0).to(device)

            outputs = model(past_values=past_values)

            # Extract predictions
            if hasattr(outputs, "prediction_outputs"):
                preds = outputs.prediction_outputs
            elif hasattr(outputs, "logits"):
                preds = outputs.logits
            else:
                preds = outputs[0]

            predictions_list.append(preds.cpu().numpy())
            ground_truth_list.append(future_values.cpu().numpy())
            past_values_list.append(past_values.cpu().numpy())

            if (i + 1) % 100 == 0:
                logging.info(f"  Processed {i + 1}/{num_samples} samples")

    predictions = np.concatenate(predictions_list, axis=0)
    ground_truth = np.concatenate(ground_truth_list, axis=0)
    past_values = np.concatenate(past_values_list, axis=0)

    logging.info("Inference completed")
    return predictions, ground_truth, past_values


def save_test_evaluation(predictions, ground_truth, output_dir, config):
    """Save test evaluation results: forecasts + metrics."""
    os.makedirs(output_dir, exist_ok=True)

    # Get channel names
    all_channel_names = get_channel_names(config, ground_truth.shape[-1])

    prediction_channel_indices = getattr(config.trainer, "prediction_channel_indices", None)
    if prediction_channel_indices is not None:
        pred_channel_indices = list(prediction_channel_indices)
        pred_channel_names = [all_channel_names[i] for i in pred_channel_indices]
        ground_truth_eval = ground_truth[..., pred_channel_indices]
    else:
        pred_channel_indices = list(range(predictions.shape[-1]))
        pred_channel_names = all_channel_names[:predictions.shape[-1]]
        ground_truth_eval = ground_truth

    # Compute metrics
    mse = np.mean((predictions - ground_truth_eval) ** 2)
    mae = np.mean(np.abs(predictions - ground_truth_eval))
    rmse = np.sqrt(mse)

    channel_metrics = {}
    for idx, ch_name in enumerate(pred_channel_names):
        pred_ch = predictions[..., idx]
        gt_ch = ground_truth_eval[..., idx]
        channel_metrics[ch_name] = {
            'mse': np.mean((pred_ch - gt_ch) ** 2),
            'mae': np.mean(np.abs(pred_ch - gt_ch)),
            'rmse': np.sqrt(np.mean((pred_ch - gt_ch) ** 2))
        }

    # Save metrics
    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write("Overall Metrics:\n")
        f.write(f"  MSE:  {mse:.6f}\n")
        f.write(f"  MAE:  {mae:.6f}\n")
        f.write(f"  RMSE: {rmse:.6f}\n")
        f.write("\nPer-Channel Metrics:\n")
        for ch_name, metrics in channel_metrics.items():
            f.write(f"  {ch_name}:\n")
            f.write(f"    MSE:  {metrics['mse']:.6f}\n")
            f.write(f"    MAE:  {metrics['mae']:.6f}\n")
            f.write(f"    RMSE: {metrics['rmse']:.6f}\n")

    logging.info(f"Metrics saved to {metrics_file}")
    logging.info(f"  Overall MSE: {mse:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    # Save all forecast samples as CSV
    logging.info(f"Saving {predictions.shape[0]} forecast samples to CSV...")

    for sample_idx in range(predictions.shape[0]):
        csv_data = {"timestep": list(range(predictions.shape[1]))}

        for idx, ch_name in enumerate(pred_channel_names):
            csv_data[f"{ch_name}_pred"] = predictions[sample_idx, :, idx]
            csv_data[f"{ch_name}_true"] = ground_truth_eval[sample_idx, :, idx]

        df = pd.DataFrame(csv_data)
        csv_file = os.path.join(output_dir, f"forecast_sample_{sample_idx:05d}.csv")
        df.to_csv(csv_file, index=False)

        if (sample_idx + 1) % 1000 == 0:
            logging.info(f"  Saved {sample_idx + 1}/{predictions.shape[0]} samples")

    logging.info(f"All {predictions.shape[0]} forecast samples saved")



def get_channel_names(config, num_channels):
    """Get channel names from datasource configuration."""
    try:
        task_type = getattr(config.data, "task_type", "unknown")
        from plato.datasources.openmeteo import DataSource
        if task_type in DataSource.TASK_CONFIGS:
            return DataSource.TASK_CONFIGS[task_type]["variables"]
    except:
        pass
    return [f"channel_{i}" for i in range(num_channels)]



def main():
    parser = argparse.ArgumentParser(
        description="Generate forecasts from trained PatchTSMixer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Future forecast
  uv run inference_timeseries.py -c config.toml --future

  # Test evaluation
  uv run inference_timeseries.py -c config.toml --num_samples 100
        """
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to configuration TOML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint file (default: uses latest)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="forecasts",
        help="Output directory (default: forecasts/)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test samples (default: all)"
    )
    parser.add_argument(
        "--future",
        action="store_true",
        help="Generate future forecast after latest data"
    )

    args = parser.parse_args()

    # Load configuration
    logging.info(f"Loading configuration from: {args.config}")
    original_argv = sys.argv.copy()
    sys.argv = ["inference_timeseries.py", "-c", args.config]
    config = Config()
    sys.argv = original_argv

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = find_latest_checkpoint(config)

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        checkpoint_dir = config.params.get("checkpoint_path", "N/A")
        logging.error(f"Checkpoint not found in: {checkpoint_dir}")
        logging.info("Please train the model first or specify --checkpoint")
        sys.exit(1)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model and checkpoint
    logging.info("Initializing model...")
    model = models_registry.get()
    model = load_checkpoint(checkpoint_path, model)

    # Load data
    logging.info("Loading data...")
    datasource = datasources_registry.get()

    # Prepare output directory
    output_dir = args.output
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.getcwd(), output_dir)

    # Run inference
    if args.future:
        logging.info("=" * 60)
        logging.info("FUTURE FORECAST MODE")
        logging.info("=" * 60)
        prediction, context = run_future_forecast(model, datasource, device, config)
        save_future_forecast(prediction, context, output_dir, config)
    else:
        logging.info("=" * 60)
        logging.info("TEST SET EVALUATION MODE")
        logging.info("=" * 60)
        testset = datasource.get_test_set()
        logging.info(f"Test set size: {len(testset)} samples")

        predictions, ground_truth, _ = run_test_evaluation(
            model, testset, device, num_samples=args.num_samples
        )
        save_test_evaluation(predictions, ground_truth, output_dir, config)

    logging.info("\n" + "=" * 60)
    logging.info("INFERENCE COMPLETED SUCCESSFULLY")
    logging.info(f"Results saved to: {output_dir}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
