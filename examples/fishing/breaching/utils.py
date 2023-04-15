"""System utilities."""

import socket
import sys

import os
import csv
import yaml

import torch
import torchvision

import random
import numpy as np
import datetime

import hydra
from omegaconf import OmegaConf, open_dict

import logging


def system_startup(process_idx, local_group_size, cfg):
    """Decide and print GPU / CPU / hostname info. Generate local distributed setting if running in distr. mode."""
    log = get_log(cfg)
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    torch.multiprocessing.set_sharing_strategy(cfg.case.impl.sharing_strategy)
    huggingface_offline_mode(cfg.case.impl.enable_huggingface_offline_mode)
    # 100% reproducibility?
    if cfg.case.impl.deterministic:
        set_deterministic()
    if cfg.seed is not None:
        set_random_seed(cfg.seed + 10 * process_idx)

    dtype = getattr(torch, cfg.case.impl.dtype)  # :> dont mess this up

    if cfg.case.impl.enable_gpu_acc and torch.cuda.is_available():
        device = torch.device(f"cuda:{process_idx}")
    else:
        device = torch.device("cpu")
    setup = dict(device=device, dtype=dtype)
    python_version = sys.version.split(" (")[0]
    log.info(f"Platform: {sys.platform}, Python: {python_version}, PyTorch: {torch.__version__}")
    log.info(f"CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}.")

    if cfg.case.impl.enable_gpu_acc and torch.cuda.is_available():
        torch.cuda.set_device(process_idx)
        log.info(f"GPU : {torch.cuda.get_device_name(device=device)}")

    # if not torch.cuda.is_available() and not cfg.dryrun:
    #     raise ValueError('No GPU allocated to this process. Running in CPU-mode is likely a bad idea. Complain to your admin.')

    return setup


def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def get_log(cfg, name=os.path.basename(__file__)):
    """Solution via https://github.com/facebookresearch/hydra/issues/1126#issuecomment-727826513"""
    if is_main_process():
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg, resolve=True))
        logger = logging.getLogger(name)
    else:

        def logger(*args, **kwargs):
            pass

        logger.info = logger
    return logger


def initialize_multiprocess_log(cfg):
    with open_dict(cfg):
        # manually save log config to cfg
        log_config = hydra.core.hydra_config.HydraConfig.get().job_logging
        # but resolve any filenames
        cfg.job_logging_cfg = OmegaConf.to_container(log_config, resolve=True)
        cfg.original_cwd = hydra.utils.get_original_cwd()


def save_summary(cfg, metrics, stats, local_time, original_cwd=True, table_name="breach"):
    """Save two summary tables. A detailed table of iterations/loss+acc and a summary of the end results."""
    # 1) detailed table:
    for step in range(len(stats["train_loss"])):
        iteration = dict()
        for key in stats:
            iteration[key] = stats[key][step] if step < len(stats[key]) else None
        save_to_table(".", f"{cfg.attack.type}_convergence_results", dryrun=cfg.dryrun, **iteration)

    try:
        local_folder = os.getcwd().split("outputs/")[1]
    except IndexError:
        local_folder = ""

    # 2) save a reduced summary
    summary = dict(
        name=cfg.name,
        usecase=cfg.case.name,
        model=cfg.case.model,
        datapoints=cfg.case.user.num_data_points,
        model_state=cfg.case.server.model_state,
        attack=cfg.attack.type,
        attacktype=cfg.attack.attack_type,
        **{k: v for k, v in metrics.items() if k != "order"},
        score=stats["opt_value"],
        total_time=str(datetime.timedelta(seconds=local_time)).replace(",", ""),
        user_type=cfg.case.user.user_type,
        gradient_noise=cfg.case.user.local_diff_privacy.gradient_noise,
        seed=cfg.seed,
        # dump extra values from here:
        **{f"ATK_{k}": v for k, v in cfg.attack.items()},
        **{k: v for k, v in cfg.case.items() if k not in ["name", "model"]},
        folder=local_folder,
    )

    location = os.path.join(cfg.original_cwd, "tables") if original_cwd else "tables"
    save_to_table(location, f"{table_name}_{cfg.case.name}_{cfg.case.data.name}_reports", dryrun=cfg.dryrun, **summary)


def save_to_table(out_dir, table_name, dryrun, **kwargs):
    """Save keys to .csv files. Function adapted from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f"table_{table_name}.csv")
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)  # noqa  # this line is testing the header
            # assert header == fieldnames[:len(header)]  # new columns are ok, but old columns need to be consistent
            # dont test, always write when in doubt to prevent erroneous table rewrites
    except Exception as e:  # noqa
        if not dryrun:
            # print('Creating a new .csv table...')
            with open(fname, "w") as f:
                writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
                writer.writeheader()
        else:
            pass
            # print(f'Would create new .csv table {fname}.')

    # Write a new row
    if not dryrun:
        # Add row for this experiment
        with open(fname, "a") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            writer.writerow(kwargs)
        # print('\nResults saved to ' + fname + '.')
    else:
        pass
        # print(f'Would save results to {fname}.')


def set_random_seed(seed=233):
    """."""
    torch.manual_seed(seed + 1)
    torch.cuda.manual_seed(seed + 2)
    torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    # Can't be too careful :>


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def avg_n_dicts(dicts):
    """https://github.com/wronnyhuang/metapoison/blob/master/utils.py."""
    # given a list of dicts with the same exact schema, return a single dict with same schema whose values are the
    # key-wise average over all input dicts
    means = {}
    for dic in dicts:
        for key in dic:
            if key not in means:
                if isinstance(dic[key], list):
                    means[key] = [0 for entry in dic[key]]
                else:
                    means[key] = 0
            if isinstance(dic[key], list):
                for idx, entry in enumerate(dic[key]):
                    means[key][idx] += entry / len(dicts)
            else:
                means[key] += dic[key] / len(dicts)
    return means


def get_base_cwd():
    try:
        return hydra.utils.get_original_cwd()
    except ValueError:  # Hydra not initialized:
        return os.getcwd()


def overview(server, user, attacker):   # NOTE(dchu): FISHING
    num_params, num_buffers = (
        sum([p.numel() for p in user.model.parameters()]),
        sum([b.numel() for b in user.model.buffers()]),
    )
    target_information = user.num_data_points * torch.as_tensor(server.cfg_data.shape).prod()
    print(f"Model architecture {user.model.name} loaded with {num_params:,} parameters and {num_buffers:,} buffers.")
    print(
        f"Overall this is a data ratio of {server.num_queries * num_params / target_information:7.0f}:1 "
        f"for target shape {[user.num_data_points, *server.cfg_data.shape]} given that num_queries={server.num_queries}."
    )
    print(user)
    print(server)
    print(attacker)


def save_reconstruction(
    reconstructed_user_data, server_payload, true_user_data, cfg, side_by_side=True, target_indx=None
):
    """If target_indx is not None, only the datapoints at target_indx will be saved to file."""
    os.makedirs("reconstructions", exist_ok=True)
    metadata = server_payload[0]["metadata"]
    if metadata["modality"] == "text":
        from breaching.cases.data.datasets_text import _get_tokenizer

        tokenizer = _get_tokenizer(
            server_payload[0]["metadata"]["tokenizer"],
            server_payload[0]["metadata"]["vocab_size"],
            cache_dir=cfg.case.data.path,
        )
        text_rec = tokenizer.batch_decode(reconstructed_user_data["data"])
        text_ref = tokenizer.batch_decode(true_user_data["data"])
        if target_indx is not None:
            text_rec = text_rec[target_indx]
            text_ref = text_ref[target_indx]

        filepath = os.path.join(
            "reconstructions", f"text_rec_{cfg.case.data.name}_{cfg.case.model}_user{cfg.case.user.user_idx}.txt"
        )

        with open(filepath, "w") as f:
            f.writelines(text_rec)
            if side_by_side:
                f.write("\n")
                f.write("========== GROUND TRUTH TEXT ===========")
                f.write("\n")
                f.writelines(text_ref)
    else:
        if hasattr(metadata, "mean"):
            dm = torch.as_tensor(metadata.mean)[None, :, None, None]
            ds = torch.as_tensor(metadata.std)[None, :, None, None]
        else:
            dm, ds = torch.tensor(
                0,
            ), torch.tensor(1)

        rec_denormalized = torch.clamp(reconstructed_user_data["data"].cpu() * ds + dm, 0, 1)
        ground_truth_denormalized = torch.clamp(true_user_data["data"].cpu() * ds + dm, 0, 1)
        if target_indx is not None:
            rec_denormalized = rec_denormalized[target_indx]
            ground_truth_denormalized = ground_truth_denormalized[target_indx]

        filepath = os.path.join(
            "reconstructions",
            f"img_rec_{cfg.case.data.name}_{cfg.case.model}_user{cfg.case.user.user_idx}.png",
        )

        if not side_by_side:
            torchvision.utils.save_image(rec_denormalized, filepath)
        else:
            torchvision.utils.save_image(torch.cat([rec_denormalized, ground_truth_denormalized]), filepath)


def dump_metrics(cfg, metrics):
    """Simple yaml dump of metric values."""

    filepath = f"metrics_{cfg.case.data.name}_{cfg.case.model}_user{cfg.case.user.user_idx}.yaml"
    sanitized_metrics = dict()
    for metric, val in metrics.items():
        try:
            sanitized_metrics[metric] = np.asarray(val).item()
        except ValueError:
            sanitized_metrics[metric] = np.asarray(val).tolist()
    with open(filepath, "w") as yaml_file:
        yaml.dump(sanitized_metrics, yaml_file, default_flow_style=False)


def huggingface_offline_mode(huggingface_offline_mode):
    if huggingface_offline_mode:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
