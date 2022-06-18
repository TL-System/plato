"""
The implementation of sending the results obtained on the Sim server
 to the local computer.


The type of the experimental results should be provided by:
    - type

The type can be:
    - all, sending all logging/models/checkpoints/results to the local
    - logging, sending logging only
    - textlogging, sending text logging
    - models, sending models only
    - checkpoints, sending checkpoints only
    - results, sending results only

For example:

    python examples/contrastive_adaptation/extract_outputs_to_local.py -t models

    just extracting the models saved on Sim to the local
"""

import argparse
import os

current_path = "./"

sim_data_path = "/data/sijia/INFOCOM23/experiments"

sbatch_output_dir = os.path.join(current_path, "slurm_loggings")

local_experiments_dir = "/Users/sijiachen/Documents/Research-Area/paperInPreparation/MMFPLMasked/repo/code"
local_logging_dir = "/Users/sijiachen/Documents/Research-Area/paperInPreparation/MMFPLMasked/repo/code"

sim_data_dir = "sijia@sim.csl.toronto.edu:/data/sijia/INFOCOM23"
sim_logging_dir = "sijia@sim.csl.toronto.edu:/home/sijia/works/sijia-infocom23/plato/slurm_loggings"

models_dir_name = "models"
checkpoints_dir_name = "checkpoints"
results_dir_name = "results"
text_logging_name = "loggings"
data_folders_name = [
    models_dir_name, checkpoints_dir_name, results_dir_name, text_logging_name
]


def obtain_data_path(data_type):
    """ Obtain the data path required by the data type. """
    extract_sim_experiments_dirs = []
    extract_sim_logging_dir = None
    to_local_experiments_dir = None
    to_local_logging_dir = None

    if data_type == "all":

        extract_sim_experiments_dirs = [
            sim_data_dir + "/experiments/" + folder_name
            for folder_name in data_folders_name
        ]
        extract_sim_logging_dir = sim_logging_dir
        to_local_experiments_dir = local_experiments_dir + "/experiments"
        to_local_logging_dir = local_logging_dir

    elif data_type == "logging":
        extract_sim_logging_dir = sim_logging_dir
        to_local_logging_dir = local_logging_dir

    elif data_type == "textlogging":
        extract_sim_logging_dir = sim_data_dir + "/experiments/" + text_logging_name
        to_local_logging_dir = to_local_experiments_dir = local_experiments_dir + "/experiments"

    elif data_type == "models":
        extract_sim_experiments_dirs = [sim_data_dir + "/experiments/models"]
        to_local_experiments_dir = local_experiments_dir + "/experiments"
    elif data_type == "checkpoints":
        extract_sim_experiments_dirs = [
            sim_data_dir + "/experiments/checkpoints"
        ]
        to_local_experiments_dir = local_experiments_dir + "/experiments"
    elif data_type == "results":
        extract_sim_experiments_dirs = [sim_data_dir + "/experiments/results"]
        to_local_experiments_dir = local_experiments_dir + "/experiments"

    return extract_sim_experiments_dirs, extract_sim_logging_dir, to_local_experiments_dir, to_local_logging_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--type',
        type=str,
        default='all',
        help=
        'Extract what data all/logging/models/checkpoints/results from sim to local.'
    )

    args = parser.parse_args()

    data_type = args.type

    extract_sim_experiments_dirs, extract_sim_logging_dir, \
        to_local_experiments_dir, to_local_logging_dir = obtain_data_path(data_type)

    if extract_sim_experiments_dirs:
        print(
            f"Extracting the {extract_sim_experiments_dirs} from sim to local {to_local_experiments_dir}"
        )
        os.makedirs(to_local_experiments_dir, exist_ok=True)
        for extract_dir in extract_sim_experiments_dirs:
            os.system("scp -r %s %s" % (extract_dir, to_local_experiments_dir))

    if extract_sim_logging_dir is not None:
        print(
            f"Extracting the {extract_sim_logging_dir} from sim to local {to_local_logging_dir}"
        )
        os.makedirs(to_local_logging_dir, exist_ok=True)
        os.system("scp -r %s %s" %
                  (extract_sim_logging_dir, to_local_logging_dir))

# scp -r sijia@sim.csl.toronto.edu:/home/sijia/works/plato-projects/plato/results /Users/sijiachen/Documents/Research-Area/paperInPreparation/MMFPLMasked/repo/sim_results/
