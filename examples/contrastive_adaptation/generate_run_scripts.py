"""
Generate the running scripts for the Sim Server
based on the implemented method and configs

One Example:

#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=21
#SBATCH --gres=gpu:3
#SBATCH --mem=120G
#SBATCH --output=<output_filename>.out

./run -c configs/CIFAR10/fedavg_resnet18.yml -b /data/bli/plato

"""

import os
import stat

current_path = "./"

sim_data_path = "/data/sijia/INFOCOM23/experiments"

desire_python = "/home/sijia/envs/miniconda3/envs/INFOCOM23/bin/python"

methods_root_dir = os.path.join(current_path, "examples",
                                "contrastive_adaptation")

configs_file_dir = os.path.join(methods_root_dir, "configs")
script_files_dir = os.path.join(methods_root_dir, "run_scripts")
sbatch_output_dir = os.path.join(current_path, "slurm_loggings")


def extract_method_file(methods_root_dir, config_file_name):
    """ Extract the Python file path of the corresponding method. """
    method_name = config_file_name.split("_")[0]
    method_python_file_path = os.path.join(methods_root_dir, method_name,
                                           method_name + ".py")

    return method_python_file_path


def create_run_script(methods_root_dir,
                      config_files_dir,
                      config_file_name,
                      sbatch_logging_dir,
                      script_save_dir,
                      extension="sh"):
    """ Create the experiment running scripts

    methods_root_dir (str): the root dir where the code for implemented methods is saved
    config_file_name (str): the name of the config file to be run
    sbatch_logging_dir (str): where to save the output file of the slurm' sbatch
    script_save_dir (str): where to save the created scripts
    extension (str): the extension of the created scripts, default ".sh
    """
    header = "#!/bin/bash"
    time_line = "#SBATCH --time=5:00:00"
    cpus_line = "#SBATCH --cpus-per-task=21"
    gpu_line = "#SBATCH --gres=gpu:1"
    mem_line = "#SBATCH --mem=100G"

    file_name_no_extension = config_file_name.split(".")[0]

    output_file_path = os.path.join(sbatch_logging_dir, file_name_no_extension)
    output_line = "#SBATCH --output=%s.out" % output_file_path

    method_code_file = extract_method_file(methods_root_dir, config_file_name)
    config_file_path = os.path.join(config_files_dir, config_file_name)
    run_code_line = "%s %s -c %s -b %s" % (desire_python, method_code_file,
                                           config_file_path, sim_data_path)

    content = "%s \n%s \n%s \n%s \n%s \n%s \n \n%s \n" % (
        header, time_line, cpus_line, gpu_line, mem_line, output_line,
        run_code_line)

    script_file_name = '{}.{}'.format(file_name_no_extension, extension)
    script_file_save_path = os.path.join(script_save_dir, script_file_name)

    os.makedirs(script_save_dir, exist_ok=True)

    print(("Creating sbatch running script {} for config file {}, ").format(
        script_file_save_path, config_file_name))

    if os.path.exists(script_file_save_path):
        print(
            ("\nAlready existed, skipping...\n").format(script_file_save_path))
    else:
        with open(script_file_save_path, 'w') as script_file:
            script_file.write(content)

    # add
    #   - stat.S_IRWXU: Read, write, and execute by owner 7
    #   - stat.S_IRGRP : Read by group
    #   - stat.S_IXGRP : Execute by group
    #   - stat.S_IROTH : Read by others
    #   - stat.S_IXOTH : Execute by others
    os.chmod(
        script_file_save_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP
        | stat.S_IROTH | stat.S_IXOTH)


if __name__ == "__main__":
    all_config_files_name = os.listdir(configs_file_dir)

    # create the output dir as the slurm will not create it
    # for the experiments
    os.makedirs(sbatch_output_dir, exist_ok=True)

    for config_file_name in all_config_files_name:
        create_run_script(methods_root_dir,
                          config_files_dir=configs_file_dir,
                          config_file_name=config_file_name,
                          sbatch_logging_dir=sbatch_output_dir,
                          script_save_dir=script_files_dir)
