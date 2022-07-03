"""
Generate the running scripts for the Sim Server
based on the implemented method and configs

One Example:

#!/bin/bash
#SBATCH --time=7:00:00
#SBATCH --cpus-per-task=21
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --output=./slurm_loggings/byol_CIFAR10_resnet18.out


Important:
    Before setting the resource requirement, please check the
exact resources on the server.

Dell Precision 7920 Tower Workstation,
    - 2 x Intel Xeon CPUs with 20 CPU cores each,
    - 1TB Intel NVMe PCIe SSD boot drive
    - 12TB data drive (two 6TB drives),
    - 256GB physical memory
    - 3 x NVIDIA RTX A4500 GPU
        20GB CUDA memory each.
#CPUs, lscpu


Just run:
    python examples/contrastive_adaptation/generate_run_scripts.py


"""

import os
import stat

import glob

current_path = "./"

sim_data_path = "/data/sijia/INFOCOM23/experiments"

desire_python = "/home/sijia/envs/miniconda3/envs/INFOCOM23/bin/python"

methods_root_dir = os.path.join(current_path, "examples",
                                "contrastive_adaptation")

config_files_root_dir = os.path.join(methods_root_dir, "configs")
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
    time_line = "#SBATCH --time=30:00:00"
    cpus_line = "#SBATCH --cpus-per-task=10"
    gpu_line = "#SBATCH --gres=gpu:1"
    mem_line = "#SBATCH --mem=32G"

    file_name_no_extension = config_file_name.split(".")[0]

    config_files_dir_name = os.path.basename(config_files_dir)
    output_file_dir = os.path.join(sbatch_logging_dir, config_files_dir_name)
    sim_data_dir_path = os.path.join(sim_data_path, config_files_dir_name)

    os.makedirs(output_file_dir, exist_ok=True)

    output_file_path = os.path.join(output_file_dir, file_name_no_extension)

    output_line = "#SBATCH --output=%s.out" % output_file_path

    method_code_file = extract_method_file(methods_root_dir, config_file_name)
    config_file_path = os.path.join(config_files_dir, config_file_name)
    run_code_line = "%s %s -c %s -b %s" % (desire_python, method_code_file,
                                           config_file_path, sim_data_dir_path)

    content = "%s \n%s \n%s \n%s \n%s \n%s \n \n%s \n" % (
        header, time_line, cpus_line, gpu_line, mem_line, output_line,
        run_code_line)

    script_file_name = '{}.{}'.format(file_name_no_extension, extension)
    script_file_save_path = os.path.join(script_save_dir, script_file_name)

    os.makedirs(script_save_dir, exist_ok=True)

    print(("Creating sbatch running script {} for config file {}/{}, ").format(
        script_file_save_path, config_files_dir, config_file_name))

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

    configs_files_path = glob.glob(
        os.path.join(config_files_root_dir, "*/", "*.yml"))

    # create the output dir as the slurm will not create it
    # for the experiments
    os.makedirs(sbatch_output_dir, exist_ok=True)

    for config_file_path in configs_files_path:
        # obtain the full path of the existed dir of the file
        # such as ./examples/contrastive_adaptation/configs/whole_global_model
        file_dir_path = os.path.dirname(config_file_path)
        # obtain the config file name
        file_name = os.path.basename(config_file_path)
        # obtain the dir name of the config file
        # such as whole_global_model
        file_dir_name = os.path.basename(file_dir_path)

        # save script to the same subdir with the same dir name
        # of the config
        to_save_dir = os.path.join(script_files_dir, file_dir_name)

        create_run_script(methods_root_dir,
                          config_files_dir=file_dir_path,
                          config_file_name=file_name,
                          sbatch_logging_dir=sbatch_output_dir,
                          script_save_dir=to_save_dir)
