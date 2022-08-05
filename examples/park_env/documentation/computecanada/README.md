## Running the algorithms on Compute Canada

### Installation
The installation process can be done from referring to the running docs written [here](https://platodocs.netlify.app/running.html)\
This is a very important command
```shell
bash ./docs/patches/patch_cc.sh
```
After running this, there should be some patches to patch version differences. Run a ```git status``` to check if anything has been patched.
## Batch file configuration and location
**Note:** Refer to the examples provided in this folder for concreteness (they are configured to be ran on the narval cluster)
* Be sure to have the batch file in the plato directory and not underneath an examples directory
* Make sure the output file is algorithm/seed specific
* No need for ```--gres=gpu``` argument as ```#SBATCH --cpus-per-task=some number``` is the bottom neck. Having multiple gpus extends wait time considerably

### Some of our metrics when we were running jobs on compute canada:
**1.** **Time Taken: (--time=)**
   1. **6 clients with 6 per round**
      * around 20 hours, we put 30 hours as a safety margin
   2. **12 clients with 12 per round**
      * around X hours, we put Y hours as a safety margin
   3. **18 clients with 18 per round**
      * around 45 hours, we put 50 hours as a safety margin

**2.** **cpus-per-task=:**
   1. **6 clients with 6 per round**
      * 8
   2. **12 clients with 12 per round**
      * 36
   3. **18 clients with 18 per round**
      * 48

**3.** **--mem=:**
   1. **6 clients with 6 per round**
      * X
   2. **12 clients with 12 per round**
      * Y
   3. **18 clients with 18 per round**
      * Z

## Narval vs Graham vs Cedar
* Narval has the lowest wait time out of the three. 
* Narval & Graham cannot download datasets while running. For that reason, you must copy over your dataset before running the job. For example, if the configuration file has FashionMNIST requested as the dataset, you must first download FashionMNIST locally by running plato and then copy over these files onto your compute canada account
### More information on the three
* More information on Narval can be found [here](https://docs.alliancecan.ca/wiki/Narval/en)
* More information on Graham can be found [here](https://docs.alliancecan.ca/wiki/Graham)
* More information on Cedar can be found [here](https://docs.alliancecan.ca/wiki/Cedar)
* The status of each cluster can be found [here](https://status.computecanada.ca/)

## Running multiple jobs
* If multiple jobs with differnet algorithms are running, make sure each checkpoint, model folders have their respective algorithm names attached to them to prevent overwriting. Please refer to the documentation on configuration files to know which parameters to change to do this
* Different port numbers with each algorithm
* **HIGHLY RECOMMENDED:** Having multiple configuration files if one wants to run multiple algorithms and just changing the configuration file argument in the batch file to avoid changing configuration file each submission. Changing each configuration file will also lead to potential submission of the wrong version of a configuration file.

**Note:** One should also make sure that if the same algorithm with a different seed (i.e. seed 5 vs seed 10) are running concurrently that the respective checkpoint, model folders should have a seed number appended to the name to avoid seed 10 overwriting seed 5 for example
### Running abr_sim
* This does not take a configuration file but a seed name, change the X in the --seedX argument in ```cc_a2c_abr_sim.sh``` to whichever seed you would like.

For more information on the configuration of the configuration files, readers are directly to the configuration documentation in this directory

## If running jobs prompts errors related to pickle
**1.** Install newest version of torch. At the time this was written, the newest version was 1.12.0 and could be installled with this command:
```shell
pip3 install torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```
**2.** If problems continue
  * Request more cpus-per-task

**3.** Check no folders are being overwriten due to conflicting names
