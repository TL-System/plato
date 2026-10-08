# Digital Research Alliance of Canada

## Prepare a checkout

This is a preparation and Slurm job template for Python 3.13. Adapt the cluster
hostname, allocation account, storage paths, module stack, and requested
resources to your site. Consult the Alliance [available software](https://docs.alliancecan.ca/wiki/Available_software),
[Python](https://docs.alliancecan.ca/wiki/Python), and
[running jobs](https://docs.alliancecan.ca/wiki/Running_jobs) documentation.
Site references include [Narval](https://docs.alliancecan.ca/wiki/Narval) and
[Rorqual](https://docs.alliancecan.ca/wiki/Rorqual/en); check the destination
and resource instructions for your allocation.

Current module availability, Plato package/wheel and CUDA compatibility, offline
assets, allocation policy, and actual cluster execution have not been verified
for this template. Direct retrieval of those official pages during this refresh
returned BotStopper Access Denied; this guide does not infer a current site-wide
installation procedure or compute-node networking policy from them.

Use your existing [CCDB account](https://ccdb.computecanada.ca/) and an approved
project or scratch location. Replace every angle-bracket placeholder before
running the following commands:

```bash
ssh <username>@<cluster-hostname>
cd <approved-project-directory>
git clone https://github.com/TL-System/plato.git
cd plato
```

## Prepare Python and dependencies before submission

Discover the Python modules available on the target cluster:

```bash
module avail python
module spider python
```

Select an available Python 3.13 module and any prerequisite modules required by
the site. Use the same module stack when preparing the environment and running
the job; no fixed module name is assumed here.

```bash
module load <site-python-3.13-module>
python -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
```

Make uv 0.12.22 available using the site's approved installation procedure.
Provision the locked environment on a site-approved preparation host before
submission, from the checkout root:

```bash
uv --version
PLATO_CLUSTER_PYTHON="$(command -v python)"
UV_PYTHON_DOWNLOADS=never uv sync --locked --python "$PLATO_CLUSTER_PYTHON"
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
```

The final interpreter check catches an existing environment with the wrong
Python version. Use a dedicated checkout/environment for a different module
stack or dependency selection. Confirm package and accelerator compatibility
on the target system before committing substantial resources.

The base command above selects the default dependencies. Follow
[Installation](install.md) for extras and workspace members; provision the
workload's selected dependencies before submission. For example, Lighteval
requires the locked `llm_eval` extra and NLTK resources, and `ssl` uses a
separate environment.

## Prepare datasets and model assets

Provision datasets, model weights, tokenizers, and other required assets on a
site-approved host with the needed access, before submitting the training job.
Use the family recipe's explicit preparation steps and point the config at the
prepared paths. Keep those paths accessible from the allocated compute node.

The [Qwen3 reference](examples/case-studies/6. Qwen3 Federated LoRA.md) describes
its pinned model and local data. The
[Lighteval installation steps](install.md#optional-server-side-llm-evaluation-with-lighteval)
and [runtime qualification](references/evaluators.md#optional-runtime-qualification)
describe its resources and bounded checks. These guides do not establish
Alliance cluster compatibility or a complete offline Hugging Face workflow.

For the illustrative MNIST job below, prepare MNIST under the configured data
path before submission. Check the selected config's data and output paths
against your storage allocation. Running training on a login node and
interrupting it is not an asset-preparation recipe.

## Create a batch script

Save the following as `plato_job.sh`, replacing the module, account, and
absolute checkout path. The CPU MNIST command is illustrative; adjust time,
memory, CPU/GPU requests, config, and device flags to the workload and site
policy. CUDA jobs need a compatible module/package stack and site-specific GPU
resource requests; consult the official job documentation.

```bash
#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=<allocation-account>
#SBATCH --output=plato-%j.out

set -euo pipefail
module load <site-python-3.13-module>
cd <absolute-path-to-plato-checkout>
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
exec .venv/bin/python plato.py --config configs/MNIST/fedavg_lenet5.toml --cpu
```

The batch command uses the already provisioned environment directly. Dependency
resolution, package installation, and asset preparation belong before
submission. The selected workload must also be configured to use its prepared
assets; calling the interpreter directly does not prevent a data source from
attempting downloads when assets are missing.

## Manage the job

After completing preparation and adapting the script, submit it and retain its
job ID:

```bash
sbatch plato_job.sh
```

Use the specific ID returned by Slurm to inspect status and output:

```bash
squeue --jobs=<job-id>
sacct --jobs=<job-id>
tail -f plato-<job-id>.out
```

Stop output monitoring with `Ctrl-C`. To cancel that training job:

```bash
scancel <job-id>
```

See the Alliance [job-management material](https://docs.alliancecan.ca/mediawiki/images/7/77/Managing_jobs.pdf)
for the roles of these commands. Submission and cancellation are shown as user
instructions; neither was executed to validate this guide.

For interactive debugging, consult the site's
[interactive job instructions](https://docs.alliancecan.ca/wiki/Running_jobs#Interactive_jobs)
and obtain an allocation with the required resources. On the allocated compute
node, load the same modules, enter the checkout, and use the prepared
`.venv/bin/python` command. Release the allocation when finished.

!!! tip "Concurrent sessions"
    Use distinct `server.port` values for concurrent sessions that may share a
    node. Matching bind addresses and ports can cause an address-in-use error.

## Troubleshooting

!!! tip "Out of CUDA memory"
    Decrease `trainer.max_concurrency` in the selected configuration and inspect
    the failed session's logs and allocated resources.

!!! tip "An earlier session is still running"
    Stop the identified interactive training session with `Ctrl-C`, or cancel
    its specific scheduler job ID with `scancel`. Verify that the session has
    released its resources before restarting.

!!! tip "Client timeout"
    Check for failed or memory-constrained clients before attributing a missing
    client to a slow server response. The default `server.ping_timeout` is
    3600 seconds. Measure delays in the affected session before choosing an
    override in the `server` configuration section. See
    [Miscellaneous Notes](misc.md#potential-runtime-errors) for related guidance.
