#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --account=INERT_PLACEHOLDER
#SBATCH --output=plato-%j.out

set -euo pipefail
module load INERT_PLACEHOLDER
cd INERT_PLACEHOLDER
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 13), sys.version'
exec .venv/bin/python plato.py --config configs/MNIST/fedavg_lenet5.toml --cpu
