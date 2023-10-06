"""
Upgrades all existing packages in the current conda environment.
"""
import subprocess
from importlib import metadata

for dist in metadata.distributions():
    if dist.metadata["Name"] is not None:
        dist_name = dist.metadata['Name']
        print(f"Upgrading package {dist_name}...")
        subprocess.call("pip install -U " + dist_name, shell=1)
