"""Bounded subprocess target for the shipped max_concurrency SCAFFOLD path."""

import json
import sys
from pathlib import Path

from tests.integration.test_scaffold_round_flow import (
    run_failed_worker_scenario,
    run_two_round_scenario,
)

if __name__ == "__main__":
    root, output = map(Path, sys.argv[1:3])
    root.mkdir(parents=True, exist_ok=True)
    result = (
        run_failed_worker_scenario(root, sys.argv[3])
        if len(sys.argv) > 3
        else run_two_round_scenario(root, spawn=True)
    )
    output.write_text(json.dumps(result, indent=2))
