"""Guarded target for actual spawned FedDyn parent/child qualification."""

import json
import sys
from pathlib import Path

from tests.integration.test_feddyn_round_flow import run_partial

if __name__ == "__main__":
    root, output = map(Path, sys.argv[1:3])
    root.mkdir(parents=True, exist_ok=True)
    result = run_partial(
        root,
        sys.argv[3],
        tuple(map(int, sys.argv[4].split(","))),
        spawn=True,
        visits=tuple(map(int, sys.argv[5].split(",")))
        if len(sys.argv) > 5
        else (1, 2, 1),
    )
    output.write_text(json.dumps(result, indent=2))
