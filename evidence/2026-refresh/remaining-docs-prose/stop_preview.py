import json
import signal
import socket
from pathlib import Path

import psutil

root = Path('/tmp/plato-refresh-worktrees/remaining-docs-prose').resolve()
owned = []
for proc in psutil.process_iter(['pid', 'cmdline', 'cwd', 'create_time']):
    args = proc.info['cmdline'] or []
    if 'mkdocs' in args and 'serve' in args and proc.info['cwd'] and Path(proc.info['cwd']).resolve() == root:
        owned.append(proc)
assert len(owned) == 1, [p.info for p in owned]
proc = owned[0]
record = dict(proc.info)
proc.send_signal(signal.SIGINT)
record['exit_code'] = proc.wait(timeout=15)
record['alive_after'] = proc.is_running()
assert not record['alive_after']
with socket.socket() as sock:
    record['port_8000_connect_ex_after'] = sock.connect_ex(('127.0.0.1', 8000))
assert record['port_8000_connect_ex_after'] != 0
Path('/tmp/plato-remaining-docs-prose-proof/serve-shutdown.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
