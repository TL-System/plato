"""Run one reviewed integration profile and retain exact execution provenance."""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--commit', required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--gate', type=Path, required=True)
    args = parser.parse_args()
    plan_path = Path('/tmp/plato-phase2-root/integration-plan.json')
    plan = json.loads(plan_path.read_text())
    root = Path(plan['root'])
    profile = next(p for p in plan['profiles'] if p['id'] == args.profile)
    # Root supplies the exact reviewed gate; preserve its bytes without guessing schema.
    gate_bytes = args.gate.read_bytes()
    gate = json.loads(gate_bytes)
    if not isinstance(gate, dict):
        raise ValueError('Phase gate must be a JSON object.')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    if head != args.commit or len(args.commit) != 40:
        raise ValueError(f'Expected exact integrated commit {args.commit}; found {head}')
    dirty = subprocess.check_output(['git', 'diff', 'HEAD', '--name-only'], cwd=root, text=True).splitlines()
    if any(not p.startswith('evidence/2026-refresh/') for p in dirty):
        raise ValueError(f'Implementation must match committed source: {dirty}')
    run = Path('/tmp/plato-phase2-integration/runs') / (args.profile + '-' + uuid.uuid4().hex[:12])
    run.mkdir(parents=True, exist_ok=False)
    (run / 'gate.json').write_bytes(gate_bytes)
    (run / 'plan.json').write_bytes(plan_path.read_bytes())
    env = os.environ.copy()
    for k, v in plan['environment_controls'].items():
        env[k] = v.replace('<profile.environment>', profile['environment'])
    env['COVERAGE_FILE'] = str(run / '.coverage')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env.pop('PYTHONPYCACHEPREFIX', None)
    receipt = {
        'profile': profile,
        'commit': head,
        'tree': subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=root, text=True).strip(),
        'lock_sha256': digest(root / 'uv.lock'),
        'gate_sha256': hashlib.sha256(gate_bytes).hexdigest(),
        'plan_sha256': digest(plan_path),
        'commands': [],
        'status': 'RUNNING',
        'run_directory': str(run),
    }
    def record():
        (run / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    record()
    def execute(name, command, timeout):
        entry = {'name': name, 'argv': command, 'cwd': str(root), 'log': str(run / (name + '.log'))}
        receipt['commands'].append(entry)
        record()
        started = time.monotonic()
        with open(entry['log'], 'wb') as log:
            try:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                entry['exit_code'] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                entry['exit_code'] = 124
                entry['timeout'] = True
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    if sig == signal.SIGTERM:
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            pass
                process.wait(timeout=10)
        entry['seconds'] = round(time.monotonic() - started, 3)
        entry['sha256'] = digest(Path(entry['log']))
        record()
        if entry['exit_code']:
            raise RuntimeError(f'{name} failed: {entry["exit_code"]}; {entry["log"]}')
    try:
        execute('sync', ['uv', 'sync', '--locked', '--python', profile['python'], '--no-default-groups', '--group', profile['group'], '--compile-bytecode'], 900)
        python = str(Path(profile['environment']) / 'bin/python')
        execute('python', [python, '--version'], 30)
        execute('packages', ['uv', 'pip', 'freeze', '--python', python], 120)
        origins = (
            "import importlib.util,json,pathlib,sys; "
            "names=['plato','torch','numpy','pytest']; "
            "specs={n:importlib.util.find_spec(n) for n in names}; "
            "out={n:{'origin':s.origin,'locations':list(s.submodule_search_locations or [])} for n,s in specs.items()}; "
            "print(json.dumps({'executable':sys.executable,'modules':out},indent=2)); "
            "assert pathlib.Path(out['plato']['origin']).resolve().is_relative_to(pathlib.Path.cwd().resolve())"
        )
        execute('import-origins', [python, '-c', origins], 30)
        test_args = [python, '-m', 'pytest', *profile['pytest_args'], '--basetemp=' + str(run / 'tmp'), '--junitxml=' + str(run / 'results.xml')]
        execute('pytest', test_args, 3600)
        xml = ET.parse(run / 'results.xml')
        counts = {key: 0 for key in ('tests', 'failures', 'errors', 'skipped')}
        cases = []
        for suite in xml.getroot().iter('testsuite'):
            for key in counts:
                counts[key] += int(suite.attrib.get(key, 0))
        for case in xml.getroot().iter('testcase'):
            for item in case:
                if item.tag in ('skipped', 'failure', 'error'):
                    cases.append({'test': case.attrib, 'outcome': item.tag, 'attributes': item.attrib, 'text': item.text})
        receipt['junit_counts'] = counts
        receipt['nonpassing_cases'] = cases
        receipt['status'] = 'EXECUTION_PASS_REQUIRES_ROOT_AUDIT'
    except Exception as exc:
        receipt['status'] = 'FAIL'
        receipt['error'] = str(exc)
        raise
    finally:
        receipt['final_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        receipt['final_lock_sha256'] = digest(root / 'uv.lock')
        if receipt['final_head'] != head or receipt['final_lock_sha256'] != receipt['lock_sha256']:
            receipt['status'] = 'SOURCE_CHANGED'
        record()
        print(json.dumps({'status': receipt['status'], 'receipt': str(run / 'receipt.json')}), flush=True)


if __name__ == '__main__':
    main()
