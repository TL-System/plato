import hashlib
import json
import re
import subprocess
import sys
import textwrap
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

root = Path('/tmp/plato-refresh-worktrees/remaining-docs-prose')
proof = Path('/tmp/plato-remaining-docs-prose-proof')
assert sys.version_info[:2] == (3, 13)
paths = [
    'AGENTS.md', 'docs/README.md', 'docs/docs/ccdb.md',
    'docs/docs/configurations/server.md', 'docs/docs/configurations/trainer.md',
    'docs/docs/development.md', 'docs/docs/examples/Getting Started.md',
    'docs/docs/examples/algorithms/1. Server Aggregation Algorithms.md',
    'docs/docs/index.md', 'docs/docs/misc.md',
]
result = {'python': sys.version, 'changed_paths': paths, 'toml': [], 'shell': []}
for name in ['docs/docs/configurations/trainer.md', 'docs/docs/examples/algorithms/1. Server Aggregation Algorithms.md']:
    source = (root / name).read_text()
    fences = re.findall(r'```toml\n(.*?)```', source, re.S)
    selected = [f for f in fences if 'sched = ' in f or 'attention_model_path' in f]
    assert len(selected) == 1
    fragment = textwrap.dedent(selected[0])
    parsed = tomllib.loads(fragment)
    if 'trainer' in parsed:
        assert parsed['trainer'] == {'type': 'timm_basic', 'lr_scheduler': 'timm'}
        assert parsed['parameters']['learning_rate'] == {
            'sched': 'cosine', 'min_lr': 1e-6, 'warmup_lr': 0.0001,
            'warmup_epochs': 3, 'cooldown_epochs': 10,
        }
    else:
        assert parsed['algorithm'] == {
            'type': 'fedavg', 'scaling_factor': 10,
            'attention_model_path': './attention_model.pt', 'pca_components': 10,
            'threshold': 0.005, 'attention_loops': 5, 'attention_hidden': 32,
        }
        uncommented = fragment.replace('# dataset_capture_dir', 'dataset_capture_dir')
        assert tomllib.loads(uncommented)['algorithm']['dataset_capture_dir'] == './attack_adaptive_dataset'
    result['toml'].append({'path': name, 'parsed': parsed, 'fragment': fragment, 'capture_uncomment_parse': 'algorithm' in parsed})

ccdb = (root / 'docs/docs/ccdb.md').read_text()
for index, code in enumerate(re.findall(r'```bash\n(.*?)```', ccdb, re.S)):
    inert = re.sub(r'<[^>]+>', 'INERT_PLACEHOLDER', code)
    path = proof / f'ccdb-{index}.sh'
    path.write_text(inert)
    process = subprocess.run(['bash', '-n', str(path)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result['shell'].append({'index': index, 'template': code, 'substituted': inert, 'bash_n_exit': process.returncode})
batch = next(row['template'] for row in result['shell'] if row['template'].startswith('#!/bin/bash'))
assert not re.search(r'\b(uv|curl|wget|pip)\b', batch)
assert 'cd <absolute-path-to-plato-checkout>' in batch
assert (root / 'configs/MNIST/fedavg_lenet5.toml').is_file()
for name in ['docs/docs/ccdb.md', 'docs/docs/misc.md']:
    source = (root / name).read_text()
    assert '3600' in source and not re.search(r'\b360\b|pkill python|python/3\.12', source)
assert (root / 'tests/trainers/test_lr_scheduler_registry.py').is_file()
assert (root / 'tests/servers/test_fedavg_strategy.py').is_file()

class Page(HTMLParser):
    def __init__(self, content):
        super().__init__()
        self.ids, self.links = set(), []
        self.feed(content)
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for attr in ['href', 'src']:
            if attr in attrs:
                self.links.append(attrs[attr])

site = root / 'docs/site'
pages = {p.resolve(): Page(p.read_text()) for p in site.rglob('*.html')}
broken, count = [], 0
for path, page in pages.items():
    for link in page.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        destination = ((site / unquote(url.path).lstrip('/')) if url.path.startswith('/') else path.parent / unquote(url.path)).resolve() if url.path else path
        if destination.is_dir():
            destination /= 'index.html'
        count += 1
        if not destination.exists():
            broken.append({'page': str(path.relative_to(site.resolve())), 'link': link, 'reason': 'missing target'})
        elif url.fragment and destination in pages and unquote(url.fragment) not in pages[destination].ids:
            broken.append({'page': str(path.relative_to(site.resolve())), 'link': link, 'reason': 'missing anchor'})
result['rendered_links'] = {'html_pages': len(pages), 'links_checked': count, 'broken': broken}

md_broken, md_count = [], 0
for name in paths:
    for raw in re.findall(r'\]\(([^\n]+?)\)', (root / name).read_text()):
        link = raw.strip('<>')
        url = urlsplit(link)
        if url.scheme or url.netloc:
            continue
        md_count += 1
        destination = (root / name).parent / unquote(url.path)
        if url.path and not destination.exists():
            md_broken.append({'page': name, 'link': link})
        if url.fragment and destination.suffix == '.md':
            relative = destination.resolve().relative_to((root / 'docs/docs').resolve())
            rendered = ((site / relative.parent / 'index.html') if relative.name == 'index.md' else (site / relative.with_suffix('') / 'index.html')).resolve()
            if unquote(url.fragment) not in pages[rendered].ids:
                md_broken.append({'page': name, 'link': link, 'reason': 'missing rendered anchor'})
result['markdown_links'] = {'checked': md_count, 'broken': md_broken}
result['source_sha256'] = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
(proof / 'static-results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ['rendered_links', 'markdown_links']}, indent=2))
assert not broken and not md_broken
