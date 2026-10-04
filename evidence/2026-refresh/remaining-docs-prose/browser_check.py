import json
import os
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

proof = Path('/tmp/plato-remaining-docs-prose-proof')
paths = [
    ('index', '/'), ('misc', '/misc/'), ('ccdb', '/ccdb/'),
    ('development', '/development/'), ('trainer', '/configurations/trainer/'),
    ('server', '/configurations/server/'), ('getting-started', '/examples/Getting Started/'),
    ('aggregation', '/examples/algorithms/1. Server Aggregation Algorithms/'),
]
result = {'pages': [], 'scope': 'Prose rendering only; KaTeX and browser asset qualification belongs to D3.'}
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': 1440, 'height': 1080}, device_scale_factor=1)
    result['browser'] = browser.version
    for label, path in paths:
        response = page.goto('http://127.0.0.1:8000' + quote(path), wait_until='domcontentloaded')
        assert response.status == 200
        article = page.locator('article')
        assert article.is_visible()
        text = article.inner_text()
        assert text.strip()
        page.screenshot(path=str(proof / f'rendered-{label}.png'))
        result['pages'].append({'path': path, 'status': response.status, 'article_text': text, 'screenshot': f'rendered-{label}.png'})
        if label == 'server':
            block = page.locator('.admonition').filter(has=page.locator('.admonition-title', has_text='inbound_processors'))
            assert block.count() == 1
            items = block.locator('li').all_text_contents()
            assert len(items) == 3, items
            assert all(x in items[i] for i, x in enumerate(['model_decompress', 'model_dequantize', 'model_dequantize_qsgd']))
            assert 'outbound_feature_ndarrays' not in block.inner_text()
            block.scroll_into_view_if_needed()
            page.screenshot(path=str(proof / 'rendered-server-processors.png'))
            result['processor_list'] = items
        if label in ['ccdb', 'misc']:
            assert '3600' in text
        if label == 'trainer':
            assert 'lr_scheduler = "timm"' in text
            page.locator('pre').filter(has_text='min_lr = 1.0e-6').scroll_into_view_if_needed()
            page.screenshot(path=str(proof / 'rendered-trainer-fragment.png'))
        if label == 'aggregation':
            page.locator('pre').filter(has_text='dataset_capture_dir').scroll_into_view_if_needed()
            page.screenshot(path=str(proof / 'rendered-aggregation-fragment.png'))
        if label == 'development':
            page.get_by_text('Example excerpt from examples/basic/basic.py:', exact=True).scroll_into_view_if_needed()
            page.screenshot(path=str(proof / 'rendered-development-attribution.png'))
        if label == 'ccdb':
            page.locator('pre').filter(has_text='#SBATCH --time=01:00:00').scroll_into_view_if_needed()
            page.screenshot(path=str(proof / 'rendered-ccdb-batch.png'))
        if label == 'getting-started':
            assert 'Gradient Leakage Attacks and Defences' in text and 'Archived Research' in text
    browser.close()
(proof / 'browser-results.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'pages': len(result['pages']), 'browser': result['browser'], 'processor_list': result['processor_list']}, indent=2))
