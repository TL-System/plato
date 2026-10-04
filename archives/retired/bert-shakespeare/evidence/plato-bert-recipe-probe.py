import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, BertConfig

torch.manual_seed(7)
config = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                    num_attention_heads=2, intermediate_size=32,
                    hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
model = AutoModelForCausalLM.from_config(config).eval()
tokens = torch.tensor([[1, 2, 3, 4]])
changed = tokens.clone()
changed[0, 3] = 5
with torch.no_grad():
    first = model(tokens).logits
    second = model(changed).logits
result = {
    'scope': 'Tiny random BERT architecture with default config, offline; no pretrained checkpoint or dataset tested.',
    'is_decoder': config.is_decoder,
    'earlier_position_max_difference_when_future_token_changes': (first[:, 0] - second[:, 0]).abs().max().item(),
}
assert result['is_decoder'] is False
assert result['earlier_position_max_difference_when_future_token_changes'] > 1e-8
Path('/tmp/plato-bert-recipe-probe.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
