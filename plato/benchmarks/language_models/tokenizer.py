"""
BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1) HuggingFace Tokenizer that can do both training and inference but is really confusing
2) Universal Wrapper that can load any HuggingFace tokenizer (e.g., for GPT-2 which has slightly different tokenization rules than GPT-4) for inference only.
"""

import os

SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>",  # user messages
    "<|user_end|>",
    "<|assistant_start|>",  # assistant messages
    "<|assistant_end|>",
    "<|python_start|>",  # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>",  # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I haven't validated that this is actually a good idea, TODO.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "gpt2")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk (e.g. "out/tokenizer")
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(
            BPE(
                byte_fallback=True,  # needed!
                unk_token=None,
                fuse_unk=False,
            )
        )
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(
            SPLIT_PATTERN
        )  # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(
                    pattern=gpt4_split_regex, behavior="isolated", invert=False
                ),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,  # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        # encode a single string
        # prepend/append can be either a string of a special token or a token id directly.
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # encode a single special token via exact match
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # save the tokenizer to disk
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


# -----------------------------------------------------------------------------
# Universal Tokenizer Wrapper that works with any HuggingFace model
class UniversalHuggingFaceTokenizer:
    """Universal wrapper that works with any HuggingFace model"""

    def __init__(self, tokenizer_dir, model_config=None):
        self.tokenizer = HuggingFaceTokenizer.from_directory(tokenizer_dir)
        self.model_config = model_config
        self._pad_token_id = None
        self._bos_token_id = None
        self._detect_special_tokens()

    def _detect_special_tokens(self):
        """Auto-detect special tokens for any model"""
        # Try to get pad token from tokenizer
        if hasattr(self.tokenizer, "tokenizer"):
            tokenizer_obj = self.tokenizer.tokenizer

            # Try common pad token names
            pad_candidates = ["<pad>", "[PAD]", "<|pad|>", "</s>", "<|endoftext|>"]
            for candidate in pad_candidates:
                if hasattr(tokenizer_obj, "token_to_id"):
                    token_id = tokenizer_obj.token_to_id(candidate)
                    if token_id is not None:
                        self._pad_token_id = token_id
                        break

            # Try common BOS token names
            bos_candidates = ["<s>", "[CLS]", "<|startoftext|>", "<|endoftext|>"]
            for candidate in bos_candidates:
                if hasattr(tokenizer_obj, "token_to_id"):
                    token_id = tokenizer_obj.token_to_id(candidate)
                    if token_id is not None:
                        self._bos_token_id = token_id
                        break

        # Fallback to config-based detection
        if self.model_config and hasattr(self.model_config, "pad_token_id"):
            self._pad_token_id = self.model_config.pad_token_id

        if self.model_config and hasattr(self.model_config, "bos_token_id"):
            self._bos_token_id = self.model_config.bos_token_id

        # Final fallbacks based on common patterns
        if self._pad_token_id is None:
            # Most models use either 0 or their EOS token
            self._pad_token_id = 0

        if self._bos_token_id is None:
            # Use pad token as fallback
            self._bos_token_id = self._pad_token_id

    def get_bos_token_id(self):
        return self._bos_token_id

    def get_pad_token_id(self):
        return self._pad_token_id

    def __call__(self, prompts, prepend=None):
        """Universal tokenization method"""
        if isinstance(prompts, str):
            prompts = [prompts]

        result = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            if prepend is not None:
                tokens = [prepend] + tokens
            result.append(tokens)

        return result[0] if len(result) == 1 else result

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)
