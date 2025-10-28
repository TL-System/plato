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
# For example, GPT2TokenizerFast doesn't have a get_bos_token_id() method,
# so we need this wrapper to provide a unified interface.


class UniversalHuggingFaceTokenizer:
    """
    Universal wrapper that provides a consistent interface for any HuggingFace tokenizer.
    
    This wrapper automatically detects special tokens (BOS, PAD, EOS) and provides
    utility methods that work across different tokenizer implementations.
    """

    def __init__(self, tokenizer):
        """
        Initialize the wrapper with a HuggingFace tokenizer.
        
        Args:
            tokenizer: A HuggingFace tokenizer instance (e.g., GPT2TokenizerFast)
        """
        self.tokenizer = tokenizer
        self._pad_token_id = None
        self._bos_token_id = None
        self._eos_token_id = None
        self._detect_special_tokens()

    def _detect_special_tokens(self):
        """
        Auto-detect special token IDs from the tokenizer.
        
        Detection strategy (in order of priority):
        1. Try direct attributes on the tokenizer (bos_token_id, pad_token_id, eos_token_id)
        2. For missing tokens, use EOS as BOS/PAD for models like GPT-2
        3. Try token_to_id() method with common token names
        4. Final fallbacks: 0 for pad, pad for bos
        """
        # Strategy 1: Direct attributes (works for most HuggingFace tokenizers)
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            self._bos_token_id = self.tokenizer.bos_token_id
        
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            self._pad_token_id = self.tokenizer.pad_token_id
        
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            self._eos_token_id = self.tokenizer.eos_token_id
            # For GPT-2 and similar models, BOS is often the same as EOS
            if self._bos_token_id is None:
                self._bos_token_id = self._eos_token_id
            # Use EOS as pad if no pad token exists
            if self._pad_token_id is None:
                self._pad_token_id = self._eos_token_id
        
        # Strategy 2: Try token_to_id method for tokenizers with nested structure
        if hasattr(self.tokenizer, "tokenizer"):
            tokenizer_obj = self.tokenizer.tokenizer

            if self._pad_token_id is None:
                pad_candidates = ["<pad>", "[PAD]", "<|pad|>", "</s>", "<|endoftext|>"]
                self._pad_token_id = self._try_token_candidates(tokenizer_obj, pad_candidates)

            if self._bos_token_id is None:
                bos_candidates = ["<s>", "[CLS]", "<|startoftext|>", "<|endoftext|>"]
                self._bos_token_id = self._try_token_candidates(tokenizer_obj, bos_candidates)

        # Strategy 3: Final fallbacks
        if self._pad_token_id is None:
            self._pad_token_id = 0  # Most models default to 0

        if self._bos_token_id is None:
            self._bos_token_id = self._pad_token_id

    def _try_token_candidates(self, tokenizer_obj, candidates):
        """
        Try to find a token ID from a list of candidate token strings.
        
        Args:
            tokenizer_obj: The tokenizer object with token_to_id method
            candidates: List of token strings to try
            
        Returns:
            Token ID if found, None otherwise
        """
        if not hasattr(tokenizer_obj, "token_to_id"):
            return None
            
        for candidate in candidates:
            token_id = tokenizer_obj.token_to_id(candidate)
            if token_id is not None:
                return token_id
        return None

    def get_bos_token_id(self):
        """Get the beginning-of-sequence token ID."""
        return self._bos_token_id

    def get_pad_token_id(self):
        """Get the padding token ID."""
        return self._pad_token_id
    
    def get_eos_token_id(self):
        """Get the end-of-sequence token ID."""
        return self._eos_token_id

    def __call__(self, prompts, prepend=None):
        """
        Tokenize prompts with optional prepended token.
        
        Args:
            prompts: Single string or list of strings to tokenize
            prepend: Optional token ID to prepend to each sequence
            
        Returns:
            List of token IDs, or list of lists if multiple prompts
        """
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
        """Delegate all other attributes to the wrapped tokenizer."""
        return getattr(self.tokenizer, name)
