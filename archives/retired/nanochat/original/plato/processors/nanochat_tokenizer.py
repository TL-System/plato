"""
Prototype tokenizer processor for Nanochat, wrapping the rustbpe+tiktoken stack.

This module exercises the build tooling for the Rust extension while providing
an adapter that conforms to Plato's processor interface.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from plato.processors.base import Processor

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

DEFAULT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]"
    r"|\s+(?!\S)|\s+"
)


class NanochatTokenizerProcessor(Processor):
    """Prototype tokenizer that can either load a saved encoding or train via rustbpe."""

    def __init__(
        self,
        tokenizer_path: str | Path | None = None,
        train_corpus: Iterable[str] | None = None,
        vocab_size: int = 32000,
        pattern: str = DEFAULT_PATTERN,
        bos_token: str = "<|bos|>",
        prepend_bos: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self.train_corpus = train_corpus
        self.vocab_size = max(vocab_size, 256 + len(SPECIAL_TOKENS))
        self.pattern = pattern
        self.bos_token = bos_token
        self.prepend_bos = prepend_bos

        self._encoding = self._load_encoding()
        self._special_tokens = self._infer_special_tokens()
        self._bos_token_id = self._special_tokens.get(self.bos_token)

    def _load_encoding(self):
        if self.tokenizer_path:
            return self._load_from_pickle(self.tokenizer_path)
        if self.train_corpus is not None:
            return self._train_from_corpus(self.train_corpus)
        raise ValueError("Either tokenizer_path or train_corpus must be provided.")

    def _load_from_pickle(self, path: Path):
        with path.open("rb") as handle:
            encoding = pickle.load(handle)
        return encoding

    def _train_from_corpus(self, corpus: Iterable[str]):
        try:
            import rustbpe
        except ImportError as exc:  # pragma: no cover - guarded import
            raise RuntimeError(
                "rustbpe extension is required to train a Nanochat tokenizer."
            ) from exc

        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover - guarded import
            raise RuntimeError(
                "tiktoken is required to construct Nanochat tokenizer encodings."
            ) from exc

        tokenizer = rustbpe.Tokenizer()
        tokenizer.train_from_iterator(
            iter(corpus),
            self.vocab_size - len(SPECIAL_TOKENS),
            pattern=self.pattern,
        )

        mergeable_ranks = {
            bytes(piece): rank for piece, rank in tokenizer.get_mergeable_ranks()
        }
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            token: tokens_offset + index for index, token in enumerate(SPECIAL_TOKENS)
        }

        return tiktoken.Encoding(
            name="nanochat-rustbpe",
            pat_str=tokenizer.get_pattern(),
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

    def _infer_special_tokens(self):
        try:
            encode_single_token = self._encoding.encode_single_token
            special_token_set = getattr(self._encoding, "special_tokens_set", set())
        except AttributeError as exc:
            raise RuntimeError(
                "tiktoken encoding missing expected interfaces."
            ) from exc

        mapping = {}
        for token in SPECIAL_TOKENS:
            if token in special_token_set:
                mapping[token] = encode_single_token(token)
        return mapping

    def _encode_one(self, text: str):
        ids = list(self._encoding.encode_ordinary(text))
        if self.prepend_bos and self._bos_token_id is not None:
            ids.insert(0, self._bos_token_id)
        return ids

    def process(self, data: Any):
        if isinstance(data, str):
            return self._encode_one(data)
        if isinstance(data, Sequence):
            return [self._encode_one(item) for item in data]
        raise TypeError(f"Unsupported payload type: {type(data)!r}")
