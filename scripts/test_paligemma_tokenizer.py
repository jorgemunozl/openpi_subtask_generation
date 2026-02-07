#!/usr/bin/env python3
"""Standalone PaliGemma tokenizer tester.

This script lets you validate a local SentencePiece model (PaliGemma) without
cloning the full repo. Optionally compares against a Hugging Face tokenizer.

Usage examples:
  python scripts/test_paligemma_tokenizer.py --spm /path/to/paligemma_tokenizer.model
  python scripts/test_paligemma_tokenizer.py --spm /path/to/model --hf google/paligemma2-3b-mix-224

Requires: sentencepiece (and transformers if using --hf).
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def need(module: str, pip: str | None = None) -> None:
    print(f"Missing dependency: {module}. Install with: pip install {pip or module}")
    sys.exit(2)


def load_sentencepiece(path: str):
    try:
        import sentencepiece as spm
    except Exception:
        need("sentencepiece", "sentencepiece")
    sp = spm.SentencePieceProcessor()
    # load model bytes if possible, otherwise load from file
    try:
        sp.load(path)
    except Exception:
        # try loading raw proto
        with open(path, "rb") as f:
            proto = f.read()
        sp = spm.SentencePieceProcessor(model_proto=proto)
    return sp


def safe_encode(sp, text: str, add_bos: bool = True) -> list[int]:
    # Try new API first, fall back to EncodeAsIds
    try:
        # some builds support `encode` with out_type
        ids = sp.encode(text, out_type=int)
    except Exception:
        try:
            ids = sp.EncodeAsIds(text)
        except Exception as e:
            raise RuntimeError("Unsupported SentencePiece API") from e
    if add_bos and hasattr(sp, "bos_id"):
        try:
            bos = sp.bos_id()
            if bos != -1:
                ids = [bos] + list(ids)
        except Exception:
            pass
    return list(map(int, ids))


def safe_decode(sp, ids: Sequence[int]) -> str:
    try:
        # new API
        return sp.decode(ids)
    except Exception:
        try:
            return sp.DecodeIds(list(ids))
        except Exception as e:
            raise RuntimeError("Unsupported SentencePiece API for decode") from e


def compare_with_hf(hf_name: str, prompts: list[str], sp):
    try:
        from transformers import AutoTokenizer
    except Exception:
        need("transformers", "transformers")
    tok = AutoTokenizer.from_pretrained(hf_name, use_fast=False)
    print(f"HF tokenizer vocab_size: {tok.vocab_size}, special: bos={tok.bos_token_id} eos={tok.eos_token_id} pad={tok.pad_token_id}")
    for p in prompts:
        print("---")
        print(f"PROMPT: {p}")
        ids_sp = safe_encode(sp, p, add_bos=True)
        det_sp = safe_decode(sp, ids_sp)
        print(f"SP IDs: {ids_sp[:40]}{'...' if len(ids_sp)>40 else ''}")
        print(f"SP decode: {det_sp}")

        ids_hf = tok.encode(p, add_special_tokens=True)
        det_hf = tok.decode(ids_hf, skip_special_tokens=True)
        print(f"HF IDs: {ids_hf[:40]}{'...' if len(ids_hf)>40 else ''}")
        print(f"HF decode: {det_hf}")


def main():
    parser = argparse.ArgumentParser(description="Test a PaliGemma SentencePiece tokenizer")
    parser.add_argument("--spm", required=True, help="Path to SentencePiece model file (.model)")
    parser.add_argument("--hf", help="Optional Hugging Face tokenizer name to compare against")
    parser.add_argument("--prompts", nargs="*", help="Optional prompts to test; otherwise a small default set is used")
    args = parser.parse_args()

    sp = load_sentencepiece(args.spm)
    print(f"Loaded SentencePiece model from {args.spm}")
    try:
        vocab = sp.get_piece_size()
    except Exception:
        vocab = None
    print(f"SP vocab size: {vocab}")
    try:
        print("SP special ids: bos=", getattr(sp, "bos_id", lambda: -1)(), "eos=", getattr(sp, "eos_id", lambda: -1)(), "pad=", getattr(sp, "pad_id", lambda: -1)())
    except Exception:
        pass

    prompts = args.prompts or [
        "Task: pick up the flashcard on the table",
        "Task: move to the red cup and pick it up.",
        "Subtask: pick up the card; Action: ",
        "Hello, how are you?",
    ]

    for p in prompts:
        print("-------------------------------")
        print(f"PROMPT: {p}")
        ids = safe_encode(sp, p, add_bos=True)
        print("Encoded IDs:", ids[:200])
        det = safe_decode(sp, ids)
        print("Decoded text:", det)

    if args.hf:
        print("\nComparing with HF tokenizer:\n")
        compare_with_hf(args.hf, prompts, sp)


if __name__ == "__main__":
    main()
