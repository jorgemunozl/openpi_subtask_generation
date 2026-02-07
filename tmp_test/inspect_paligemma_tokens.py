#!/usr/bin/env python3
"""Inspect the last 128 tokens of PaliGemma tokenizer to see if they're special tokens."""

import sentencepiece
import openpi.shared.download as download

# Load the PaliGemma tokenizer
path = download.maybe_download("/x2robot_v2/xinyuanfang/projects_v2/.cache/openpi/big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
with path.open("rb") as f:
    tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

vocab_size = tokenizer.vocab_size()
print(f"PaliGemma vocab size: {vocab_size}")
print(f"\nInspecting last 128 tokens (indices {vocab_size - 128} to {vocab_size - 1}):\n")

# Check special token IDs
bos_id = tokenizer.bos_id() if hasattr(tokenizer, 'bos_id') else -1
eos_id = tokenizer.eos_id() if hasattr(tokenizer, 'eos_id') else -1
unk_id = tokenizer.unk_id() if hasattr(tokenizer, 'unk_id') else -1
pad_id = tokenizer.pad_id() if hasattr(tokenizer, 'pad_id') else -1

print(f"Special token IDs: BOS={bos_id}, EOS={eos_id}, UNK={unk_id}, PAD={pad_id}\n")

# Inspect the last 128 tokens
last_128_start = vocab_size - 128
special_count = 0
control_count = 0
printable_count = 0

print("Last 128 tokens:")
for i in range(last_128_start, vocab_size):
    token_id = i
    try:
        piece = tokenizer.id_to_piece(token_id)
        # Check if it's a special token (starts with special markers)
        is_special = piece.startswith('<') and piece.endswith('>')
        is_control = not piece.isprintable() or len(piece) == 0
        
        if token_id in [bos_id, eos_id, unk_id, pad_id]:
            special_count += 1
            status = " [SPECIAL]"
        elif is_special:
            special_count += 1
            status = " [SPECIAL_FORMAT]"
        elif is_control:
            control_count += 1
            status = " [CONTROL]"
        else:
            printable_count += 1
            status = ""
        
        # Show first 20 and last 20
        if i < last_128_start + 20 or i >= vocab_size - 20:
            print(f"  {token_id:6d}: {repr(piece):30s}{status}")
    except Exception as e:
        print(f"  {token_id:6d}: ERROR - {e}")

print(f"\nSummary of last 128 tokens:")
print(f"  Special tokens (BOS/EOS/UNK/PAD or <...> format): {special_count}")
print(f"  Control/non-printable tokens: {control_count}")
print(f"  Regular printable tokens: {printable_count}")
print(f"  Total: {special_count + control_count + printable_count}")

# Also check if there are any special tokens in the entire vocabulary
print(f"\nChecking entire vocabulary for special tokens...")
all_special = []
for i in range(vocab_size):
    try:
        piece = tokenizer.id_to_piece(i)
        if piece.startswith('<') and piece.endswith('>'):
            all_special.append((i, piece))
    except:
        pass

print(f"Found {len(all_special)} tokens with <...> format in entire vocabulary:")
for token_id, piece in all_special[:20]:  # Show first 20
    print(f"  {token_id:6d}: {piece}")
if len(all_special) > 20:
    print(f"  ... and {len(all_special) - 20} more")


