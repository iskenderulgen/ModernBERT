import sentencepiece as spm
import json

model_prefix    = "turkish_bert_spm"
special_tokens  = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# 4) Train SentencePiece
spm.SentencePieceTrainer.Train(
    input=["tr_article_2.txt", "tr_thesis_1.txt"],
    model_prefix="turkish_bert_spm",
    vocab_size=32000,
    model_type="unigram",
    character_coverage=0.9999,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='[PAD]',
    unk_piece='[UNK]',
    bos_piece='[CLS]',
    eos_piece='[SEP]',
    user_defined_symbols='[MASK]',
    hard_vocab_limit=False,
    max_sentence_length=2048,
    byte_fallback=True,
    train_extremely_large_corpus=False,
    accept_language="tr",
    normalization_rule_name="nmt_nfkc_cf",
    num_sub_iterations=15,
    split_digits=True,
)


# 5) Load & verify
sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
print("Special token IDs:")
for tok in special_tokens:
    print(f"  {tok:6s} -> {sp.piece_to_id(tok)}")


# 6) Save BERT‐style vocab
with open(f"{model_prefix}.vocab", "w", encoding="utf-8") as f:
    for i in range(sp.get_piece_size()):
        f.write(f"{sp.id_to_piece(i)}\t{i}\n")


"""
# 7) Save minimal JSON config
config = {
    "do_lower_case": False,
    "vocab_size": sp.get_piece_size(),
    "model_type": "unigram",
    "special_tokens": {
        "pad_token":   "[PAD]",
        "unk_token":   "[UNK]",
        "cls_token":   "[CLS]",
        "sep_token":   "[SEP]",
        "mask_token":  "[MASK]"
    }
}
with open(f"{model_prefix}.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
"""

print("Done. Generated .model, .vocab, .json under", model_prefix + ".*")