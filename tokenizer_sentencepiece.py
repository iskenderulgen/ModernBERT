import sentencepiece as spm
from pathlib import Path

model_prefix    = "turkish_bert_spm"
special_tokens  = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# dynamically collect all .txt files under data/txt_files
txt_dir = Path("data/txt_files")
input_files = [str(p) for p in sorted(txt_dir.glob("*.txt"))]

# 4) Train SentencePiece
spm.SentencePieceTrainer.Train(
    input=input_files,
    model_prefix="turkish_bert_spm",
    vocab_size=32000,
    accept_language="tr",
    model_type="unigram",
    num_threads=75,
    max_sentence_length=10_000_000,
    user_defined_symbols='[MASK]',
    normalization_rule_name="nmt_nfkc_cf",
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3,
    unk_piece='[UNK]',
    bos_piece='[CLS]',
    eos_piece='[SEP]',
    pad_piece='[PAD]',
    unk_surface='[UNK]',
    train_extremely_large_corpus=True,
)


# 5) Load & verify
sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
print("Special token IDs:")
for tok in special_tokens:
    print(f"  {tok:6s} -> {sp.piece_to_id(tok)}")