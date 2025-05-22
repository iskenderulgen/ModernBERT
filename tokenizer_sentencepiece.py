import os
from tqdm.auto import tqdm
from tokenizers import Tokenizer
from tokenizers.models import Unigram
from tokenizers.normalizers import Sequence, NFKC, Lowercase
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import UnigramTrainer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import unicodedata
import regex as re

_CONTROL_CHARS_RE = re.compile(r"[\p{C}]")  # remove control/invisible chars
_HYPHEN_RE = re.compile(r"[–—−]")  # en‐dash, em‐dash, minus
_QUOTE_RE = re.compile(r"[„“”«»]")  # various curly/double quotes

_ELLIPSIS_RE = re.compile(r"\.{3,}")
_REPEAT_PUNC = re.compile(r"([^\w\s])\1+")
_NON_TURKISH = re.compile(r"[^\p{Latin}\p{N}\p{P}\s]+")
_SENT_END_RE = re.compile(r"([.!?])(?=\p{Lu})")
_WS_RE = re.compile(r"\s+")


def clean_text(text):
    """
    Clean Turkish text for tokenizer training.
    Optimized version addressing your questions.
    """
    # strip control/invisible chars
    text = _CONTROL_CHARS_RE.sub("", text)
    # unify dashes → hyphen-minus
    text = _HYPHEN_RE.sub("-", text)
    # unify curly quotes → ASCII double-quote
    text = _QUOTE_RE.sub('"', text)

    # NFKC normalization - handles most quote normalization
    text = unicodedata.normalize("NFKC", text)

    # Additional apostrophe normalization (NFKC doesn't catch all variants)
    # Keep this step as you might encounter other apostrophe variants in real data
    text = text.replace("’", "'").replace("‘", "'")

    # Preserve ellipsis but normalize it
    text = _ELLIPSIS_RE.sub("...", text)

    # Collapse all repeated punctuation to single
    text = _REPEAT_PUNC.sub(r"\1", text)

    # Turkish-specific: preserve Turkish letters explicitly
    text = _NON_TURKISH.sub("", text)

    # Add space after sentence-ending punctuation followed by capital letters
    text = _SENT_END_RE.sub(r"\1 \2", text)

    # Normalize whitespace (single operation instead of two)
    text = _WS_RE.sub(" ", text).strip()

    # Turkish-aware lowercase
    # Make sure this handles İ→i and I→ı correctly
    text = wtp.lowercase_turkish_text(text)

    return text


def train_unigram_tokenizer(
    parquet_dir="data/texts",
    output_dir="turkish_sp_unigram",
    vocab_size=32_000,
    n_sub_iterations=3,
    shrinking_factor=0.99,
    use_manual_postproc=True,
):
    # 1) Initialize Unigram tokenizer
    tokenizer = Tokenizer(Unigram())
    tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()

    # 2) Configure the trainer
    trainer = UnigramTrainer(
        vocab_size=vocab_size,
        unk_token="[UNK]",
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        n_sub_iterations=n_sub_iterations,
        shrinking_factor=shrinking_factor,
    )

    # 3) Stream all Parquet files
    data_files = os.path.join(parquet_dir, "*.parquet")
    ds = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
        streaming=True,
    )

    # 4) More efficient iterator - clean and yield in larger batches
    def iterator():
        for example in ds:                         # ds is a streaming IterableDataset
            text = example.get("text", "")
            if not text:
                continue
            cleaned = clean_text(text)
            if cleaned.strip():
                yield cleaned

    # 5) Train
    tokenizer.train_from_iterator(
        iterator(),  # Remove tqdm wrapper here since iter() handles progress better
        trainer=trainer,
    )

    # 6) Post-processing for BERT (optional)
    if use_manual_postproc:
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]")),
            ],
        )

    # 7) Save raw JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(json_path)

    # 8) Wrap as a HF fast tokenizer without needing AutoTokenizer
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=json_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=512,
        type_vocab_size=2,
    )
    hf_tok.save_pretrained(output_dir)

    print(f"Tokenizer saved to {output_dir}/")


if __name__ == "__main__":
    train_unigram_tokenizer()
