import os
from datasets import load_dataset
from tokenizers import SentencePieceUnigramTokenizer, normalizers, Regex
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast


def train_unigram_tokenizer(
    parquet_dir="data/parquets_cleaned",
    output_dir="turkish_sp_unigram",
    vocab_size=32_000
):
    # Instantiate the tokenizer with your normalizers
    tokenizer = SentencePieceUnigramTokenizer()
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Strip(),
            normalizers.Nmt(),
            normalizers.NFKC(),
            normalizers.Replace(Regex(" {2,}"), " "),
            normalizers.Replace("I", "ı"),
            normalizers.Replace("İ", "i"),
            normalizers.Lowercase(),
        ]
    )

    # Load the dataset from parquet files
    ds = load_dataset(
        "parquet",
        data_files=os.path.join(parquet_dir, "*.parquet"),
        split="train",
        streaming=True,
    )

    # Define an iterator to yield text samples
    def iterator():
        for ex in ds:
            txt = ex.get("text", "")
            if txt:
                yield txt

    # Train in streaming mode
    tokenizer.train_from_iterator(
        iterator=iterator(),
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        initial_alphabet=list("abcçdefgğhıijklmnoöprsştuüvyz0123456789,.;:!?/-%"),
        unk_token="[UNK]"
    ) 

    # BERT‐style post‐processing
    tokenizer.enable_truncation(512)
    tokenizer.post_processor = BertProcessing(
        sep=("[SEP]", tokenizer.token_to_id("[SEP]")),
        cls=("[CLS]", tokenizer.token_to_id("[CLS]")),
    )

    # Save and wrap in HF Fast tokenizer
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(json_path)

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