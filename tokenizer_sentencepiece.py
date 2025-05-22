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
        unk_token= "[UNK]",
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

    # 4) Optional: apply Turkish-aware lowercasing here
    def iterator():
        for ex in ds:
            text = ex.get("text", "")
            yield text

    # 5) Train
    tokenizer.train_from_iterator(
        tqdm(iterator(), desc="Training Unigram tokenizer"),
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
