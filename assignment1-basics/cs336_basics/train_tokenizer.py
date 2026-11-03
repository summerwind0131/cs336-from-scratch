import argparse
import json
from pathlib import Path
from time import perf_counter

from cs336_basics.bpe_train import train_bpe
from cs336_basics.tokenizer import Tokenizer, gpt2_byte_decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]

    start = perf_counter()
    vocab, merges = train_bpe(
        input_path=args.input,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )
    elapsed = perf_counter() - start

    # 将任意字节转换为 GPT-2 文件格式使用的可打印字符。
    byte_encoder = {
        byte: character
        for character, byte in gpt2_byte_decoder().items()
    }

    def token_to_text(token: bytes) -> str:
        return "".join(byte_encoder[byte] for byte in token)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = args.output_dir / "vocab.json"
    merges_path = args.output_dir / "merges.txt"

    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                token_to_text(token): token_id
                for token_id, token in vocab.items()
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with merges_path.open("w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for left, right in merges:
            f.write(f"{token_to_text(left)} {token_to_text(right)}\n")

    original = Tokenizer(vocab, merges, special_tokens)
    loaded = Tokenizer.from_files(
        vocab_path, merges_path, special_tokens
    )

    # 检查保存过程中没有改变词表、ID 或合并顺序。
    assert loaded.vocab == original.vocab
    assert loaded.merges == original.merges

    samples = [
        "",
        "Hello, world!",
        "你好，世界！",
        "first\n\nsecond  ",
        "Hello<|endoftext|>你好",
    ]
    for text in samples:
        ids = loaded.encode(text)
        assert ids == original.encode(text)
        assert loaded.decode(ids) == text

    longest = max(vocab.values(), key=len)
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Merge count: {len(merges)}")
    print(f"Longest token ({len(longest)} bytes): {longest!r}")
    print(f"Saved to: {args.output_dir}")
    print("Save/load and roundtrip checks passed.")


if __name__ == "__main__":
    main()