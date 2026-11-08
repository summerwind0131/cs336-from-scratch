import argparse
import tempfile
from itertools import islice
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer


def prepare_file(
    tokenizer,
    input_path: Path,
    output_path: Path,
    chunk_size: int = 65536,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = 0

    # 编码前不知道 token 总数，先分批写入临时二进制文件。
    # 临时文件放在输出目录，结束后自动删除。
    with tempfile.TemporaryFile(dir=output_path.parent) as temp:
        with input_path.open(
            "r", encoding="utf-8", newline=""
        ) as source, tqdm(
            desc=input_path.name,
            unit="tok",
            unit_scale=True,
        ) as progress:
            chunks = iter(lambda: source.read(chunk_size), "")
            token_ids = tokenizer.encode_iterable(chunks)

            while True:
                batch = np.fromiter(
                    islice(token_ids, chunk_size),
                    dtype=np.uint16,
                )
                if batch.size == 0:
                    break

                batch.tofile(temp)
                total_tokens += batch.size
                progress.update(batch.size)

        # 总长度已知，创建带有正确 shape 和 dtype 的 .npy。
        temp.seek(0)
        output = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=np.uint16,
            shape=(total_tokens,),
        )

        # 分块复制，不把完整 token 数组载入内存。
        for start in range(0, total_tokens, chunk_size):
            count = min(chunk_size, total_tokens - start)
            output[start : start + count] = np.fromfile(
                temp,
                dtype=np.uint16,
                count=count,
            )

        output.flush()
        del output

    print(f"Saved {output_path}: {total_tokens:,} tokens")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges,
        special_tokens=["<|endoftext|>"],
    )

    # 先处理较小的验证集。
    prepare_file(
        tokenizer,
        args.val_input,
        args.output_dir / "val.npy",
    )
    prepare_file(
        tokenizer,
        args.train_input,
        args.output_dir / "train.npy",
    )


if __name__ == "__main__":
    main()