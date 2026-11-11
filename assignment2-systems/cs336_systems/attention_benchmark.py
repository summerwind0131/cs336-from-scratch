import torch
import argparse
import csv
import gc
import itertools
import statistics
import timeit
from pathlib import Path

from cs336_basics.model import scaled_dot_product_attention

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--d-models", type=int, nargs="+",
        default=[16, 32, 64, 128],
    )
    parser.add_argument(
        "--seq-lens", type=int, nargs="+",
        default=[256, 1024, 4096, 8192, 16384],
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/attention.csv")

    parser.add_argument("-compile",action="store_true")

    args = parser.parse_args()

    if args.warmup_steps < 0 or args.measurement_steps < 1:
        parser.error("warmup_steps 必须 >= 0，measurement_steps 必须 >= 1")

    if any(x <= 0 for x in args.d_models + args.seq_lens):
        parser.error("所有维度必须为正整数")

    return args


def benchmark_one(args, d_model, seq_len):
    """测量一组配置，返回前向时间、反向时间和反向前显存。"""
    device = torch.device(args.device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    torch.manual_seed(args.seed)

    # 没有 head 维度，形状为 (batch_size, seq_len, d_model)。
    shape = (8, seq_len, d_model)
    Q, K, V = [
        torch.randn(
            shape,
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
        for _ in range(3)
    ]

    # 预先生成输出的梯度，避免把 loss 的计算混入计时。
    dO = torch.randn(shape, device=device, dtype=torch.float32)

    # True 表示允许关注该位置，广播到 batch 维度。
    mask = torch.ones(
        seq_len, seq_len, device=device, dtype=torch.bool
    ).tril_()
    attention_fn=scaled_dot_product_attention
    if args.compile:
        attention_fn=torch.compile(attention_fn)

    forward_times = []
    backward_times = []
    memory_samples = []

    for i in range(args.warmup_steps + args.measurement_steps):
        # 清空上一轮输入梯度，避免梯度累加。
        Q.grad = K.grad = V.grad = None
        sync()

        # 1. 前向计时：每次都创建新的计算图。
        start = timeit.default_timer()

        output = attention_fn(Q, K, V, mask)

        sync()
        forward_ms = (timeit.default_timer() - start) * 1000

        # 2. forward 完成、backward 尚未开始时的当前在用显存。
        before_backward = (
            torch.cuda.memory_allocated(device)
            if device.type == "cuda"
            else None
        )

        # 3. 反向计时。
        start = timeit.default_timer()

        output.backward(dO)

        sync()
        backward_ms = (timeit.default_timer() - start) * 1000

        del output

        # 预热执行完整前后向，但不计入统计。
        if i >= args.warmup_steps:
            forward_times.append(forward_ms)
            backward_times.append(backward_ms)

            if before_backward is not None:
                memory_samples.append(before_backward)

    return {
        "forward_ms": statistics.mean(forward_times),
        "backward_ms": statistics.mean(backward_times),
        "before_backward_mib": (
            max(memory_samples) / 1024**2
            if memory_samples
            else None
        ),
    }


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 Python 环境无法使用 CUDA")

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "d_model",
        "seq_len",
        "forward_ms",
        "backward_ms",
        "before_backward_mib",
        "status",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        # 笛卡尔积：默认 4 × 5 = 20 组。
        for d_model, seq_len in itertools.product(
            args.d_models, args.seq_lens
        ):
            print(
                f"Running d={d_model}, seq_len={seq_len}",
                flush=True,
            )

            row = {
                "d_model": d_model,
                "seq_len": seq_len,
                "status": "OK",
            }

            try:
                row.update(benchmark_one(args, d_model, seq_len))

            except torch.OutOfMemoryError:
                # 该组未完成，时间和显存留空，继续下一组。
                row["status"] = "OOM"

            finally:
                # 函数退出后清理该组张量，包括 OOM 时遗留的对象。
                gc.collect()

                if args.device == "cuda":
                    torch.cuda.empty_cache()

            writer.writerow(row)
            file.flush()
            print(row, flush=True)

    print(f"Saved results to {path}")


if __name__ == "__main__":
    main()