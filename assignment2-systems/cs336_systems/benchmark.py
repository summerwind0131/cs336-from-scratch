import torch
import argparse
import timeit
import json
import statistics
from pathlib import Path
from contextlib import nullcontext
import math
import torch.cuda.nvtx as nvtx
from einops import einsum

import cs336_basics.model as basics_model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy,softmax
from cs336_systems.optimizer import AdamW

MODEL_FIELDS=("d_model","d_ff","num_layers","num_heads")

MODEL_SIZES={
    "small":  (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large":  (1280, 5120, 36, 20),
    "xl":     (1600, 6400, 48, 25),
    "2.7B":   (2560, 10240, 32, 32),
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the basics Transformer."
    )

    # 模型配置：预设 + 可选的单项覆盖
    parser.add_argument(
        "--model-size", choices=MODEL_SIZES, default="small"
    )
    for name in MODEL_FIELDS:
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=int, default=None
        )

    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--batch-size", type=int, default=4)

    # 测量配置
    parser.add_argument(
        "--mode",
        choices=("forward", "forward-backward","train"),
        default="forward-backward",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)

    # 运行环境与结果保存
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)


    # 是否启用 NVTX，以及结果保存位置
    parser.add_argument("--nvtx", action="store_true")
    parser.add_argument("--output", default=None)

    #是否启用memoproprofiling,以及结果保存位置
    parser.add_argument("--memory-profile",action="store_true")
    parser.add_argument("--memory-output",default=None)

    #是否测试torch.compile
    parser.add_argument("--compile",action="store_true")



    args = parser.parse_args()

    # 用户显式传入的参数优先，其余从预设补齐
    preset = dict(zip(MODEL_FIELDS, MODEL_SIZES[args.model_size]))
    for name, value in preset.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    positive_fields = (
        *MODEL_FIELDS,
        "vocab_size",
        "context_length",
        "batch_size",
        "measurement_steps",
    )
    for name in positive_fields:
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")

    if args.warmup_steps < 0:
        parser.error("warmup_steps must be nonnegative")
    if args.rope_theta <= 0:
        parser.error("rope_theta must be positive")
    if args.d_model % args.num_heads != 0:
        parser.error("d_model must be divisible by num_heads")
    if (args.d_model // args.num_heads) % 2 != 0:
        parser.error("RoPE requires an even head dimension")

    if args.memory_profile:
        if torch.device(args.device).type != "cuda":
            parser.error("--memory-profile requires --device cuda")

        if not args.memory_output:
            parser.error("--memory-profile requires --memory-output")

    return args

    return args

def prepare(args):
    torch.manual_seed(args.seed)

    device=torch.device(args.device)
    dtype_map={
        "float32":torch.float32,
        "float16":torch.float16,
        "bfloat16":torch.bfloat16,
    }
    dtype=dtype_map[args.dtype]

    model=BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device=device,dtype=dtype)

    shape=(args.batch_size,args.context_length)

    token_ids=torch.randint(
        low=0,
        high=args.vocab_size,
        size=shape,
        dtype=torch.long,
        device=device,
    )

    targets=torch.randint(
        low=0,
        high=args.vocab_size,
        size=shape,
        dtype=torch.long,
        device=device,
    )

    return model,token_ids,targets

def nvtx_range(name, enabled):
    return nvtx.range(name) if enabled else nullcontext()

def annotated_attention(Q,K,V,mask=None):
    enabled=Q.is_cuda
    with nvtx_range("attention",enabled):
        with nvtx_range("attention/qk_matmul",enabled):
            scores=einsum(
                Q,K,"... query d_k,... key d_k->... query key"
            )
        with nvtx_range("attention/scale_mask",enabled):
            scores=scores/math.sqrt(K.shape[-1])
            if mask is not None:
                scores=torch.where(mask,scores,float("-inf"))

        with nvtx_range("attention/softmax",enabled):
            weights=softmax(scores,dim=-1)
        with nvtx_range("attention/av_matmul",enabled):
            output=einsum(
                weights,V,"... query key, ... key d_v-> ... query d_v"
            )
        return output
        

def run_step(model,token_ids,targets,mode,optimizer=None,
             nvtx_enabled=False):
    device=token_ids.device
    def sync():
        if device.type=="cuda":
            torch.cuda.synchronize()
    if mode=="forward":
        with torch.no_grad():
            sync()
            with nvtx_range("forward",nvtx_enabled):
                t0=timeit.default_timer()
                model(token_ids)
                sync()
                t1=timeit.default_timer()

            return {"forward":t1-t0,}
          
    elif mode in ("forward-backward","train"):
        model.zero_grad(set_to_none=True)
        sync()

        with nvtx_range("forward",nvtx_enabled):
            t0=timeit.default_timer()
            logits=model(token_ids)
            sync()
            t1=timeit.default_timer()

        with nvtx_range("loss",nvtx_enabled):
            loss=cross_entropy(logits,targets)
            sync()
            t2=timeit.default_timer()

        with nvtx_range("backward",nvtx_enabled):
            loss.backward()
            sync()
            t3=timeit.default_timer()

        timings={
            "forward":t1-t0,
            "loss":t2-t1,
            "backward":t3-t2,
        }

        if mode == "train":
            assert optimizer is not None
            with nvtx_range("optimizer_step", nvtx_enabled):
                start = timeit.default_timer()
                optimizer.step()
                sync()
                timings["optimizer_step"] = timeit.default_timer() - start

        return timings


    
def benchmark(model,token_ids,targets,args,optimizer=None):
    warmup_steps=args.warmup_steps
    measurement_steps=args.measurement_steps
    mode=args.mode
    device=token_ids.device

    #预热
    with nvtx_range("warmup", args.nvtx):
        for i in range(warmup_steps):
            run_step(model,token_ids,targets,mode,optimizer=optimizer,nvtx_enabled=args.nvtx)


    #峰值统计
    if args.memory_profile:
        torch.cuda.synchronize(device)
        torch.cuda.memory.reset_peak_memory_stats(device)

    #开始正式测量
    results=[]
    with nvtx_range("measurement", args.nvtx):
        for i in range(measurement_steps):
            with nvtx_range(f"step_{i}", args.nvtx):
                timings = run_step(
                    model, token_ids, targets, mode,
                    optimizer=optimizer, nvtx_enabled=args.nvtx,
                )
            results.append(timings)
    #读取峰值
    memory_stats=None
    if args.memory_profile:
        torch.cuda.synchronize(device)
        memory_stats={
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes":torch.cuda.max_memory_reserved(device)
        }
    return results,memory_stats


def report(results,args,memory_stats=None):
    summary={}

    print(
        f"mode={args.mode}, device={args.device}, "
        f"dtype={args.dtype}, n={len(results)}"
    )
    print(f"{'stage':<12} {'mean (ms)':>12} {'std (ms)':>12}")

    # forward 模式只有一个键；反向模式有三个键
    for stage in results[0]:
        samples = [step[stage] for step in results]

        mean_seconds = statistics.mean(samples)
        std_seconds = (
            statistics.stdev(samples)
            if len(samples) > 1
            else None
        )

        summary[stage] = {
            "mean_seconds": mean_seconds,
            "std_seconds": std_seconds,
        }

        std_text = (
            f"{std_seconds * 1000:.3f}"
            if std_seconds is not None
            else "N/A"
        )
        print(
            f"{stage:<12} "
            f"{mean_seconds * 1000:>12.3f} "
            f"{std_text:>12}"
        )

    if memory_stats is not None:
        allocated_gib = memory_stats["peak_allocated_bytes"] / 1024**3
        reserved_gib = memory_stats["peak_reserved_bytes"] / 1024**3

        print(f"Peak allocated: {allocated_gib:.3f} GiB")
        print(f"Peak reserved:  {reserved_gib:.3f} GiB")

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": vars(args),
            "timing_unit": "seconds",
            "std_ddof": 1,
            "summary": summary,
            "measurements": results,
            "memory":memory_stats,
        }

        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(f"Saved results to {output_path}")
    


def main():
    args = parse_args()

    if args.nvtx:
        basics_model.scaled_dot_product_attention = annotated_attention

    recording = False
    memory_path = None

    try:
        if args.memory_profile:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable in this environment"
                )

            memory_path = Path(args.memory_output)
            memory_path.parent.mkdir(parents=True, exist_ok=True)

            # 在创建模型之前开启，保留参数分配的调用栈
            torch.cuda.memory._record_memory_history(
                max_entries=1000000
            )
            recording = True

        model, token_ids, targets = prepare(args)
        if args.compile:
            model=torch.compile(model)
        optimizer = None
        if args.mode == "train":
            optimizer = AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        results, memory_stats = benchmark(
            model, token_ids, targets, args,
            optimizer=optimizer,
        )

        # 只有成功完成测量，才保存正式实验快照
        if memory_path is not None:
            torch.cuda.synchronize(token_ids.device)

            torch.cuda.memory._dump_snapshot(str(memory_path))
            print(f"Saved memory snapshot to {memory_path}")

        report(results, args, memory_stats)

    finally:
        # 计算、保存或报告发生异常时，也会执行这里
        if recording:
            torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()


