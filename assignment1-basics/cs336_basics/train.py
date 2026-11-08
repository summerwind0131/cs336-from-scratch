import argparse
from pathlib import Path

import numpy as np
import torch

from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import(
    AdamW,
    lr_cosine_schedule,
    gradient_clipping,
)
from cs336_basics.losses import cross_entropy_loss
from cs336_basics.data import(
    data_loading,
    load_checkpoint,
    save_checkpoint,
)

@torch.no_grad()
def evaluate(model,dataset,args):
    model.eval()
    total_loss=0.0

    for _ in range(args.eval_batches):
        x,y=data_loading(
            dataset,
            args.batch_size,
            args.context_length,
            args.device
        )
        total_loss+=cross_entropy_loss(model(x),y).item()

    model.train()
    return total_loss/args.eval_batches


def main():
    parser = argparse.ArgumentParser()

    # 数据、设备和 checkpoint
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    # 模型超参数
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # 训练和优化器超参数
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--cosine-steps", type=int, default=10000)
    parser.add_argument(
        "--betas", type=float, nargs=2, default=(0.9, 0.95)
    )
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # 日志、验证和保存频率
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # .npy 文件采用内存映射，不把整个数据集载入内存。
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")

    model=TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope=True,
        max_seq_len=args.context_length,
        theta=args.rope_theta,
        device=args.device,
        dtype=torch.float32,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(
            args.resume, model, optimizer
        )
        print(f"Resumed after {start_step} completed steps")

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.train()

    for step in range(start_step,args.max_steps):
        # 1. 更新当前学习率
        lr = lr_cosine_schedule(
            t=step,
            alpha_max=args.max_lr,
            alpha_min=args.min_lr,
            Tw=args.warmup_steps,
            Tc=args.cosine_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # 2. 随机采样一个 batch
        x, y = data_loading(
            train_data,
            args.batch_size,
            args.context_length,
            args.device,
        )

        # 3. 清空梯度、前向计算、反向传播
        optimizer.zero_grad(set_to_none=True)

        logits = model(x)                  # (B, T, V)
        loss = cross_entropy_loss(logits, y)

        loss.backward()

        # 4. 梯度裁剪后更新参数
        gradient_clipping(
            model.parameters(), args.grad_clip
        )
        optimizer.step()

        completed = step + 1
        last_step = completed == args.max_steps

        # 5. 输出当前训练 batch 的 loss
        if completed % args.log_every == 0 or last_step:
            print(
                f"step={completed} "
                f"train_loss={loss.item():.4f} "
                f"lr={lr:.6g}"
            )

        # 验证前释放当前 batch 的输出。
        del logits, loss

        # 6. 定期计算多个验证 batch 的平均 loss
        if completed % args.eval_every == 0 or last_step:
            val_loss = evaluate(model, val_data, args)
            print(
                f"step={completed} val_loss={val_loss:.4f}"
            )

        # 7. 定期保存；最后一步也保存
        if completed % args.save_every == 0 or last_step:
            save_checkpoint(
                model,
                optimizer,
                completed,
                args.checkpoint,
            )
            print(f"Saved checkpoint: {args.checkpoint}")

            
if __name__ == "__main__":
    main()