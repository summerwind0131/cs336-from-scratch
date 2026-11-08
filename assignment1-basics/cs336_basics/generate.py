import torch
from cs336_basics.losses import cross_entropy_loss
from cs336_basics.softmax import softmax
@torch.no_grad()

def generate(
    model,prompt_ids,
    context_length,
    eos_token_id,
    max_new_tokens=100,
    temperature=1.0,
    top_p=1.0
):
    model.eval()
    device=next(model.parameters()).device

    tokens=torch.tensor([prompt_ids],dtype=torch.long,device=device)
    for _ in range(max_new_tokens):
        x=tokens[:,-context_length:]
        logits=model(x)[:,-1,:]

        if temperature==0:
            next_token=logits.argmax(
                dim=-1,keepdims=True
            )
        else:
            probs=softmax(logits/temperature,d=-1)

            if top_p<1.0:
                sorted_probs, sorted_ids = torch.sort(
                    probs, dim=-1, descending=True
                )

                
                cumulative_before = (
                    sorted_probs.cumsum(dim=-1) - sorted_probs
                )
                sorted_probs = sorted_probs.masked_fill(
                    cumulative_before >= top_p, 0.0
                )

                # 对保留的概率重新归一化。
                sorted_probs = sorted_probs / sorted_probs.sum(
                    dim=-1, keepdim=True
                )

                # 先抽取排序后的下标，再映射回原始 token ID。
                sampled_index = torch.multinomial(
                    sorted_probs, num_samples=1
                )
                next_token = sorted_ids.gather(
                    dim=-1, index=sampled_index
                )
            else:
                next_token = torch.multinomial(
                    probs, num_samples=1
                )

        tokens = torch.cat([tokens, next_token], dim=-1)

        if next_token.item() == eos_token_id:
            break

    return tokens[0].tolist()