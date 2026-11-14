import torch

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q,K,V,is_casual=False):
        batch_size,n_queries,d=Q.shape
        n_keys=K.shape[1]
        q_tile_size=16
        k_tile_size=16
        scale=d**-0.5

        acc_dtype=torch.promote_types(Q.dtype,torch.float32)
        O=torch.empty_like(Q)
        L=Q.new_empty((batch_size,n_queries),dtype=acc_dtype)

        for q_start in range(0,n_queries,q_tile_size):
            q_slice=slice(q_start,q_start+q_tile_size)
            q_i=Q[:,q_slice].to(acc_dtype)

            row_shape=(batch_size,q_i.shape[1],1)
            m_i=q_i.new_full(row_shape,float("-inf"))
            l_i=q_i.new_zeros(row_shape)
            acc_i=torch.zeros_like(q_i)

            for k_start in range(0,n_keys,k_tile_size):
                k_slice = slice(k_start, k_start + k_tile_size)
                k_j = K[:, k_slice].to(acc_dtype)
                v_j = V[:, k_slice].to(acc_dtype)

                scores=(q_i @k_j.transpose(-2,-1))*scale
                m_new = torch.maximum(m_i, scores.amax(dim=-1, keepdim=True))

                alpha = torch.exp(m_i - m_new)
                p = torch.exp(scores - m_new)

                l_i = alpha * l_i + p.sum(dim=-1, keepdim=True)
                acc_i = alpha * acc_i + (p @ v_j)
                m_i = m_new
            O[:, q_slice] = (acc_i / l_i).to(Q.dtype)
            L[:, q_slice] = (m_i + torch.log(l_i)).squeeze(-1)
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, *grad_outputs) :
        return NotImplementedError