# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.cuda.amp as amp
from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

from ..modules.model import sinusoidal_embedding_1d


def pad_tensor(x, dim: int, padding_size: int):
    shape = list(x.shape)
    shape[dim] = padding_size
    pad = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float32).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_rank()
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def usp_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    pose_embedding=None,
    y=None,
    img_ref=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)
    
    pose_embedding, _ = self.project_pose(pose_embedding)
    pose_embedding = pose_embedding.unsqueeze(0)
 
    # x = [x_one + pose_embedding_one for x_one, pose_embedding_one in zip(x, pose_embedding)]


    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]

    # seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    # assert seq_lens.max() <= seq_len

    img_ref = img_ref[0].transpose(0,1)
    img_ref = self.ref_conv(img_ref).flatten(2).transpose(1, 2)

    grid_sizes = torch.stack([torch.tensor([u[0] + 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)
    seq_len += img_ref.size(1)

    x = [torch.concat([_img_ref.unsqueeze(0), u], dim=1) for _img_ref, u in zip(img_ref, x)]

    img_ref_length = img_ref.size(1)
    pose_list = [p for p in pose_embedding] 
    aligned_pose_list = []
    for u_x, u_p in zip(x, pose_list):
        if img_ref_length > 0:
            pad_front = u_p.new_zeros(img_ref_length, u_p.size(1))
            u_p = torch.cat([pad_front, u_p], dim=0)
        pad_len = seq_len - u_p.size(0)
        if pad_len > 0:
            pad_back = u_p.new_zeros(pad_len, u_p.size(1))
            u_p = torch.cat([u_p, pad_back], dim=0)
        aligned_pose_list.append(u_p)
    pose_embedding = torch.cat([p.unsqueeze(0) for p in aligned_pose_list], dim=1)

 
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))



    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_len,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
    
    sp_world = get_sequence_parallel_world_size()
    if seq_len % sp_world:
        padding_size = sp_world - (seq_len % sp_world)
   
        x = pad_tensor(x, dim=1, padding_size=padding_size)
        pose_embedding = pad_tensor(pose_embedding, dim=1, padding_size=padding_size)

    # Context Parallel
    sp_rank = get_sequence_parallel_rank()
    x = torch.chunk(
        x, sp_world,
        dim=1)[sp_rank]
    pose_embedding = torch.chunk(
        pose_embedding, sp_world,
        dim=1)[sp_rank]



    for block in self.blocks:
        x = block(x, pose_embedding=pose_embedding, **kwargs)
    
    
    
    grid_sizes = torch.stack([torch.tensor([u[0] - 1, u[1], u[2]]) for u in grid_sizes]).to(grid_sizes.device)

    # head
    x = self.head(x, e)

    # Context Parallel
    x = get_sp_group().all_gather(x, dim=1)

    img_ref_length = img_ref.size(1)
    x = x[:, img_ref_length:]

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def usp_attn_forward(self,
                     x,
                     seq_lens,
                     grid_sizes,
                     freqs,
                     pose_embedding=None,
                     dtype=torch.bfloat16
                     ):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    # TODO: We should use unpaded q,k,v for attention.
    # k_lens = seq_lens // get_sequence_parallel_world_size()
    # if k_lens is not None:
    #     q = torch.cat([u[:l] for u, l in zip(q, k_lens)]).unsqueeze(0)
    #     k = torch.cat([u[:l] for u, l in zip(k, k_lens)]).unsqueeze(0)
    #     v = torch.cat([u[:l] for u, l in zip(v, k_lens)]).unsqueeze(0)

    x_out = xFuserLongContextAttention()(
        None,
        query=half(q),
        key=half(k),
        value=half(v))
        # window_size=self.window_size)

    # TODO: padding after attention.
    # x = torch.cat([x, x.new_zeros(b, s - x.size(1), n, d)], dim=1)
    if pose_embedding is not None:
       
        k_p = self.norm_k_pose(self.k_pose(pose_embedding)).view(b, s, n, d)
        v_p = self.v_pose(pose_embedding).view(b, s, n, d)
        k_p_roped = rope_apply(k_p, grid_sizes, freqs)
        pose_out = xFuserLongContextAttention()(
            None,
            query=half(q),  # Video Query
            key=half(k_p_roped),  # Pose Key
            value=half(v_p), )     # Pose Value
            # window_size=self.window_size)
        x_out = x_out + self.pose_gate * pose_out
    # output
    x_final = x_out.flatten(2)
    x_final = self.o(x_final)
    return x_final