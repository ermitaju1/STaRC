
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # import pdb; pdb.set_trace()
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class AttnPool1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Parameter(torch.randn(dim))
        self.Wk = nn.Linear(dim, dim, bias=False)
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.tau = nn.Parameter(torch.tensor(10.0))

    def forward(self, x, mask=None):  # x: (B, N, D)
        K = self.Wk(x)                           # (B,N,D)
        V = self.Wv(x)                           # (B,N,D)
        attn = (K * self.q).sum(-1) / self.tau.clamp_min(1e-3)  # (B,N)
        if mask is not None:
            attn = attn.masked_fill(~mask.bool(), float('-inf'))
        w = F.softmax(attn, dim=1)               # (B,N)
        global_feat = (w.unsqueeze(-1) * V).sum(1)  # (B,D)
        return global_feat


class MeanPool1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return x.mean(dim=1)
        m = mask.float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1e-6)
        return (x * m).sum(dim=1) / denom


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_dim, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # No Drop Path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(mlp_dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer (+ Global Aggregation Module) """
    def __init__(self, num_features=100, embed_dim=768, depth=12,
                 num_heads=12, mlp_dim=2048, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, global_mode: str = 'attn'):
        super().__init__()
        assert global_mode in ('attn', 'mean')
        self.depth = depth
        self.num_features = num_features
        self.embed_dim = embed_dim
        self.global_mode = global_mode

        self.pos_embed = nn.Parameter(torch.zeros(1, num_features, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_dim=mlp_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for _ in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        if global_mode == 'attn':
            self.global_head = AttnPool1D(embed_dim)
        else:
            self.global_head = MeanPool1D()

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0); nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def _add_pos(self, x):
        if x.size(1) != self.pos_embed.size(1):
            time_embed = self.pos_embed.transpose(1, 2)  # (1,D,P)
            new_time_embed = F.interpolate(time_embed, size=(x.size(1)), mode='nearest')
            new_time_embed = new_time_embed.transpose(1, 2)  # (1,N,D)
            return x + new_time_embed
        else:
            return x + self.pos_embed

    def encode_local(self, x, mode=None):
        # if add_pos:
        # if mode != "inference":
        # #     import pdb; pdb.set_trace()
        # # else:
        x = self._add_pos(x)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        #!여기에 B, T, D를 넣어도 됨

        # B, T, D = x.shape
        # # video_clone = x.clone() #!여기에 window를 만들어도 됨

        # # if self.args.self_attn:
        # vid_zero = torch.zeros_like(x)
        # count = torch.zeros(B, T, 1, device=x.device)

        # for window in self.window_sizes:
        #     for batch_idx in range(B):
        #         sample_not_ln = x[batch_idx]  # [T, D]
        #         sample = x[batch_idx]

        #         for i in range(0, T - window + 1, self.stride):
        #             mini_batch = sample[i : i + window, :]  # [window, D]
        #             mini_batch_not_ln = sample_not_ln[i : i + window, :]
        #             q = k = mini_batch_not_ln
        #             v = mini_batch.clone()
        #             scale = D**0.5

        #             attn_scores = torch.matmul(q, k.T) / scale  # [window, window]
        #             attn_weights = F.softmax(
        #                 attn_scores, dim=-1
        #             )  # [window, window]
        #             attn_output = torch.matmul(attn_weights, v)  # [window, D]

        #             vid_zero[batch_idx, i : i + window, :] += attn_output
        #             count[batch_idx, i : i + window, :] += 1

        # vid_zero = vid_zero / (count + 1e-6)
        # # vid_zero = self.norm(vid_zero)
        # x = x + vid_zero
        x = self.norm(x)
        return x


    def forward(self, x):
        # import pdb; pdb.set_trace()
        return self.encode_local(x)

    def forward_with_global(self, local, mask=None, mode=None):
        # if mode != "inference":
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # if mode == "training":
        #     global_feat = self.global_head(x, mask)  # (B, D)
        # else:
        x = self.encode_local(local, mode=mode)
        global_feat  = self.global_head(x, mask)  # (B, D)
            # x = self.encode_local(x, mode=mode)         # (B, N, D)
        return x, global_feat

# def softattention_select(self, memory_bank, window_tokens, mode, uns_video=None):
#     """
#     window_tokens: [B, W, D]  # 이미 윈도우 요약된 특징(사전 계산됨)
#     return:
#       topk_window_embeds: [B, W, D]  # 각 윈도우별 계층 검색 결과 임베딩
#       total_window_sents: placeholder 리스트(기존 호환)
#     """
#     assert window_tokens.dim() == 3, f"Expected [B, W, D], got {window_tokens.shape}"
#     assert self.args.ret_option in {"hier_ret", "hier_concat"}
#     assert self.args.sim_match == "anchor_cos"

#     B, W, D = window_tokens.shape
#     soft_k   = self.args.soft_k
#     do_norm  = getattr(self.args, "retrieval_norm", True)  # 코사인 일관성 위해 권장

#     # (선택) 정규화
#     if do_norm:
#         window_tokens = F.normalize(window_tokens, dim=-1)

#     topk_window_embeds = []
#     total_window_sents = []

#     for b in range(B):
#         batch_out = []
#         for i in range(W):
#             target_feature = window_tokens[b, i, :]            # [D]
#             if do_norm:
#                 target_feature = F.normalize(target_feature.unsqueeze(0), dim=-1).squeeze(0)
#             # 계층형 메모리 검색 (이미 클러스터 구조가 memory_bank에 있음)
#             topk_embed = self.hierarchical_memory_search(target_feature, soft_k, memory_bank)  # [1, D]
#             batch_out.append(topk_embed)
#         batch_out = torch.cat(batch_out, dim=0).unsqueeze(0).float()  # [1, W, D]
#         topk_window_embeds.append(batch_out)
#         total_window_sents.append('no')  # 필요시 실제 summary로 교체

#     topk_window_embeds = torch.cat(topk_window_embeds, dim=0)  # [B, W, D]
#     return topk_window_embeds, total_window_sents

# def ret(self, window_tokens, memory_bank, mode, uns_video=None):
#     """
#     window_tokens: [B, W, D]  # 이미 준비된 윈도우 요약 특징
#     return:
#       ret: [B, W_or_1, D_out]  # ret_encoder=='avg'면 W_or_1=1
#     """
#     # 계층 검색
#     topk_embeds, _ = self.softattention_select(memory_bank, window_tokens, mode, uns_video=uns_video)
#     if topk_embeds is None or (hasattr(topk_embeds, "__len__") and len(topk_embeds) == 0):
#         B, _, D = window_tokens.shape
#         return torch.zeros(B, 1, D, device=window_tokens.device)

#     # 윈도우 축 평균(옵션)
#     value_vectors = topk_embeds  # [B, W, D]
#     if getattr(self.args, "ret_encoder", None) == "avg":
#         value_vectors = value_vectors.mean(dim=1, keepdim=True)  # [B, 1, D]

#     # 투영
#     if hasattr(self, "ret2t5_proj") and self.ret2t5_proj is not None:
#         value_vectors = self.ret2t5_proj(value_vectors)          # [B, W_or_1, D_out]

#     return value_vectors