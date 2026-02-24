import torch
import torch.nn as nn
from .modeling_t5 import T5ForConditionalGeneration
from .vit import VisionTransformer
from transformers import T5Tokenizer
from transformers.modeling_outputs import (
    BaseModelOutput,
)
from util.HD_loss import loss_saliency
import numpy as np
import torch.nn.functional as F
from .reward import RewardComputer
import random, math
from torch.nn.functional import cosine_similarity
from .asot import *
from time import time

def _get_tokenizer(tokenizer_path, num_bins=0):
    if 't5' in tokenizer_path:
        tokenizer = T5Tokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if num_bins:
            new_tokens = ["<time=" + str(i) + ">" for i in range(num_bins)]
            tokenizer.add_tokens(list(new_tokens))
    else:
        raise NotImplementedError(tokenizer_path)
    return tokenizer


class Vid2Seq(torch.nn.Module):
    def __init__(self,
                 t5_path,
                 num_features=100,
                 embed_dim=768,
                 depth=12,
                 heads=12,
                 mlp_dim=2048,
                 vis_drop=0.,
                 tokenizer=None,
                 enc_drop=0.,
                 dec_drop=0.1,
                 use_speech=True,
                 use_video=True,
                 num_bins=100,
                 label_smoothing=0.1,
                 args=None):
        super().__init__()
        self.args = args
        self.t5_model = T5ForConditionalGeneration.from_pretrained(encoder_dropout=enc_drop, decoder_dropout=dec_drop, label_smoothing=label_smoothing,
                                                                   pretrained_model_name_or_path=t5_path, local_files_only=True, is_gated_act="v1_1" in t5_path)
        self.t5_model.resize_token_embeddings(len(tokenizer) - num_bins)  # remove the weights of the 28 tokens that are not used (32128 vs 32100 in the tokenizer)
        self.t5_model.resize_token_embeddings(len(tokenizer))  # add time tokens
        self.visual_encoder = VisionTransformer(num_features=num_features,
                                                embed_dim=embed_dim,
                                                depth=depth,
                                                num_heads=heads,
                                                mlp_dim=mlp_dim,
                                                qkv_bias=True,
                                                qk_scale=None,
                                                drop_rate=vis_drop,
                                                attn_drop_rate=vis_drop,
                                                norm_layer=nn.LayerNorm)
        self.t5_tokenizer = tokenizer
        self.use_speech = use_speech
        self.use_video = use_video
        self.proj_v2t = nn.Linear(embed_dim, embed_dim)
        self.proj_r2t = nn.Linear(embed_dim, embed_dim)

        hidden_dim = 768
        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.saliency_tok_proj = nn.Linear(1, self.t5_model.model_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # self.window_sizes = [5, 35, 65] #8, 32, 64가 아직 까지는 best
        self.stride = args.stride
        if self.stride == 5:
            self.window_sizes = [5, 35, 65]
        elif self.stride == 6:
            self.window_sizes = [6, 36, 66]
        elif self.stride == 7:
            self.window_sizes = [7, 35, 63]
        elif self.stride == 9:
            self.window_sizes = [9, 36, 63]
        elif self.stride == 8:
            self.window_sizes = [8, 32, 64] #, 32, 64]
        elif self.stride == 10:
            self.window_sizes = [10, 40, 70]
        self.topk = 10

        #10 40 70 완 5~10/ 30~40/ 60~70
        #5 35 65 완
        #6 36 66
        #7 35 63
        #9 36 63


        self.K = getattr(args, "asot_K", 8)
        anchors = torch.randn(self.K, embed_dim)
        self.anchors = nn.Parameter(F.normalize(anchors, dim=-1))

    # ---------- saliency → segment 유틸 ----------
    def _smooth1d(self, x, k=5):
        if k <= 1:
            return x
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        if x.dim() == 0:
            return x.view(1)              # 스칼라 → [1]
        if x.dim() == 1:
            x = x[None, :]                # [T] → [1,T]
        B, T = x.shape
        if T == 0:
            return x.squeeze(0) if B == 1 else x
        pad = (k - 1) // 2
        w = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / k
        y = F.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
        y = F.conv1d(y, w).squeeze(1)     # [B,T]
        return y.squeeze(0) if B == 1 else y

    def _find_segments_from_mask(self, mask_1d: torch.Tensor):
        """
        mask_1d: [T] (bool/0-1) 예상. 스칼라/빈 텐서/길이1도 안전 처리.
        return: [(start, end)] with end exclusive
        """
        # 0) 스칼라/None 방어
        if mask_1d is None:
            return []
        if not torch.is_tensor(mask_1d):
            mask_1d = torch.tensor(mask_1d)

        # 1) 1D 보장
        if mask_1d.dim() == 0:
            mask_1d = mask_1d.view(1)
        elif mask_1d.dim() > 1:
            mask_1d = mask_1d.flatten()

        T = mask_1d.numel()
        if T == 0:
            return []
        if T == 1:
            return [(0,1)] if bool(mask_1d.item()) else []

        # 2) diff 기반 시작/끝 검출
        m = mask_1d.to(torch.int32)
        # 아래에서는 입력이 1D임이 보장되어 pad 사용 가능
        dm = F.pad(m[1:] - m[:-1], (1, 0))  # +1: start, -1: end
        starts = (dm == 1).nonzero(as_tuple=False).flatten()
        ends   = (dm == -1).nonzero(as_tuple=False).flatten()

        if m[0] == 1:
            starts = torch.cat([torch.tensor([0], device=m.device), starts])
        if m[-1] == 1:
            ends = torch.cat([ends, torch.tensor([T], device=m.device)])

        return [(int(s), int(e)) for s, e in zip(starts, ends)]

    def _merge_close_and_filter(self, segs, min_len=4, max_gap=2):
        # 짧은 구간 제거 + 가까우면 병합
        segs = [(s, e) for (s, e) in segs if e - s >= min_len]
        if not segs:
            return []
        segs.sort()
        merged = [segs[0]]
        for s, e in segs[1:]:
            ps, pe = merged[-1]
            if s - pe <= max_gap:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    def _select_top_segments(self, segs, w_1d, top_m=8):
        # 구간 점수 = saliency 합(필요시 평균으로 변경 가능)
        if not segs:
            return []
        scored = [ (se, float(w_1d[se[0]:se[1]].sum())) for se in segs ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [se for se, _ in scored[:top_m]]

    def _pool_segments(self, video_bt, w_b, segs, mode="attn"):
        # video_bt: [T,D], w_b: [T], segs: [(s,e)] → [N_seg, D]
        if not segs:
            return video_bt.mean(dim=0, keepdim=True)
        pooled = []
        for (s, e) in segs:
            chunk = video_bt[s:e]  # [L,D]
            if mode == "attn":
                ww = F.softmax(w_b[s:e], dim=0).unsqueeze(-1)  # [L,1]
                pooled.append((ww * chunk).sum(dim=0, keepdim=True))
            else:
                pooled.append(chunk.mean(dim=0, keepdim=True))
        return torch.cat(pooled, dim=0)

    def _segments_from_saliency_batch(self, w_bt, percentile=0.7, min_len=4, max_gap=2, top_m=8, smooth_k=5):
        if w_bt.dim() == 0:
            w_bt = w_bt.view(1, 1)        # 스칼라 → [1,1]
        elif w_bt.dim() == 1:
            w_bt = w_bt[None, :]          # [T] → [1,T]
        B, T = w_bt.shape
        if T == 0:
            return [[] for _ in range(B)]

        w_s = self._smooth1d(w_bt, k=smooth_k)     # [B,T]
        segs_all = []
        for b in range(B):
            wb = w_s[b].reshape(-1)                # ★ 1D 보장
            T_b = wb.numel()
            if T_b == 0:
                segs_all.append([])
                continue
            if T_b == 1:
                segs_all.append([(0,1)] if wb[0] >= wb[0] else [])  # 항상 [(0,1)] 또는 []
                continue

            # torch.quantile 이 구버전에서 문제되면 kthvalue로 대체 가능(아래 주석 참고)
            tau = torch.quantile(wb, q=torch.tensor(percentile, device=wb.device, dtype=wb.dtype))
            mask = (wb >= tau)

            segs = self._find_segments_from_mask(mask)
            segs = self._merge_close_and_filter(segs, min_len=min_len, max_gap=max_gap)
            segs = self._select_top_segments(segs, wb, top_m=top_m)
            segs_all.append(segs)
        return segs_all

# (참고) 구버전 PyTorch에서 quantile 이슈 시:
# idx = max(1, int(math.ceil(T_b * percentile)))
# tau = wb.kthvalue(idx).values
    # (선택) 세그먼트 수가 너무 많을 때 KMeans로 더 요약
    def _cluster_segments_optional(self, seg_embs: torch.Tensor, K: int = 0):
        """
        seg_embs: [N_seg, D]; K>0이면 간단 K-means로 N_seg→K 축약
        """
        if K <= 0 or seg_embs.size(0) <= K:
            return seg_embs
        X = F.normalize(seg_embs, dim=-1)
        idx = torch.linspace(0, X.size(0) - 1, steps=K, device=X.device).round().long()
        C = X[idx].clone()
        for _ in range(10):
            sim = X @ C.t()               # [N,K]
            y = sim.argmax(dim=1)         # [N]
            newC = []
            for k in range(K):
                pts = X[y == k]
                if pts.numel() == 0:
                    newC.append(C[k:k+1])
                else:
                    newC.append(F.normalize(pts.mean(dim=0, keepdim=True), dim=-1))
            C = torch.cat(newC, dim=0)
        outs = []
        sim = X @ C.t()
        y = sim.argmax(dim=1)
        for k in range(K):
            pts = seg_embs[y == k]
            outs.append((pts.mean(dim=0, keepdim=True) if pts.numel() else seg_embs.mean(dim=0, keepdim=True)))
        return torch.cat(outs, dim=0)

    def _decode_text(self, input_ids):
        """배치 input_ids → 문자열 리스트 복원"""
        texts = []
        for ids in input_ids:
            texts.append(self.t5_tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip())
        return texts

    def _build_frame_mask_from_saliency(self, s, mode="sigmoid", temp=1.0, eps=1e-6, smooth_kernel_size=0):
        """
        s: [B, T]  (HD layer에서 나온 saliency score)
        return m: [B, T] in [0, 1], 평균 1로 스케일된 곱셈 마스크(아래에서 affine로 처리)
        """
        B, T = s.shape

        # 1) 정규화: 분포/스케일 튀는 것 방지 (z-score → sigmoid or softmax)
        if mode == "softmax":  # 확률분포, 합=1 → 이후 평균=1 맞출 때 자동 스케일 업
            m = torch.softmax(s / (temp + eps), dim=-1)               # [B, T], sum=1
        else:
            # z-score 후 시그모이드: outlier에 둔감, temp로 샤프닝
            s_n = (s - s.mean(dim=1, keepdim=True)) / (s.std(dim=1, keepdim=True) + eps)
            m = torch.sigmoid(s_n / (temp + eps))                     # [B, T], 0~1

        # 2) (선택) 시간 스무딩: 프레임 토글링 방지
        if smooth_kernel_size and smooth_kernel_size > 1:
            pad = (smooth_kernel_size - 1) // 2
            k = torch.ones(1, 1, smooth_kernel_size, device=s.device, dtype=s.dtype) / smooth_kernel_size
            m_ = m.unsqueeze(1)                                       # [B, 1, T]
            m_ = torch.nn.functional.pad(m_, (pad, pad), mode="replicate")
            m = torch.nn.functional.conv1d(m_, k).squeeze(1)          # [B, T]
            m = m.clamp(0, 1)

        # 3) 평균 1로 맞추는 affine 스케일 (곱셈 후 전체 스케일 유지)
        #    video' = video * (a + b*m). 보통 a=0, b>0로 쓰되, 평균 1 맞추려면:
        #    a=0, b = T / sum(m)  => mean(a+b*m)=1
        sum_m = m.sum(dim=1, keepdim=True) + eps
        b = (m.shape[1]) / sum_m                                     # [B, 1]
        a = 0.0
        m_affine = a + b * m                                          # 평균이 1이 되도록
        return m_affine, m                                            # (곱셈용, 원본0~1)

    def _score_sequence_logprob(self, encoder_outputs, encoder_atts, y_input_ids, y_attention_mask):
        """
        주어진 encoder_outputs/attn 에 대해, decoder teacher-forcing으로
        y (input_ids) 의 토큰 로그확률 합을 계산.
        """
        # 라벨: 다음 토큰 예측하도록 한칸 쉬프트
        labels = y_input_ids.masked_fill(y_attention_mask == 0, -100)
        out = self.t5_model(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_atts,
            decoder_input_ids=None,         # labels만 주면 내부에서 shift
            labels=labels,
            return_dict=True
        )
        # T5ForConditionalGeneration.loss 는 mean CE. 합(logprob)이 필요하면 토큰 단위로 다시 계산:
        # trick: reduction='none'이 없어 mean이라 직접 로짓으로 재계산:
        # 보다 쉬운 방법: 모델에 logits를 내도록 하고 우리가 직접 NLL을 합산
        logits = out.logits                    # [B, L, V]
        vocab_size = logits.size(-1)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y_input_ids[:, 1:].contiguous()
        shift_mask = y_attention_mask[:, 1:].contiguous()
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logp = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        token_logp = token_logp * shift_mask
        seq_logp = token_logp.sum(dim=1)  # [B]
        return seq_logp

    def _apply_saliency_global_attention(self, video_clone, s):
        """
        video_clone: [B, T, D] - 윈도우 어텐션으로 증폭된 비디오 프레임 임베딩
        s:           [B, T]    - saliency score (정규화 전/후 상관없음; 함수 내에서 z-score 처리)

        return:
        video_input: [B, T, D] - 전역 가중으로 재가중된 비디오 임베딩
        w:           [B, T]    - 프레임별 전역 가중 (softmax)
        """
        B, T, D = video_clone.shape

        # 1) saliency z-score 정규화
        s = s - s.mean(dim=1, keepdim=True)
        s = s / (s.std(dim=1, keepdim=True) + 1e-6)          # [B, T]

        # 2) saliency → 전역 쿼리(1개)로 투영
        sal_q_seq = self.sal_query_proj(s.unsqueeze(-1))      # [B, T, D]
        sal_q = sal_q_seq.mean(dim=1, keepdim=True)           # [B, 1, D]  전역 요약 쿼리

        # 3) 키/밸류 준비 (원하면 sal_kv_proj로 투영)
        k = video_clone                                       # [B, T, D]
        v = video_clone
        # k = self.sal_kv_proj(video_clone); v = k            # (선택) 투영 사용시

        # 4) 점수 → 가중치
        scale = D ** 0.5
        attn_logits = torch.matmul(sal_q, k.transpose(-2, -1)) / scale   # [B, 1, T]
        w = torch.softmax(attn_logits, dim=-1)                           # [B, 1, T]
        w = w.squeeze(1)                                                 # [B, T]

        # 5) 프레임 재가중 (1.0 offset으로 과도 소거 방지)
        #    sal_weight_scale로 가중 세기 조절 가능 (초기 1.0)
        video_input = video_clone * (1.0 + self.sal_weight_scale * w.unsqueeze(-1))  # [B, T, D]
        return video_input, w

    def get_topk_indices(self,similarity_scores, k):
        if similarity_scores.shape[0] < k:
            return torch.arange(similarity_scores.shape[0])
        return torch.topk(similarity_scores, k, dim=0).indices

    def hierarchical_memory_search(self, target_feature, soft_k, memory_hierarchy):
        k = soft_k
        threshold = 0.7
        selected_levels = self.args.hier_use
        retrieval_type = "top-k"
        
        combined_vectors = []
        topk_clusters = []
        
        max_level = max(memory_hierarchy.keys(), key=lambda x: int(x.split('_')[1]))
        sorted_levels = sorted(memory_hierarchy.keys(), key=lambda x: int(x.split('_')[1]), reverse=True)
        
        # Create a dictionary to store summaries for each level
        level_summaries = {level: [] for level in sorted_levels}
        for i,level in enumerate(sorted_levels):
            clusters = memory_hierarchy[level]
            if not topk_clusters:
                topk_clusters = [
                    (cosine_similarity(target_feature.unsqueeze(0), cluster["clip_embedding"]), cluster)
                    for cluster_id, cluster in clusters.items()
                ]
                
                topk_clusters.sort(key=lambda x: x[0], reverse=True)
                if retrieval_type == "max":
                    topk_clusters = topk_clusters[:1]
                elif retrieval_type == "top-k":
                    # topk_clusters = topk_clusters[:k]
                    topk_clusters = topk_clusters[:k]
                elif retrieval_type == "similarity":
                    topk_clusters = [(score, cluster) for score, cluster in topk_clusters if score >= threshold]
                    if not topk_clusters:
                        return torch.zeros_like(target_feature.unsqueeze(0))  # No clusters exceed the threshold

            else:
                next_level_clusters = []
                
                for _, cluster in topk_clusters:
                    parent_ids = cluster["parent_clusters"]
                    
                    if isinstance(parent_ids, list):
                        for parent_id in parent_ids:
                            if f'cluster_{parent_id}' in clusters:
                                sub_cluster = clusters[f'cluster_{parent_id}']
                                sub_score = cosine_similarity(target_feature.unsqueeze(0), sub_cluster["clip_embedding"])
                                next_level_clusters.append((sub_score, sub_cluster))
                    else:
                        if f'cluster_{parent_ids}' in clusters:
                            sub_cluster = clusters[f'cluster_{parent_ids}']
                            sub_score = cosine_similarity(target_feature.unsqueeze(0), sub_cluster["clip_embedding"])
                            next_level_clusters.append((sub_score, sub_cluster))
                
                next_level_clusters.sort(key=lambda x: x[0], reverse=True)
                if retrieval_type == "max":
                    next_level_clusters = next_level_clusters[:1]
                elif retrieval_type == "top-k":
                    next_level_clusters = next_level_clusters[:k]
                elif retrieval_type == "similarity":
                    next_level_clusters = [(score, cluster) for score, cluster in next_level_clusters if score >= threshold]
                    if not next_level_clusters:
                    # If retrieval_type is adaptive and topk_clusters is empty, return averaged combined_vectors
                        if combined_vectors:
                            final_embedding = torch.cat(combined_vectors, dim=0).mean(dim=0, keepdim=True)
                            return final_embedding
                        else:
                            return torch.zeros_like(target_feature.unsqueeze(0))  # No clusters exceed the threshold

            # Append the summary texts for the current level
            for _, cluster in topk_clusters:
                if "summary" in cluster:
                    level_summaries[level].append(cluster["summary"])
            if level in selected_levels:
                level_vectors = torch.stack([cluster["clip_embedding"] for _, cluster in topk_clusters]).squeeze(dim=1)
                if retrieval_type != "max":
                    level_vectors=level_vectors.mean(dim=0,keepdim=True)
                combined_vectors.append(level_vectors)
        # for level, summaries in level_summaries.items():
        #     print(f"Level: {level}")
        #     for summary in summaries:
        #         print(f" - Summary: {summary}")
        final_embedding = torch.cat(combined_vectors, dim=0)  
        final_embedding = final_embedding.mean(dim=0,keepdim=True)

        return final_embedding

    def softattention_select(self, memory_bank, window_tokens, mode, uns_video=None):
        """
        window_tokens: [B, W, D]  # 이미 윈도우 요약된 특징(사전 계산됨)
        return:
        topk_window_embeds: [B, W, D]  # 각 윈도우별 계층 검색 결과 임베딩
        total_window_sents: placeholder 리스트(기존 호환)
        """
        # assert window_tokens.dim() == 3, f"Expected [B, W, D], got {window_tokens.shape}"
        assert self.args.ret_option in {"hier_ret", "hier_concat"}
        assert self.args.sim_match == "anchor_cos"
        window_tokens = torch.stack(window_tokens, dim=0)  # [B, W, D]
        # import pdb; pdb.set_trace()
        B, W, D = window_tokens.shape
        soft_k   = self.args.soft_k
        do_norm  = getattr(self.args, "retrieval_norm", True)  # 코사인 일관성 위해 권장

        # (선택) 정규화
        if do_norm:
            window_tokens = F.normalize(window_tokens, dim=-1)

        topk_window_embeds = []
        total_window_sents = []

        for b in range(B):
            batch_out = []
            for i in range(W):
                target_feature = window_tokens[b, i, :]            # [D]
                if do_norm:
                    target_feature = F.normalize(target_feature.unsqueeze(0), dim=-1).squeeze(0)
                # 계층형 메모리 검색 (이미 클러스터 구조가 memory_bank에 있음)
                topk_embed = self.hierarchical_memory_search(target_feature, soft_k, memory_bank)  # [1, D]
                batch_out.append(topk_embed)
            batch_out = torch.cat(batch_out, dim=0).unsqueeze(0).float()  # [1, W, D]
            topk_window_embeds.append(batch_out)
            total_window_sents.append('no')  # 필요시 실제 summary로 교체

        topk_window_embeds = torch.cat(topk_window_embeds, dim=0)  # [B, W, D]
        return topk_window_embeds, total_window_sents

    def ret(self, window_tokens, memory_bank, mode=None, uns_video=None):
        """
        window_tokens: [B, W, D]  # 이미 준비된 윈도우 요약 특징
        return:
        ret: [B, W_or_1, D_out]  # ret_encoder=='avg'면 W_or_1=1
        """
        # 계층 검색
        topk_embeds, _ = self.softattention_select(memory_bank, window_tokens, mode, uns_video=uns_video)
        if topk_embeds is None or (hasattr(topk_embeds, "__len__") and len(topk_embeds) == 0):
            B, _, D = window_tokens.shape
            return torch.zeros(B, 1, D, device=window_tokens.device)

        # 윈도우 축 평균(옵션)
        value_vectors = topk_embeds  # [B, W, D]
        if getattr(self.args, "ret_encoder", None) == "avg":
            value_vectors = value_vectors.mean(dim=1, keepdim=True)  # [B, 1, D]

        # 투영
        if hasattr(self, "ret2t5_proj") and self.ret2t5_proj is not None:
            value_vectors = self.ret2t5_proj(value_vectors)          # [B, W_or_1, D_out]

        return value_vectors

    def cluster_momentum_adaptive(self, video, iou_seg):
        """
        Args:
            video: (B, T, D) tensor
            iou_seg: list of list of (start, end) tuples
                    e.g. [[(22, 30), (32, 44), ...], [(8, 14), (17, 20), ...]]
        Returns:
            segments_all: list of (N_seg_i, D) tensors (평균 pooled segment features)
            iou_seg: 그대로 반환
        """
        B, T, D = video.shape
        video = F.normalize(video, dim=-1)
        segments_all = []

        for b in range(B):
            frames = video[b]
            seg_ranges = iou_seg[b]
            pooled_segments = []

            for (s, e) in seg_ranges:
                # 경계가 유효한 경우에만 평균
                if e > s and e <= T:
                    pooled_segments.append(frames[s:e].mean(dim=0))

            # segment가 하나도 없을 경우 전체 mean 사용
            if not pooled_segments:
                pooled_segments = [frames.mean(dim=0)]

            segments_all.append(torch.stack(pooled_segments))

        return segments_all, iou_seg

    def foreground(self, timestamp, num_frames, video_length, device, sharpness=10.0):
        # import pdb; pdb.set_trace()
        # B = timestamp.shape[0]
        B = len(timestamp)
        # device = timestamp.device
        frame_positions = torch.linspace(0, 1, steps=num_frames, device=device)
        frame_mask = torch.zeros(B, num_frames, dtype=torch.float32, device=device)

        for b in range(B):
            for start, end in timestamp[b]:
                if start == 0 and end == 0:
                    continue    
                start_norm = start / video_length[b]
                end_norm = end / video_length[b]
                left = torch.sigmoid(sharpness * (frame_positions - start_norm))
                right = torch.sigmoid(sharpness * (end_norm - frame_positions))
                mask = left * right
                frame_mask[b] = torch.maximum(frame_mask[b], mask)
        return frame_mask

    def spherical_kmeans(self, x, K, iters=10):
        """
        x: [N, D] (float32) — 이미 배치/마스크로 골라낸 프레임 임베딩
        반환:
        centroids: [K, D]  (정규화됨)
        assign:    [N]     (각 프레임의 클러스터 인덱스)
        """
        # L2 정규화
        x = F.normalize(x, dim=-1)

        N, D = x.shape
        # 초기 중심: 무작위 샘플
        if N < K:
            # 극단적으로 프레임 수가 K보다 작으면, 중복 허용해서 채움
            idx = torch.randint(0, N, (K,), device=x.device)
        else:
            idx = torch.randperm(N, device=x.device)[:K]
        centroids = x[idx]  # [K, D]

        for _ in range(iters):
            # 할당 (cosine similarity 최대)
            sim = x @ centroids.t()            # [N, K]
            assign = sim.argmax(dim=1)         # [N]

            # 새 중심 업데이트
            new_centroids = torch.zeros_like(centroids)  # [K, D]
            for k in range(K):
                mask = (assign == k)
                if mask.any():
                    new_centroids[k] = F.normalize(x[mask].mean(dim=0), dim=-1)
                else:
                    # 비어진 클러스터는 임의 재초기화
                    new_centroids[k] = x[torch.randint(0, N, (1,), device=x.device)]

            # 수렴 체크(선택): 필요 없으면 생략 가능
            if torch.allclose(new_centroids, centroids, atol=1e-4, rtol=0):
                centroids = new_centroids
                break
            centroids = new_centroids

        return centroids, assign

    # Vid2Seq 클래스 안에 추가 (spherical_kmeans 아래에 두면정리 좋음)
    def kmeans_segments_simple(self, video_bt, w_bt=None, atts_bt=None, K=None):
        """
        video_bt: [B, T, D]  (프레임 임베딩)
        w_bt:     [B, T]     (saliency in [0,1]; 없으면 균등)
        atts_bt:  [B, T]     (1/0 mask; 유효 프레임만 사용)
        K:        int        (클러스터 개수; None이면 self.K)

        return:
        seg_list:      List[Tensor], 길이 B, 각 원소 shape [K_eff, D]  (K_eff<=K)  -- 기존과 동일
        iou_seg_list:  List[List[List[int]]]  (배치마다, 시간축 run 단위 세그먼트의 프레임 인덱스 리스트)
        """
        import torch
        import torch.nn.functional as F

        B, T, D = video_bt.shape
        if K is None:
            K = getattr(self, "K", 8)

        X_all = F.normalize(video_bt, dim=-1)
        if atts_bt is None:
            m_all = torch.ones(B, T, dtype=torch.bool, device=video_bt.device)
        else:
            m_all = atts_bt.bool()
        if w_bt is None:
            w_all = torch.ones(B, T, device=video_bt.device)
        else:
            w_all = w_bt

        seg_list = []
        iou_seg_list = []

        for b in range(B):
            Xb = X_all[b]   # [T,D]
            mb = m_all[b]   # [T] bool
            wb = w_all[b]   # [T]

            # 유효 프레임이 없으면 기본값
            if mb.sum().item() == 0:
                seg_list.append(Xb.mean(dim=0, keepdim=True))       # [1,D]
                iou_seg_list.append([list(range(T))] if T > 0 else [])
                continue

            # 유효 프레임만 추출
            Xv = Xb[mb]            # [Tv,D]
            wv = wb[mb]            # [Tv]
            Tv = Xv.size(0)
            K_eff = min(K, max(1, Tv))

            # spherical k-means (유저 구현 사용)
            centroids, assign = self.spherical_kmeans(Xv, K_eff, iters=10)  # [K_eff,D], [Tv]

            # ---------------------------------------------
            # (A) 기존 반환: 클러스터별 풀링(변경 없음)
            # ---------------------------------------------
            pooled = []
            for k in range(K_eff):
                mask_k = (assign == k)
                if not mask_k.any():
                    pooled.append(Xv.mean(dim=0, keepdim=True))
                else:
                    Xk = Xv[mask_k]                      # [Nk,D]
                    wk = wv[mask_k].unsqueeze(-1) + 1e-6 # [Nk,1]
                    vec = (wk * Xk).sum(dim=0) / wk.sum()
                    pooled.append(vec.unsqueeze(0))
            segs = torch.cat(pooled, dim=0)              # [K_eff, D]
            seg_list.append(segs)

            # ---------------------------------------------
            # (B) IoU용 세그먼트: 시간축 run-length 경계로 분할
            # - assign이 바뀌는 지점을 경계로 사용
            # - 원본 시간 인덱스(배치 내)로 반환
            # ---------------------------------------------
            # 유효 프레임의 원본 인덱스
            t_idx = torch.nonzero(mb, as_tuple=False).squeeze(1)  # [Tv]

            # run-length 인코딩으로 경계 계산
            runs = []
            start = 0
            for i in range(1, Tv):
                if assign[i].item() != assign[i-1].item():
                    runs.append((start, i))
                    start = i
            runs.append((start, Tv))  # 마지막 구간

            # (선택) 너무 짧은 세그먼트 병합하고 싶으면 아래 주석 해제
            # min_len = 1
            # merged = []
            # for s, e in runs:
            #     if merged and (e - s) < min_len:
            #         ps, pe = merged[-1]
            #         merged[-1] = (ps, e)   # 이전 구간과 병합
            #     else:
            #         merged.append((s, e))
            # runs = merged

            # 각 run을 원본 프레임 인덱스 리스트로 변환
            iou_segs_b = [t_idx[s:e].tolist() for (s, e) in runs if e > s]
            # 길이 1도 포함하려면 위 if e > s 를 제거하고 그대로 사용
            iou_seg_list.append(iou_segs_b)

        return seg_list, iou_seg_list


    def retrieval(self, video_list, memory_bank, args, anchor_ids_list=None):
        soft_k = args.soft_k

        if getattr(args, "memory_bank_type", "").lower() == "hicm2":
            do_norm = True
            retrieved_embeds_list = []
            for segs in video_list:                     # segs: [S_b, D]
                # import pdb; pdb.set_trace()
                if segs is None or segs.numel() == 0:
                    # 비어 있으면 더미 1개
                    retrieved_embeds_list.append(
                        torch.zeros(1, self.hidden_dim, device=self.anchors.device, dtype=self.anchors.dtype)
                    )
                    continue

                # (선택) 정규화
                q_mat = F.normalize(segs, dim=-1) if do_norm else segs
                S_b, _ = q_mat.shape
                out_b = []

                # 세그먼트별로 계층 검색
                for i in range(S_b):
                    q = q_mat[i]
                    if do_norm:
                        q = F.normalize(q.unsqueeze(0), dim=-1).squeeze(0)
                    emb = self.hierarchical_memory_search(q, soft_k, memory_bank)  # [1, D]
                    out_b.append(emb)
                out_b = torch.cat(out_b, dim=0)  # [S_b, D]
                retrieved_embeds_list.append(out_b)

            # pad + mask
            lengths = torch.tensor([e.size(0) for e in retrieved_embeds_list], device=retrieved_embeds_list[0].device)
            max_len = int(lengths.max().item())
            D = retrieved_embeds_list[0].size(-1)
            padded_embeds = torch.stack([
                F.pad(e, (0, 0, 0, max_len - e.size(0))) for e in retrieved_embeds_list
            ], dim=0)  # [B, L_max, D]
            mask = (torch.arange(max_len, device=padded_embeds.device).unsqueeze(0) < lengths.unsqueeze(1))  # [B, L_max]
            return padded_embeds.float(), mask


        else:
            vocab_features = torch.tensor(memory_bank['vid_sent_embeds'], device=video_list[0].device).float()
            sentences = memory_bank['vide_sent_captions']
            vocab_features = vocab_features.squeeze(1)
            vocab_features = F.normalize(vocab_features, dim=-1)

            if hasattr(args, 'subset_ratio') and args.subset_ratio < 1.0:
                total = vocab_features.size(0)
                subset_size = max(1, int(total * args.subset_ratio))
                indices = random.sample(range(total), subset_size)
                vocab_features = vocab_features[indices]
                sentences = [sentences[i] for i in indices]

            retrieved_sentences_all = []
            retrieved_embeds_list = []
            retrieved_anchor_ids_all = []  # ★ 추가

            for b, video in enumerate(video_list):
                video = F.normalize(video, dim=-1)
                scores = torch.matmul(video, vocab_features.T)
                _, indices = torch.topk(scores, k=soft_k, dim=-1)
                retrieved_sentences = [[sentences[idx] for idx in ind.tolist()] for ind in indices]
                retrieved_sentences_all.append(retrieved_sentences)
                topk_embeds = vocab_features[indices]
                retrieved_embeds = topk_embeds.mean(dim=1)
                retrieved_embeds_list.append(retrieved_embeds)
                if anchor_ids_list is not None:
                    retrieved_anchor_ids_all.append(anchor_ids_list[b])


            lengths = torch.tensor([e.size(0) for e in retrieved_embeds_list])
            max_len = max(lengths.tolist())
            padded_embeds = torch.stack([
                F.pad(e, (0, 0, 0, max_len - e.size(0))) for e in retrieved_embeds_list
            ], dim=0)
            mask = torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)

            if anchor_ids_list is not None:
                return padded_embeds, mask ,retrieved_sentences_all, retrieved_anchor_ids_all
            else:
                return padded_embeds, mask, None, None


    def forward(self, video, input_tokenized, output_tokenized, timestamp, duration, sal_target = None, mode='None',uns_video=None, memory_bank=None, epoch=0):
        if self.use_video:
            if isinstance(video, dict):  # cached
                video, video_origin, atts_vis = video["video"], video["video_origin"].clone(), video["atts_vis"] #두번째 video는 video clone
            else:
                video_origin = video.clone()
                video = self.visual_encoder(video)  # B T D #!여기에서 temporal encoding이 되는거 
                if self.proj_v2t is not None:
                    video = self.proj_v2t(video)
                atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)
                # video_origin = video.clone()
            
            B, T, D = video.shape
            video_clone = video.clone() #!여기에 window를 만들어도 됨

            if self.args.self_attn:
                vid_zero = torch.zeros_like(video)
                count = torch.zeros(B, T, 1, device=video.device)

                for window in self.window_sizes:
                    for batch_idx in range(B):
                        sample_not_ln = video[batch_idx]  # [T, D]
                        sample = video[batch_idx]

                        for i in range(0, T - window + 1, window):
                            mini_batch = sample[i : i + window, :]  # [window, D]
                            mini_batch_not_ln = sample_not_ln[i : i + window, :]
                            q = k = mini_batch_not_ln
                            v = mini_batch.clone()
                            scale = D**0.5

                            attn_scores = torch.matmul(q, k.T) / scale  # [window, window]
                            attn_weights = F.softmax(
                                attn_scores, dim=-1
                            )  # [window, window]
                            attn_output = torch.matmul(attn_weights, v)  # [window, D]

                            vid_zero[batch_idx, i : i + window, :] += attn_output
                            count[batch_idx, i : i + window, :] += 1


                vid_zero = vid_zero / (count + 1e-6)
                vid_zero = self.norm(vid_zero)
                video_clone = video + vid_zero


            video_dict = {"video": video_clone, "video_origin": video_origin, "atts_vis": atts_vis} 

            if self.args.use_saliency: #or self.args.use_ret: # and self.args.sali4vid == False:
                _, video_global = self.visual_encoder.forward_with_global(video_clone, mode = "training")  # B T D

                saliency_score = (
                    torch.sum(self.saliency_proj1(video_clone) * self.saliency_proj2(video_global).unsqueeze(1), dim=-1)
                    / np.sqrt(self.hidden_dim)
                )  # [B, T]

                ############여기부터 self attention
                saliency_scores = {"saliency_scores": saliency_score, "video_mask": atts_vis}

                s_n = (saliency_score - saliency_score.mean(dim=1, keepdim=True)) \
                    / (saliency_score.std(dim=1, keepdim=True) + 1e-6)
                w = torch.sigmoid(s_n)   # [B, T], 각 프레임별 weight (0~1)
                # w = torch.softmax(s_n, dim=1)   # [B, T], 각 프레임별 확률 (합 = 1)

                # video_segs = self.kmeans_segments_simple(
                #     video_origin,   # 실험적으로 video_clone 혹은 video로 바꿔도 됨
                #     w_bt=w,
                #     atts_bt=atts_vis,
                #     K=self.K
                # )

                # # # 점수/Top-K 없이, 배치별 클러스터 임베딩들을 그대로 retrieval 입력으로 사용
                # video_list = [segs for segs in video_segs]   # 각 segs: [K_eff, D]
                # start_time = time()
                video_segs, seg_scores, assign_list, frames_by_anchor_list, segments_meta_list = asot_segments_aux(
                    video_origin,   # [B,T,D]
                    w,             # [B,T]
                    atts_vis,      # [B,T]
                    self.args,
                    self.anchors,   # <<< NEW
                    asot_mode="train"

                )

                topk = getattr(self.args, "asot_topk_for_retrieval", 8)

                video_list = []
                anchor_ids_list = []   # ★ 추가: 세그먼트별 앵커 ID 보관

                for b in range(len(video_segs)):
                    emb = video_segs[b]        # [N_seg, D]
                    scr = seg_scores[b]        # [N_seg]
                    metas = segments_meta_list[b]  # ★ 각 세그먼트 메타(여기에 "anchor"가 있음)
                    if emb.dim() == 1: 
                        emb = emb.unsqueeze(0)
                    k = min(topk, emb.size(0))

                    top_idx = torch.topk(scr, k=k, dim=0).indices  # [k]
                    video_list.append(emb[top_idx])                # [k, D]

                    # ★ 선택된 세그먼트들의 앵커 id 수집
                    anchor_ids = [int(metas[i]["anchor"]) for i in top_idx.tolist()]
                    anchor_ids_list.append(anchor_ids)

            if self.args.use_ret:
                # video_list = self.cluster_momentum_adaptive(video_origin, tau=1.0)
                # #! cluster_video 대신 saliency score로 clustering을 하는거야
                retrieval_emb, ret_mask, _, _ = self.retrieval(video_list, memory_bank, self.args)
                retrieval_emb = self.proj_r2t(retrieval_emb)
                atts_ret = torch.ones(retrieval_emb.size()[:-1], dtype=torch.long).to(video.device)
                # atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)

            ##############not FiLM gating########
            if self.args.use_saliency:
                saliency_tok = self.saliency_tok_proj(saliency_score.unsqueeze(-1))  # [B, T, D]
                atts_sal = torch.ones(saliency_tok.size()[:-1], dtype=torch.long, device=video.device)
                video_input = torch.cat([video, saliency_tok], dim=1)                # [B, 2T, D]
                atts_vis_input = torch.cat([atts_vis, atts_sal], dim=1)
                # video_input = video_clone.clone() #torch.cat([video_clone, saliency_tok], dim=1)                # [B, 2T, D]
                # atts_vis_input = atts_vis.clone() #torch.cat([atts_vis, atts_sal], dim=1)
                ##############not FiLM gating########
            else:
                video_input = video.clone()
                # video_input = video.clone() #torch.cat([video_clone, saliency_tok], dim=1)                # [B, 2T, D]
                atts_vis_input = atts_vis.clone() #torch.cat([atts_vis, atts_sal], dim=1)

            # else:
            #     # atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)
            #     atts_vis_input = atts_vis.clone()
            #     cluster_video = self.cluster_momentum_adaptive(video, tau=1.0)
            #     retrieval_emb, mask = self.retrieval(cluster_video, memory_bank, self.args)
            #     timestamps_mask = self.foreground(timestamp, 100, duration, video.device).to(video.device)
            #     video_input = self.visual_encoder(video) * timestamps_mask.unsqueeze(-1)
            #     retrieval_emb = self.proj_r2t(retrieval_emb)
            #     atts_ret = torch.ones(retrieval_emb.size()[:-1], dtype=torch.long).to(video.device)
            #     video_dict = {"video": video_input, "video_origin":video, "atts_vis": atts_vis}


        else:
            video_dict = None

        if self.use_speech:
            text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])  # B L D
            encoded = self.t5_model.encoder(
                attention_mask=input_tokenized['attention_mask'],
                inputs_embeds=text,
            )

        if self.use_video and self.use_speech:

            # atts_vis_input = torch.ones(video_input.size()[:-1], dtype=torch.long, device=video.device)  # [B, 2T(+R)]
            if self.args.use_ret:
                encoded.last_hidden_state = torch.cat([video_input, retrieval_emb, encoded.last_hidden_state], dim=1)
                encoder_atts = torch.cat([atts_vis_input, atts_ret, input_tokenized['attention_mask']], dim=1)
            else:
                encoded.last_hidden_state = torch.cat([video_input, encoded.last_hidden_state], dim=1)
                encoder_atts = torch.cat([atts_vis_input, input_tokenized['attention_mask']], dim=1)

        elif self.use_video: 
            # import pdb; pdb.set_trace()
            if self.args.use_ret:
                hidden_state = torch.cat([video_input, retrieval_emb], dim=1)
                encoded = BaseModelOutput(last_hidden_state=hidden_state)
                encoder_atts = torch.cat([atts_vis_input, atts_ret], dim=1)
            else:
                hidden_state = video_input
                encoded = BaseModelOutput(last_hidden_state=hidden_state)
                encoder_atts = atts_vis_input
        elif self.use_speech:
            encoder_atts = input_tokenized['attention_mask']

        targets = output_tokenized['input_ids'].masked_fill(
            output_tokenized['input_ids'] == self.t5_tokenizer.pad_token_id, -100
        )
        outputs = self.t5_model(
            encoder_outputs=encoded,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokenized['attention_mask'],
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        if sal_target is not None:
            sal_loss = loss_saliency(self.args, saliency_scores, sal_target)
            loss += sal_loss * self.args.alpha

        return {"loss": loss}, video_dict # ,saliency_scores

    @torch.no_grad()
    def generate(
            self,
            video,
            input_tokenized,
            use_nucleus_sampling=False,
            num_beams=4,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_captions=1,
            temperature=1,
            memory_bank=None, 
            gt_duration=None,
            sal_target=None,
    ):
        """
        Args:
            video (torch.Tensor): A tensor of shape (batch_size, T, D)
            input_tokenized (torch.Tensor): A tensor of shape (batch_size, L)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        video_origin = video.clone()
        video = self.visual_encoder(video)  # B T D
        # video_origin = video.clone()
        if self.proj_v2t is not None:
            video = self.proj_v2t(video)
        # video, video_global = self.visual_encoder.forward_with_global(video)
        atts_vis = torch.ones(video.size()[:-1], dtype=torch.long).to(video.device)
        atts_vis_input = atts_vis.clone()

        B, T, D = video.shape
        video_clone = video.clone() #!여기에 window를 만들어도 됨
        # video_origin = video.clone()
        # start_time = time()
        if self.args.self_attn:
            vid_zero = torch.zeros_like(video)
            count = torch.zeros(B, T, 1, device=video.device)

            for window in self.window_sizes:
                for batch_idx in range(B):
                    sample_not_ln = video[batch_idx]  # [T, D]
                    sample = video[batch_idx]

                    for i in range(0, T - window + 1, window):
                        mini_batch = sample[i : i + window, :]  # [window, D]
                        mini_batch_not_ln = sample_not_ln[i : i + window, :]
                        q = k = mini_batch_not_ln
                        v = mini_batch.clone()
                        scale = D**0.5

                        attn_scores = torch.matmul(q, k.T) / scale  # [window, window]
                        attn_weights = F.softmax(
                            attn_scores, dim=-1
                        )  # [window, window]
                        attn_output = torch.matmul(attn_weights, v)  # [window, D]

                        vid_zero[batch_idx, i : i + window, :] += attn_output
                        count[batch_idx, i : i + window, :] += 1


            vid_zero = vid_zero / (count + 1e-6)
            vid_zero = self.norm(vid_zero)
            video_clone = video + vid_zero

        # eval_time = time() - start_time
        # print(f"evaluation took {eval_time} seconds")
        # import pdb; pdb.set_trace()
        # video, video_global = self.visual_encoder.forward_with_global(video_clone)  # B T D
        # vid_zero, video_global = self.visual_encoder.forward_with_global(vid_zero)  # B T D
        # video_clone = video + vid_zero


        # import pdb; pdb.set_trace()
        # video_global = video_clone.mean(dim=1)  # [B, D]

        if self.args.use_saliency : # or (self.args.use_ret and self.args.sali4vid == False)
            video_, video_global = self.visual_encoder.forward_with_global(video_clone)  # B T D
            saliency_score = (
                torch.sum(self.saliency_proj1(video_clone) * self.saliency_proj2(video_global).unsqueeze(1), dim=-1)
                / np.sqrt(self.hidden_dim)
            )  # [B, T]
            # import pdb; pdb.set_trace()
            # saliency_score = sal_target["saliency_all_labels"].float().to(video.device)
            # w = saliency_score.float().clone()


            s_n = (saliency_score - saliency_score.mean(dim=1, keepdim=True)) \
                / (saliency_score.std(dim=1, keepdim=True) + 1e-6)
            w = torch.sigmoid(s_n)   # [B, T], 각 프레임별 weight (0~1)
            # w = torch.softmax(s_n, dim=1)   # [B, T], 각 프레임별 확률 (합 = 1)
            # saliency_score = torch.flip(saliency_score, dims=[1])
            # w = 1.0 - w  # 역전
            # w = s_n

            # start_time = time()
            # video_segs, iou_seg = self.kmeans_segments_simple(
            #     video_origin,   # 실험적으로 video_clone 혹은 video로 바꿔도 됨
            #     atts_bt=atts_vis,
            #     K=10
            # )

            # # 점수/Top-K 없이, 배치별 클러스터 임베딩들을 그대로 retrieval 입력으로 사용
            # video_list = [segs for segs in video_segs]   # 각 segs: [K_eff, D]


            video_segs, seg_scores, assign_list, anchor_info, segments_meta_list = asot_segments_aux(
                video_origin,   # [B,T,D]
                w,             # [B,T]
                atts_vis,      # [B,T]
                self.args,
                self.anchors,
                asot_mode="infer"
            )

            segment_frames_list = anchor_info[0]
            segment_anchor_ids_list = anchor_info[1]

            topk = getattr(self.args, "asot_topk_for_retrieval", 8)

            video_list = []                              # [B] 각 배치의 [k, D]
            selected_frames_per_batch = []               # [B][k]   (세그먼트별 frame indices)
            selected_anchor_ids_per_batch = []           # [B][k]   (세그먼트별 anchor id)

            for b in range(len(video_segs)):
                emb = video_segs[b]                      # [N_seg, D]
                scr = seg_scores[b]                      # [N_seg]
                if emb.dim() == 1: 
                    emb = emb.unsqueeze(0)
                k = min(topk, emb.size(0))

                top_idx = torch.topk(scr, k=k, dim=0).indices  # [k]

                # 1) 인코더에 넣을 선택된 세그먼트 임베딩
                video_list.append(emb[top_idx])                # [k, D]

                # 2) ★ 같은 순서로 앵커 ID/프레임 인덱스를 추출(세그먼트 단위 정렬 일치)
                sel_anchor_ids = segment_anchor_ids_list[b] #[i]  for i in top_idx.tolist()]
                sel_frames     = segment_frames_list[b] #[i]     for i in top_idx.tolist()]
                # import pdb; pdb.set_trace()
                selected_anchor_ids_per_batch.append(sel_anchor_ids)
                selected_frames_per_batch.append(sel_frames)

        if self.args.use_ret:
            # video_list, iou_seg = self.cluster_momentum_adaptive(video_origin, iou_seg=gt_duration)
            # #! cluster_video 대신 saliency score로 clustering을 하는거야 sentences, sent_idx 
            retrieval_emb, ret_mask, sentences, sent_idx  = self.retrieval(video_list, memory_bank, self.args, anchor_ids_list=None) #selected_anchor_ids_per_batch)
            # retrieval_emb = self.ret(video_list, memory_bank)  # [B
            retrieval_emb = self.proj_r2t(retrieval_emb)
            atts_ret = torch.ones(retrieval_emb.size()[:-1], dtype=torch.long).to(video.device)

        ##############not FiLM gating########
        #이 부분만 video clone 말고 video 가 쓰였을 때 성능이 좋아진다고
        if self.args.use_saliency:

            saliency_tok = self.saliency_tok_proj(saliency_score.unsqueeze(-1))  # [B, T, D]
            # saliency_tok = torch.zeros_like(saliency_tok_2)
            atts_sal = torch.ones(saliency_tok.size()[:-1], dtype=torch.long, device=video.device)

            video_input = torch.cat([video_, saliency_tok], dim=1)                # [B, 2T, D]
            atts_vis_input = torch.cat([atts_vis, atts_sal], dim=1)
            ##############not FiLM gating########
        else:
            video_input = video_.clone() #* w.unsqueeze(-1)
            atts_vis_input = atts_vis.clone() #torch.cat([atts_vis, atts_sal], dim=1)

        if self.use_speech:
            text = self.t5_model.encoder.embed_tokens(input_tokenized['input_ids'])  # B L D
            encoded = self.t5_model.encoder(
                attention_mask=input_tokenized['attention_mask'],
                inputs_embeds=text,
            )

        if self.use_video and self.use_speech:
            if self.args.use_ret:
                encoded.last_hidden_state = torch.cat([video_input, retrieval_emb, encoded.last_hidden_state], dim=1)
                encoder_atts = torch.cat([atts_vis_input, atts_ret, input_tokenized['attention_mask']], dim=1)
            else:
                encoded.last_hidden_state = torch.cat([video_input, encoded.last_hidden_state], dim=1)
                encoder_atts = torch.cat([atts_vis_input, input_tokenized['attention_mask']], dim=1)
        elif self.use_video:
            if self.args.use_ret:
                hidden_state = torch.cat([video_input, retrieval_emb], dim=1)
                encoded = BaseModelOutput(last_hidden_state=hidden_state)
                encoder_atts = torch.cat([atts_vis_input, atts_ret], dim=1)
            else:
                hidden_state = video_input
                encoded = BaseModelOutput(last_hidden_state=hidden_state)
                encoder_atts = atts_vis_input
        elif self.use_speech:
            encoder_atts = input_tokenized['attention_mask']

        outputs = self.t5_model.generate(
                encoder_outputs=encoded,
                attention_mask=encoder_atts,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
                # output_attentions=True,
                # return_dict_in_generate=True,
                # output_scores=False,
        )
        
        output_text = self.t5_tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        feature_dict = {
            "video_origin": video_origin.detach().cpu(),   # [B, T, D]
            "video": video.detach().cpu(),                 # encoder 이후 feature
            "saliency_score": saliency_score.detach().cpu() if (self.args.use_saliency or (self.args.use_ret and self.args.sali4vid == False)) else None,
    # "cross_attn": A.detach().cpu(),                # [B, L_dec,
        }
        seg_scores, selected_frames_per_batch, selected_anchor_ids_per_batch, sentences, sent_idx = None, None, None, None, None

        return output_text, [seg_scores, selected_frames_per_batch, selected_anchor_ids_per_batch, sentences, sent_idx], feature_dict