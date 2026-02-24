
import torch
import torch.nn.functional as F

# def loss_saliency(args, outputs, targets, log=True):
#     """
#     Binary saliency 학습 (rank 제거, listwise 버전).
#     - 라벨/마스크/스코어 길이 정렬
#     - video_mask==1인 위치만 사용
#     - 양성(라벨==1) 위치의 softmax log-prob를 최대화
#     """
#     if "saliency_all_labels" not in targets:
#         # 관례적으로 0 텐서 반환
#         return {"loss_saliency": torch.tensor(0.0, device=outputs["saliency_scores"].device)}

#     # (N, L)
#     saliency_scores = outputs["saliency_scores"]          # logits
#     vid_token_mask  = outputs["video_mask"]               # 0/1
#     labels          = targets["saliency_all_labels"]      # 0/1 (float 또는 int)
#     labels = labels.to(saliency_scores.device)

#     # ---- 길이 정렬 ----
#     N, Ls = saliency_scores.shape
#     Ll = labels.shape[1]
#     Lm = vid_token_mask.shape[1]
#     L = min(Ls, Ll, Lm)
#     if (Ls != L) or (Ll != L) or (Lm != L):
#         saliency_scores = saliency_scores[:, :L]
#         labels          = labels[:, :L]
#         vid_token_mask  = vid_token_mask[:, :L]

#     # ---- 마스크/라벨 정리 ----
#     valid_mask = (vid_token_mask > 0)            # (N, L) bool
#     pos_mask   = (labels > 0) & valid_mask       # (N, L) bool

#     # 유효 토큰 외에는 큰 음수로 가려 softmax에서 제외
#     logits = torch.where(
#         valid_mask,
#         saliency_scores,
#         saliency_scores.new_full(saliency_scores.shape, -1e3)
#     )

#     # 온도(선택): 필요 없다면 tau=1.0으로 두면 됨
#     tau = args.tau
#     logits = logits / tau

#     # 안정화 후 log-prob
#     logits = logits - logits.max(dim=1, keepdim=True)[0]
#     log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True) + 1e-6)

#     # 배치별 양성 평균 log-prob (양성 없는 샘플은 무시)
#     pos_float = pos_mask.float()
#     has_pos   = (pos_float.sum(1) > 0).float()               # (N,)
#     mean_logp_pos = (pos_float * log_prob).sum(1) / (pos_float.sum(1) + 1e-6)

#     loss = (-(mean_logp_pos) * has_pos).sum() / (has_pos.sum() + 1e-6)

#     return loss


# def loss_saliency(args, outputs, targets, log=True):
#     """
#     Saliency listwise loss.
#     - args.dense_sal == False: 기존 binary (라벨 > 0 → 1)
#     - args.dense_sal == True:  가우시안 소프트 라벨을 가중치로 사용
#     """
#     if "saliency_all_labels" not in targets:
#         return {"loss_saliency": torch.tensor(0.0, device=outputs["saliency_scores"].device)}

#     # (N, L)
#     saliency_scores = outputs["saliency_scores"]          # logits
#     vid_token_mask  = outputs["video_mask"]               # 0/1
#     labels          = targets["saliency_all_labels"]      # binary or gaussian
#     labels = labels.to(saliency_scores.device)

#     # ---- 길이 정렬 ----
#     N, Ls = saliency_scores.shape
#     Ll = labels.shape[1]
#     Lm = vid_token_mask.shape[1]
#     L = min(Ls, Ll, Lm)
#     if (Ls != L) or (Ll != L) or (Lm != L):
#         saliency_scores = saliency_scores[:, :L]
#         labels          = labels[:, :L]
#         vid_token_mask  = vid_token_mask[:, :L]

#     # ---- 마스크/라벨 정리 ----
#     valid_mask = (vid_token_mask > 0)            # (N, L) bool

#     is_dense = getattr(args, 'dense_sal', False)
#     if is_dense:
#         # 가우시안 라벨: 연속 가중치 그대로 사용, 마스크 적용
#         weights = labels.float() * valid_mask.float()       # (N, L)
#         pos_mask = (weights > 0)
#     else:
#         # 기존 binary
#         pos_mask = (labels > 0) & valid_mask
#         weights = pos_mask.float()

#     # 유효 토큰 외에는 큰 음수로 가려 softmax에서 제외
#     logits = torch.where(
#         valid_mask,
#         saliency_scores,
#         saliency_scores.new_full(saliency_scores.shape, -1e3)
#     )

#     # 온도
#     tau = args.tau
#     logits = logits / tau

#     # 안정화 후 log-prob
#     logits = logits - logits.max(dim=1, keepdim=True)[0]
#     log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True) + 1e-6)

#     # 배치별 가중 평균 log-prob (양성 없는 샘플은 무시)
#     w_sum   = weights.sum(1)                                  # (N,)
#     has_pos = (w_sum > 0).float()                             # (N,)
#     mean_logp = (weights * log_prob).sum(1) / (w_sum + 1e-6)

#     loss = (-(mean_logp) * has_pos).sum() / (has_pos.sum() + 1e-6)

    # return loss

def loss_saliency(args, outputs, targets, log=True):
    """
    Saliency listwise loss.
    - binary:   라벨 > 0 → 1
    - gaussian: 가우시안 소프트 라벨을 가중치로 사용
    - sigmoid:  sigmoid 소프트 라벨을 가중치로 사용
    """
    if "saliency_all_labels" not in targets:
        return {"loss_saliency": torch.tensor(0.0, device=outputs["saliency_scores"].device)}

    # (N, L)
    saliency_scores = outputs["saliency_scores"]          # logits
    vid_token_mask  = outputs["video_mask"]               # 0/1
    labels          = targets["saliency_all_labels"]      # binary or soft
    labels = labels.to(saliency_scores.device)

    # ---- 길이 정렬 ----
    N, Ls = saliency_scores.shape
    Ll = labels.shape[1]
    Lm = vid_token_mask.shape[1]
    L = min(Ls, Ll, Lm)
    if (Ls != L) or (Ll != L) or (Lm != L):
        saliency_scores = saliency_scores[:, :L]
        labels          = labels[:, :L]
        vid_token_mask  = vid_token_mask[:, :L]

    # ---- 마스크/라벨 정리 ----
    valid_mask = (vid_token_mask > 0)            # (N, L) bool

    dense_sal = getattr(args, 'dense_sal', 'binary')
    if dense_sal in ('gaussian', 'sigmoid'):
        # soft label: 연속 가중치 그대로 사용
        weights = labels.float() * valid_mask.float()       # (N, L)
        pos_mask = (weights > 0)
    else:
        # binary
        pos_mask = (labels > 0) & valid_mask
        weights = pos_mask.float()

    # 유효 토큰 외에는 큰 음수로 가려 softmax에서 제외
    logits = torch.where(
        valid_mask,
        saliency_scores,
        saliency_scores.new_full(saliency_scores.shape, -1e3)
    )

    # 온도
    tau = args.tau
    logits = logits / tau

    # 안정화 후 log-prob
    logits = logits - logits.max(dim=1, keepdim=True)[0]
    log_prob = logits - torch.log(torch.exp(logits).sum(1, keepdim=True) + 1e-6)

    # 배치별 가중 평균 log-prob (양성 없는 샘플은 무시)
    w_sum   = weights.sum(1)                                  # (N,)
    has_pos = (w_sum > 0).float()                             # (N,)
    mean_logp = (weights * log_prob).sum(1) / (w_sum + 1e-6)

    loss = (-(mean_logp) * has_pos).sum() / (has_pos.sum() + 1e-6)

    return loss