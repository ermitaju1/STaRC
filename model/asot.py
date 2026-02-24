import torch
import torch.nn.functional as F
import math

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def _kot_cost_from_anchors(X_b, w_b, anchors, mu: float = 0.05):
    """
    X_b: [T,D] normalized
    w_b: [T]   saliency in [0,1]
    anchors: [K,D] normalized
    mu: small bias scale (0~0.1 권장)
    """
    Ck = 1.0 - X_b @ anchors.t()  # [T,K]
    if mu and mu != 0.0:
        # 주의: 행(프레임)별 상수 이동은 Sinkhorn 정규화에서 상쇄될 수 있음
        Ck = Ck - mu * (w_b - w_b.mean()).unsqueeze(-1)

    T, D, K = 100, 128, 6  # T: 프레임 수, D: feature dim, K: anchor 개수

    # =========================
    # 2️⃣ affinity matrix 계산
    # =========================
    # affinity = X_b @ anchors.t()  # cosine similarity (range: [-1, 1])
    # affinity = affinity.cpu().numpy()

    # =========================
    # 3️⃣ 시각화
    # =========================
    # plt.figure(figsize=(7, 5))
    # plt.imshow(affinity, aspect='auto', origin='lower', interpolation='nearest')
    # plt.xlabel("Anchor (Action) index")
    # plt.ylabel("Frame index")
    # plt.title("Raw frame–action affinity matrix (cosine similarity)")
    # plt.colorbar(label="affinity")

    # plt.tight_layout()
    # plt.savefig("affinity_matrix_demo.pdf", bbox_inches="tight")
    # plt.savefig("affinity_matrix_demo.png", dpi=300, bbox_inches="tight")
    # import pdb; pdb.set_trace()
    return Ck

def potts_decode_from_T(T_b, valid_mask=None, tau=1.5, min_len=6, eps=1e-8):
    T, K = T_b.shape
    dev = T_b.device
    if valid_mask is None:
        valid_mask = torch.ones(T, dtype=torch.bool, device=dev)

    U = -torch.log(T_b.clamp_min(eps))  # unary cost

    dp  = torch.full((T, K), float('inf'), device=dev)
    ptr = torch.full((T, K), -1, dtype=torch.long, device=dev)
    dp[0] = U[0]

    for t in range(1, T):
        keep_cost   = dp[t-1] + U[t]                 # same state
        switch_base = dp[t-1].min().item() + tau     # switch from best
        switch_cost = switch_base + U[t]
        better_keep = keep_cost <= switch_cost
        dp[t]  = torch.where(better_keep, keep_cost, switch_cost)
        argmin_prev = dp[t-1].argmin().item()
        ptr[t] = torch.where(better_keep,
                             torch.arange(K, device=dev),
                             torch.full((K,), argmin_prev, device=dev))
        if not valid_mask[t]:
            dp[t]  = dp[t-1]
            ptr[t] = torch.arange(K, device=dev)

    y = torch.full((T,), -1, dtype=torch.long, device=dev)
    y[T-1] = dp[T-1].argmin()
    for t in range(T-2, -1, -1):
        y[t] = ptr[t+1, y[t+1]]

    # 최소 지속시간 후처리(짧은 run을 이웃으로 흡수)
    if min_len > 1:
        s = 0
        while s < T:
            k = y[s].item()
            e = s + 1
            while e < T and y[e].item() == k: e += 1
            if (e - s) < min_len:
                left  = y[s-1].item() if s-1 >= 0 else k
                right = y[e].item()   if e   < T else k
                repl = left if left == right else (left if left != k else right)
                y[s:e] = repl
            s = e
    y[~valid_mask] = -1
    return y

def asot_segments_aux(video_clone, w, atts_vis, args, anchors_param, asot_mode):
    """
    video_clone: [B,T,D]  (frame embeddings)
    w:           [B,T]    (saliency in [0,1])
    atts_vis:    [B,T]    (1/0 valid mask)
    anchors_param: nn.Parameter [K,D] (learnable anchors, normalized)
    """
    B, T, D = video_clone.shape
    X = F.normalize(video_clone, dim=-1)
    mask_bool = atts_vis.bool()
    K = getattr(args, "asot_K", anchors_param.shape[0])
    
    # hyper
    radius  = getattr(args, "asot_radius", 0.04)
    mu_bias = args.asot_mu_salbias  # 작게 권장 (0~0.1)
    if asot_mode == "infer":
        lam_f   = args.asot_lambda_frames
        eps     = 0.04
        rho_temp= 0.0
        alpha   = 0.6
    else:
        eps     = getattr(args, "asot_eps", 0.07)
        alpha   = getattr(args, "asot_alpha", 0.3)
        rho_temp= 0.0
        lam_f   = args.asot_lambda_frames

    ub_f    = True   # 프레임 마진을 saliency로 강제
    ub_a    = False  # 앵커 쪽은 균등 유지
    lam_a   = 0.12

    # anchors normalize (safety)
    A = F.normalize(anchors_param, dim=-1)

    seg_list = []
    score_list = []
    assign_list = []               # [B] list of LongTensor(T,)  (-1 or 0..K-1)
    frames_by_anchor_list = []     # [B] list of list[K] of LongTensor
    segments_meta_list = []        # [B] list of list[dict]

    # ★ 추가: 세그먼트 단위 결과(같은 단위로 쓰기 위함)
    segment_frames_list = []       # [B] list of list[LongTensor]  각 세그먼트의 프레임 인덱스
    segment_anchor_ids_list = []   # [B] list of list[int]        각 세그먼트의 앵커 ID

    for b in range(B):
        local_mask = mask_bool[b].clone()  # [T] bool

        # (B) KOT 비용 (anchors 기반) + 작은 saliency bias
        Ck_b = _kot_cost_from_anchors(X[b], w[b], A, mu=mu_bias)  # [T,K]
        if rho_temp > 0.0:
            Ck_b = Ck_b + temporal_prior(T, K, rho_temp, Ck_b.device)
        Ck  = Ck_b.unsqueeze(0)                # [1,T,K]
        m_b = local_mask.unsqueeze(0)          # [1,T]

        # saliency 기반 프레임 마진 dx 생성
        w_b = w[b].clone()
        w_pos = (w_b.clamp_min(1e-6)) * local_mask.float()            # [T]
        if w_pos.sum() <= 0:
            w_pos = local_mask.float() + 1e-6
        dx_b = (w_pos / w_pos.sum()).view(1, T, 1)                    # [1,T,1]
        dy_b = torch.ones(1, K, 1, device=Ck.device) / K              # [1,K,1]

        # (C) FGW-OT 최적화
        T_b, _ = segment_asot(
            cost_matrix=Ck, mask=m_b,
            eps=eps, alpha=alpha, radius=radius,
            ub_frames=ub_f, ub_actions=ub_a,
            lambda_frames=lam_f, lambda_actions=lam_a,
            n_iters=(getattr(args, "asot_outer", 25), getattr(args, "asot_sinkhorn", 1)),
            dx=dx_b, dy=dy_b
        )  # [1,T,K]

        T_b = T_b[0]                 # [T,K]
        y = T_b.argmax(dim=-1)       # [T]

        valid_idx = torch.nonzero(local_mask, as_tuple=False).flatten()
        assign_full = torch.full((T,), -1, device=y.device, dtype=torch.long)
        if valid_idx.numel() > 0:
            assign_full[valid_idx] = y[valid_idx]

        # 앵커별 프레임 인덱스(전체 프레임 기준)
        frames_by_anchor = []
        for k_id in range(K):
            frames_by_anchor.append((assign_full == k_id).nonzero(as_tuple=False).flatten())

        if valid_idx.numel() == 0:
            seg_list.append(X[b].mean(dim=0, keepdim=True))
            score_list.append(torch.tensor([0.0], device=X.device))
            segments_meta = []
            assign_list.append(assign_full)
            frames_by_anchor_list.append(frames_by_anchor)
            segments_meta_list.append(segments_meta)
            # 추가 리스트도 빈 값으로
            segment_frames_list.append([])
            segment_anchor_ids_list.append([])
            continue

        # 세그먼트 컷 탐지
        yv = y[valid_idx]; tv = valid_idx
        cuts = [tv[0].item()]
        for i in range(1, yv.numel()):
            if yv[i] != yv[i-1]:
                cuts.append(tv[i].item())
        cuts.append(tv[-1].item() + 1)  # exclusive

        pooled, scores = [], []
        segments_meta = []
        seg_frames = []         # ★ 세그먼트별 프레임 인덱스
        seg_anchor_ids = []     # ★ 세그먼트별 앵커 ID

        for s, e in zip(cuts[:-1], cuts[1:]):
            if e <= s:
                continue
            chunk = X[b, s:e]                         # [L,D]
            ww    = (w[b, s:e] + 1e-6).unsqueeze(-1)  # [L,1]
            pooled_vec = (ww * chunk).sum(dim=0) / ww.sum()
            # pooled_vec = chunk.mean(dim=0)
            pooled.append(pooled_vec)

            k_seg  = y[s].item()
            ot_mass= T_b[s:e, k_seg].sum()
            sal_m  = w[b, s:e].mean()
            length = e - s
            ot_score = ot_mass / max(length, 1)
            score  = ot_score  * math.log1p(length)
            scores.append(score)

            # ★ 같은 단위를 위해 세그먼트별 정보 보관
            seg_frames.append(torch.arange(s, e, device=X.device, dtype=torch.long))
            seg_anchor_ids.append(int(k_seg))

            segments_meta.append({
                "start": int(s),
                "end": int(e),          # exclusive
                "anchor": int(k_seg),
                "score": float(score.detach().item()),
                "ot_mass": float(ot_mass.detach().item()),
                "sal_mean": float(sal_m.detach().item()),
                "length": int(length),
            })

        if not pooled:
            pooled = [X[b].mean(dim=0)]
            scores = [torch.tensor(0.0, device=X.device)]

        seg_list.append(torch.stack(pooled, dim=0))
        score_list.append(torch.stack([torch.as_tensor(s, device=X.device, dtype=torch.float32)
                                       for s in scores], dim=0))
        assign_list.append(assign_full)
        frames_by_anchor_list.append(frames_by_anchor)
        segments_meta_list.append(segments_meta)
        segment_frames_list.append(seg_frames)            # ★ 추가
        segment_anchor_ids_list.append(seg_anchor_ids)    # ★ 추가

    return (
        seg_list,                 # [B] list of Tensor [N_seg, D]
        score_list,               # [B] list of Tensor [N_seg]
        assign_list,              # [B] list of LongTensor [T]
        [segment_frames_list,      # ★ [B] list of list[LongTensor] (세그먼트 기준 프레임 인덱스)
        segment_anchor_ids_list],   # ★ [B] list of list[int]        (세그먼트 기준 앵커 ID)
        segments_meta_list,       # [B] list of list[dict]
    )


def construct_Cv_filter(N, r, device):
    abs_r = int(N * r)
    weights = torch.ones(2 * abs_r + 1, device=device) / r
    weights[abs_r] = 0.
    return weights[None, None, :]


def mult_Cv(Cv_weights, X):
    B, N, K = X.shape
    Y_flat = F.conv1d(X.transpose(1, 2).reshape(-1, 1, N), Cv_weights, padding='same')
    return Y_flat.reshape(B, K, N).transpose(1, 2)


# === gradients ===

def grad_fgw(T, cost_matrix, alpha, Cv):
    T_Ck = T.sum(dim=2, keepdim=True) - T
    return alpha * mult_Cv(Cv, T_Ck) + (1. - alpha) * cost_matrix

def grad_kld(T, p, lambd, axis):
    # p is marginal, dim is marginal axes
    marg = T.sum(dim=axis, keepdim=True)
    return lambd * (torch.log(marg / p + 1e-12) + 1.)

def grad_entropy(T, eps):
    return - torch.log(T + 1e-12) * eps


# === Sinkhorn projection for balanced case ===

def project_to_polytope_KL(cost_matrix, mask, eps, dx, dy, n_iters=10, stable_thres=7.):
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
    dual_pot = torch.exp(-cost_matrix / eps) * mask.unsqueeze(2)
    dual_pot = dual_pot / dual_pot.max()
    b = torch.ones((B, K, 1), device=dev)
    u = torch.zeros((B, N, 1), device=dev)
    v = torch.zeros((K, 1), device=dev)

    for i in range(n_iters):
        a = dx / (dual_pot @ b)
        a = torch.nan_to_num(a, posinf=0., neginf=0.)
        b = dy / (dual_pot.transpose(1, 2) @ a)
        b = torch.nan_to_num(b, posinf=0., neginf=0.)
        if torch.any(torch.log(a).abs() > stable_thres) or torch.any(torch.log(b).abs() > stable_thres):
            if i != n_iters - 1:
                u = torch.nan_to_num(u + eps * torch.log(a), posinf=0., neginf=0.)
                v = torch.nan_to_num(v + eps * torch.log(b), posinf=0., neginf=0.)
                dual_pot = torch.exp((u + v.transpose(1, 2) - cost_matrix) / eps) * mask.unsqueeze(2)
                b = torch.ones_like(b)
    T = a * dual_pot * b.transpose(1, 2)
    return T


# === objective ===

def kld(a, b, eps=1e-10):
    return (a * torch.log(a / b + eps)).sum(dim=1)

def entropy(T, eps=1e-10):
    return (-T * torch.log(T + eps) + T).sum(dim=(1, 2))

def asot_objective(T, cost_matrix, eps, alpha, radius, ub_frames, ub_actions,
                   lambda_frames, lambda_actions, mask=None, dx=None, dy=None):
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
        
    if mask is None:
        mask = torch.full((B, N), 1, dtype=bool, device=dev)
    nnz = mask.sum(dim=1)
    T_mask = T * mask.unsqueeze(2)
    
    # FGW
    Cv = construct_Cv_filter(N, radius, dev)
    fgw_obj = (grad_fgw(T_mask, cost_matrix, alpha, Cv) * T_mask).sum(dim=(1, 2))
    
    # Marginals (allow external dx/dy; fallback to uniform)
    if dy is None:
        dy_eval = torch.ones((B, K), device=dev) / K
    else:
        dy_eval = dy.squeeze(-1)  # [B,K]
    if dx is None:
        dx_eval = torch.ones((B, N), device=dev) / nnz[:, None]
    else:
        dx_eval = dx.squeeze(-1)  # [B,N]
    
    frames_marg = T_mask.sum(dim=2)   # [B,N]
    frames_ub_penalty = kld(frames_marg, dx_eval) * lambda_frames
    actions_marg = T_mask.sum(dim=1)  # [B,K]
    actions_ub_penalty = kld(actions_marg, dy_eval) * lambda_actions
    
    ub = torch.zeros(B, device=dev)
    if ub_frames:
        ub += frames_ub_penalty
    if ub_actions:
        ub += actions_ub_penalty
    
    # entropy reg
    entr = -eps * entropy(T)
    
    # objective
    obj = 0.5 * fgw_obj + ub + entr
    
    return obj


# === solver ===

def segment_asot(cost_matrix, mask=None, eps=0.07, alpha=0.3, radius=0.04, ub_frames=False,
                 ub_actions=True, lambda_frames=0.1, lambda_actions=0.05, n_iters=(25, 1),
                 stable_thres=7., step_size=None, dx=None, dy=None):
    dev = cost_matrix.device
    B, N, K = cost_matrix.shape
    if mask is None:
        mask = torch.full((B, N), 1, dtype=bool, device=dev)
    nnz = mask.sum(dim=1)

    # 초기 마진: 외부 전달 값 우선, 없으면 균등
    if dy is None:
        dy = torch.ones((B, K, 1), device=dev) / K
    if dx is None:
        dx = torch.ones((B, N, 1), device=dev) / nnz[:, None, None]

    # 초기화
    T = dx * dy.transpose(1, 2)
    T = T * mask.unsqueeze(2)
    
    Cv = construct_Cv_filter(N, radius, dev)
    
    obj_trace = []
    it = 0

    while True:
        with torch.no_grad():
            obj = asot_objective(T, cost_matrix, eps, alpha, radius, ub_frames, ub_actions,
                                 lambda_frames, lambda_actions, mask=mask, dx=dx, dy=dy)
        obj_trace.append(obj)
        
        if it >= n_iters[0]:
            break
        
        # gradient for mirror descent
        fgw_cost_matrix = grad_fgw(T, cost_matrix, alpha, Cv)
        grad_obj = fgw_cost_matrix - grad_entropy(T, eps)
        if ub_frames:
            grad_obj += grad_kld(T, dx, lambda_frames, 2)
        if ub_actions:
            grad_obj += grad_kld(T, dy.transpose(1, 2), lambda_actions, 1)
        
        # step-size auto-calibration
        if it == 0 and step_size is None:
            step_size = 4. / grad_obj.max().item()

        # mirror descent
        T = T * torch.exp(-step_size * grad_obj)
        
        # projection / normalization
        if not ub_frames and not ub_actions:
            T = project_to_polytope_KL(fgw_cost_matrix, mask, eps, dx, dy,
                                       n_iters=n_iters[1], stable_thres=stable_thres)
        elif not ub_frames:
            T /= T.sum(dim=2, keepdim=True)
            T = torch.nan_to_num(T, posinf=0., neginf=0.)
            T = T * dx
        elif not ub_actions:
            T /= T.sum(dim=1, keepdim=True)
            T = torch.nan_to_num(T, posinf=0., neginf=0.)
            T = T * dy.transpose(1, 2)
        
        it += 1
    
    T = T * nnz[:, None, None]  # rescale so marginals per frame sum to 1
    obj_trace = torch.cat(obj_trace)
    return T, obj_trace


def temporal_prior(n_frames, n_clusters, rho, device):
    temp_prior = torch.abs(torch.arange(n_frames)[:, None] / n_frames - torch.arange(n_clusters)[None, :] / n_clusters).to(device)
    return rho * temp_prior