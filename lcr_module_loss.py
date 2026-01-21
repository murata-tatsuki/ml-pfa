# lcr_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment   # Hungarian matching

# ------------------------------------------------------------
# Utility: permutation-invariant Hungarian loss
# ------------------------------------------------------------
def hungarian_set_loss(pred_fourvec, pred_cls, true_fourvec, true_cls,
                       cost_class=1.0, cost_bbox=2.0):
    """
    pred_fourvec : (B, M, 4)   – weighted sums from LCR (M=K seeds)
    pred_cls     : (B, M, C)
    true_fourvec : (B, N, 4)   – truth particles (varying N<=M)
    true_cls     : (B, N)
    Return      : scalar loss
    """
    batch_loss = 0.0
    for b in range(pred_fourvec.size(0)):
        # Expand tensors for pair-wise cost matrix
        P = pred_fourvec[b]                    # (M,4)
        T = true_fourvec[b]                    # (N,4)
        Pcls = pred_cls[b].softmax(-1)         # (M,C)
        # Tcl  = F.one_hot(true_cls[b], Pcls.size(-1)).float()  # (N,C)
        # print("truth_four_vector:", T)
        # print("uniqu_label:", true_cls[b])
        # print("uniqu_label min/max:", true_cls[b].min().item(), true_cls[b].max().item())

        # print(true_cls[b].shape, pred_cls[b].shape, Pcls.size(-1), Pcls)

        if not ((true_cls[b] >= 0) & (true_cls[b] < Pcls.size(-1))).all():
            print(f"Invalid true_cls index in batch {b}: ", true_cls[b])
            raise ValueError("true_cls contains out-of-bound class indices.")

        Tcl = F.one_hot(true_cls[b].long(), Pcls.size(-1)).float()

        # print(T.shape, P.shape)
        if T.ndim == 1:
            T = T.unsqueeze(0)
        if P.ndim == 1:
            P = P.unsqueeze(0)

        # ① L1 cost on 4-vector (or custom ΔR,E etc.)
        C_bbox = torch.cdist(P, T, p=1)        # (M,N)

        # ② classification cost (cross-entropy)
        C_cls  = -(Pcls @ Tcl.T)               # (M,N) negative log-prob

        C      = cost_bbox*C_bbox + cost_class*C_cls  # (M,N)
        if torch.isnan(C).any() or torch.isinf(C).any():
            print("NaN or inf detected in cost matrix C")
            print("C_bbox:", C_bbox)
            print("C_cls:", C_cls)
            raise ValueError("Invalid values in cost matrix")

        row, col = linear_sum_assignment(C.cpu().detach().numpy())
        batch_loss += C[row, col].sum() / len(row)

    return batch_loss / pred_fourvec.size(0)

def hungarian_set_loss_bbox_only(pred_fourvec, true_fourvec):
    """
    pred_fourvec : (B, M, 4)   – weighted sums from LCR (M=K seeds)
    true_fourvec : (B, N, 4)   – truth particles (varying N<=M)
    Return       : scalar loss
    """
    batch_loss = 0.0

    for b in range(pred_fourvec.size(0)):
        P = pred_fourvec[b]    # (M, 4)
        T = true_fourvec[b]    # (N, 4)

        if T.ndim == 1:
            T = T.unsqueeze(0)
        if P.ndim == 1:
            P = P.unsqueeze(0)

        # L1 distance cost matrix (M, N)
        C_bbox = torch.cdist(P, T, p=1)

        if torch.isnan(C_bbox).any() or torch.isinf(C_bbox).any():
            print("Invalid values in cost matrix C_bbox")
            print("C_bbox:", C_bbox)
            raise ValueError("NaN or Inf detected in cost matrix")

        # ハンガリアン法による最適マッチング
        row, col = linear_sum_assignment(C_bbox.cpu().detach().numpy())
        batch_loss += C_bbox[row, col].sum() / len(row)

    return batch_loss / pred_fourvec.size(0)

def hungarian_set_loss_new(pred_fourvec, true_fourvec, wE=1.0, wMag=1.0, wDir=1.0, eps=1e-8):
    """
    pred_fourvec : (B, M, 4)   – weighted sums from LCR (M=K seeds)
    true_fourvec : (B, N, 4)   – truth particles (varying N<=M)
    wE, wMag, wDir : エネルギー、大きさ、方向の重み
    Return       : scalar loss
    """
    batch_loss = 0.0

    for b in range(pred_fourvec.size(0)):
        P = pred_fourvec[b]    # (M, 4)
        T = true_fourvec[b]    # (N, 4)
        M, N = P.shape[0], T.shape[0]

        # ΔE/E
        dE = torch.log(torch.abs(P[:,None,0] - T[None,:,0]) + 1)  # (M,N)
        # dE = torch.abs(P[:,None,0] - T[None,:,0]) / (T[None,:,0] + eps)  # (M,N)
        # 方向誤差
        p_pred = P[:,None,1:]  # (M,1,3)
        p_true = T[None,:,1:]  # (1,N,3)
        # cos_theta = torch.sum(p_pred * p_true, dim=-1) / (
        #     torch.norm(p_pred, dim=-1) * torch.norm(p_true, dim=-1) + eps
        # )
        # dTheta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))  # (M,N)
        mag_pred = torch.norm(p_pred, dim=-1)
        mag_true = torch.norm(p_true, dim=-1)
        dMag = torch.abs(mag_pred - mag_true) / (mag_true + eps)  # (M,N)

        p_pred_norm = p_pred / (mag_pred.unsqueeze(-1) + eps)
        p_true_norm = p_true / (mag_true.unsqueeze(-1) + eps)
        dDir = torch.sum((p_pred_norm - p_true_norm)**2, dim=-1)  # (M,N)

        C = (wE * dE + wMag * dMag + wDir * dDir)
        row, col = linear_sum_assignment(C.detach().cpu().numpy())

        batch_loss += C[row, col].sum() / len(row)

    return batch_loss / pred_fourvec.size(0)

def hungarian_set_loss_new_sub(pred_fourvec, true_fourvec, wE=1.0, wMag=1.0, wDir=1.0, eps=1e-8):
    """
    pred_fourvec : (B, M, 4)   – weighted sums from LCR (M=K seeds)
    true_fourvec : (B, N, 4)   – truth particles (varying N<=M)
    wE, wMag, wDir : エネルギー、大きさ、方向の重み
    Return       : scalar loss
    """
    batch_loss = 0.0
    loss_E = 0.0
    loss_Mag = 0.0
    loss_Dir = 0.0

    for b in range(pred_fourvec.size(0)):
        P = pred_fourvec[b]    # (M, 4)
        T = true_fourvec[b]    # (N, 4)
        M, N = P.shape[0], T.shape[0]

        # ΔE/E
        dE = torch.log(torch.abs(P[:,None,0] - T[None,:,0]) + 1)  # (M,N)
        # dE = torch.abs(P[:,None,0] - T[None,:,0]) / (T[None,:,0] + eps)  # (M,N)
        # 方向誤差
        p_pred = P[:,None,1:]  # (M,1,3)
        p_true = T[None,:,1:]  # (1,N,3)
        # cos_theta = torch.sum(p_pred * p_true, dim=-1) / (
        #     torch.norm(p_pred, dim=-1) * torch.norm(p_true, dim=-1) + eps
        # )
        # dTheta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))  # (M,N)
        mag_pred = torch.norm(p_pred, dim=-1)
        mag_true = torch.norm(p_true, dim=-1)
        dMag = torch.abs(mag_pred - mag_true) / (mag_true + eps)  # (M,N)

        p_pred_norm = p_pred / (mag_pred.unsqueeze(-1) + eps)
        p_true_norm = p_true / (mag_true.unsqueeze(-1) + eps)
        dDir = torch.sum((p_pred_norm - p_true_norm)**2, dim=-1)  # (M,N)

        C = (wE * dE + wMag * dMag + wDir * dDir)
        row, col = linear_sum_assignment(C.detach().cpu().numpy())

        loss_E += dE[row, col].sum() / len(row)
        loss_Mag += dMag[row, col].sum() / len(row)
        loss_Dir += dDir[row, col].sum() / len(row)
        batch_loss += C[row, col].sum() / len(row)

    components = dict(
        loss_E = loss_E / pred_fourvec.size(0), 
        loss_Mag = loss_Mag / pred_fourvec.size(0),
        loss_Dir = loss_Dir / pred_fourvec.size(0)
        )

    return batch_loss / pred_fourvec.size(0), components

def hungarian_set_loss_new_sub_mask(pred_fourvec, true_fourvec, pred_mask=None, true_mask=None, wE=1.0, wMag=1.0, wDir=1.0, eps=1e-8):
    """
    pred_fourvec : (B, M, 4)   – predicted 4-vectors
    true_fourvec : (B, N, 4)   – truth 4-vectors
    pred_mask    : (B, M) bool – True=無効 (padding) 
    true_mask    : (B, N) bool – True=無効 (padding)
    """
    batch_loss = 0.0
    loss_E = 0.0
    loss_Mag = 0.0
    loss_Dir = 0.0

    B = pred_fourvec.size(0)

    for b in range(B):
        # --- 修正: maskを適用 ---
        if pred_mask is not None:
            if pred_mask.shape[1] != pred_fourvec.shape[1]:
                print(f"[DEBUG] pred_fourvec[b].shape = {pred_fourvec[b].shape}")
                print(f"[DEBUG] pred_mask[b].shape = {pred_mask[b].shape}")
                print(f"[DEBUG] pred_mask[b] sum = {pred_mask[b].sum()}")
            P = pred_fourvec[b][~pred_mask[b]]  # (M_valid, 4)
        else:
            P = pred_fourvec[b]

        if true_mask is not None:
            T = true_fourvec[b][~true_mask[b]]  # (N_valid, 4)
        else:
            T = true_fourvec[b]
        # -------------------------

        if P.size(0) == 0 or T.size(0) == 0:
            continue  # 有効データがなければスキップ

        # ΔE/E (log)
        dE = torch.log(torch.abs(P[:, None, 0] - T[None, :, 0]) + 1)  # (M,N)

        # 運動量大きさ
        p_pred = P[:, None, 1:]  # (M,1,3)
        p_true = T[None, :, 1:]  # (1,N,3)
        mag_pred = torch.norm(p_pred, dim=-1)
        mag_true = torch.norm(p_true, dim=-1)
        dMag = torch.abs(mag_pred - mag_true) / (mag_true + eps)  # (M,N)

        # 方向
        p_pred_norm = p_pred / (mag_pred.unsqueeze(-1) + eps)
        p_true_norm = p_true / (mag_true.unsqueeze(-1) + eps)
        dDir = torch.sum((p_pred_norm - p_true_norm) ** 2, dim=-1)  # (M,N)

        # コスト行列
        C = (wE * dE + wMag * dMag + wDir * dDir)
        row, col = linear_sum_assignment(C.detach().cpu().numpy())

        loss_E += dE[row, col].sum() / len(row)
        loss_Mag += dMag[row, col].sum() / len(row)
        loss_Dir += dDir[row, col].sum() / len(row)
        batch_loss += C[row, col].sum() / len(row)

    components = dict(
        loss_E = loss_E / B,
        loss_Mag = loss_Mag / B,
        loss_Dir = loss_Dir / B
    )

    return batch_loss / B, components

def soft_matching_loss(pred_four, true_four, beta_pred=None, pred_mask=None, true_mask=None, alpha_E=1.0, alpha_dir=0.2, beta_temp=1.0):
    """
    pred_four: (B, N_pred, 4)
    true_four: (B, N_true, 4)
    beta_pred: (B, N_pred) optional, β-weight for seed confidence
    pred_mask: (B, N_pred), bool tensor, True=有効, False=padding
    true_mask: (B, N_true), bool tensor, True=有効, False=padding
    """

    E_pred, p_pred = pred_four[..., 0], pred_four[..., 1:]
    E_true, p_true = true_four[..., 0], true_four[..., 1:]

    # normalize direction
    p_pred_norm = F.normalize(p_pred, dim=-1)
    p_true_norm = F.normalize(p_true, dim=-1)

    # pairwise cosine similarity
    cos_sim = torch.einsum('bik,bjk->bij', p_pred_norm, p_true_norm).clamp(-1, 1)
    ang_diff = 1 - cos_sim  # small if aligned

    # relative energy diff
    rel_E = torch.abs(E_pred.unsqueeze(2) - E_true.unsqueeze(1)) / (E_true.unsqueeze(1) + 1e-6)

    # pairwise distance (smaller is better)
    dist = alpha_E * rel_E + alpha_dir * ang_diff

    # convert to similarity (larger is better)
    sim = -dist / beta_temp  # temperature controls softness

    # apply masks before softmax
    if true_mask is not None:
        # set invalid true entries to large negative
        mask_j = (~true_mask).unsqueeze(1).expand_as(sim)
        sim = sim.masked_fill(mask_j, -1e9)


    # soft assignment weights (B, N_pred, N_true)
    w = F.softmax(sim, dim=-1)

    # β-weighted version
    if beta_pred is not None:
        beta_w = beta_pred.unsqueeze(-1) / (beta_pred.sum(dim=1, keepdim=True) + 1e-6)
        w = w * beta_w  # apply β weight
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)  # renormalize
    
    # apply pred mask: invalid rows (padding) should not contribute to loss
    if pred_mask is not None:
        mask_i = pred_mask.unsqueeze(-1).float()
        w = w * mask_i
        dist = dist * mask_i

    if true_mask is not None:
        mask_j = true_mask.unsqueeze(1).float()
        w = w * mask_j
        dist = dist * mask_j

    # compute loss
    valid_sum = (w > 0).float().sum(dim=[1, 2]) + 1e-6
    loss_pair = (w * dist).sum(dim=[1, 2]) / valid_sum
    loss = loss_pair.mean()
    # loss_pair = dist  # (B, N_pred, N_true)
    # loss = (w * loss_pair).sum(dim=[1, 2]).mean()

    components = dict(
        loss = loss,
        loss_E = 0, 
        loss_Mag = 0, 
        loss_Dir = 0
    )

    return loss, components

def soft_matching_cross_attention_loss(pred_four, true_four, cross_attention, beta_pred=None, pred_mask=None, true_mask=None, alpha_E=1.0, alpha_dir=0.2, beta_temp=1.0):
    """
    pred_four: (B, N_pred, 4)
    true_four: (B, N_true, 4)
    beta_pred: (B, N_pred) optional, β-weight for seed confidence
    pred_mask: (B, N_pred), bool tensor, True=有効, False=padding
    true_mask: (B, N_true), bool tensor, True=有効, False=padding
    """

    E_pred, p_pred = pred_four[..., 0], pred_four[..., 1:]
    E_true, p_true = true_four[..., 0], true_four[..., 1:]

    # normalize direction
    p_pred_norm = F.normalize(p_pred, dim=-1)
    p_true_norm = F.normalize(p_true, dim=-1)

    # pairwise cosine similarity
    cos_sim = torch.einsum('bik,bjk->bij', p_pred_norm, p_true_norm).clamp(-1, 1)
    ang_diff = 1 - cos_sim  # small if aligned

    # relative energy diff
    rel_E = torch.abs(E_pred.unsqueeze(2) - E_true.unsqueeze(1)) / (E_true.unsqueeze(1) + 1e-6)

    # pairwise distance (smaller is better)
    dist = alpha_E * rel_E + alpha_dir * ang_diff

    # convert to similarity (larger is better)
    sim = -dist / beta_temp  # temperature controls softness

    # apply masks before softmax
    if true_mask is not None:
        # set invalid true entries to large negative
        mask_j = (~true_mask).unsqueeze(1).expand_as(sim)
        sim = sim.masked_fill(mask_j, -1e9)


    # soft assignment weights (B, N_pred, N_true)
    w = F.softmax(sim, dim=-1)

    # β-weighted version
    if beta_pred is not None:
        beta_w = beta_pred.unsqueeze(-1) / (beta_pred.sum(dim=1, keepdim=True) + 1e-6)
        w = w * beta_w  # apply β weight
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-6)  # renormalize
    
    # apply pred mask: invalid rows (padding) should not contribute to loss
    if pred_mask is not None:
        mask_i = pred_mask.unsqueeze(-1).float()
        w = w * mask_i
        dist = dist * mask_i

    if true_mask is not None:
        mask_j = true_mask.unsqueeze(1).float()
        w = w * mask_j
        dist = dist * mask_j

    # compute loss
    valid_sum = (w > 0).float().sum(dim=[1, 2]) + 1e-6
    loss_pair = (w * dist).sum(dim=[1, 2]) / valid_sum
    loss = loss_pair.mean()
    # loss_pair = dist  # (B, N_pred, N_true)
    # loss = (w * loss_pair).sum(dim=[1, 2]).mean()

    components = dict(
        loss = loss,
        loss_E = 0, 
        loss_Mag = 0, 
        loss_Dir = 0
    )

    return loss, components

"""
def attention_cluster_loss(attn, truth_cluster, is_track_query, eps=1e-8):
    ""
    attn: (B, Nq, Nk)  softmax済み attention
    truth_cluster: (B, N_hits) int cluster ID (tracks + calo 並んだ同一次元)
    is_track_query: (B, Nq) bool mask, True if query is track
    ""

    B, Nq, Nk = attn.shape
    
    # クラスタIDを (B, Nq, Nk) に broadcast
    # 例）cluster[q] == cluster[k] → positive mask
    q_cluster = truth_cluster.unsqueeze(2).expand(B, Nq, Nk)  # (B, Nq, Nk)
    k_cluster = truth_cluster.unsqueeze(1).expand(B, Nq, Nk)  # (B, Nq, Nk)
    
    pos_mask = (q_cluster == k_cluster)        # same cluster
    neg_mask = ~pos_mask                       # different cluster
    
    # ---- (A) Charged: Cross Entropy風 Loss ----
    # positive cluster への総attentionを最大化
    attn_pos = attn * pos_mask
    sum_pos = attn_pos.sum(dim=-1)  # (B, Nq)
    
    # track queries のみ対象
    track_pos = sum_pos[is_track_query]
    loss_charged = -torch.log(track_pos + eps).mean()

    # ---- (B) Neutral: KL divergence ----
    # truth adjacency matrix を soft label として作成
    with torch.no_grad():
        truth_adj = pos_mask.float()
        truth_adj = truth_adj / (truth_adj.sum(dim=-1, keepdim=True) + eps)

    # neutral query mask
    neutral_mask = ~is_track_query
    attn_neutral = attn[neutral_mask]          # (Nn, Nk)
    truth_neutral = truth_adj[neutral_mask]    # (Nn, Nk)

    # KL divergence: log(attn) vs truth distribution
    # reduction separately to avoid weight imbalance
    if attn_neutral.numel() > 0:
        loss_neutral = F.kl_div(
            attn_neutral.log(), truth_neutral, reduction="batchmean"
        )
    else:
        loss_neutral = torch.tensor(0.0, device=attn.device)

    # ---- 合算 ----
    loss = loss_charged + loss_neutral

    return {
        "loss": loss,
        "loss_charged": loss_charged,
        "loss_neutral": loss_neutral
    }
"""

def attention_loss_old(
    attn,                  # (B, Nq, Nk) softmax済み attention
    truth_cluster,         # (B, N_hits) int cluster ID
    beta,                  # (B, N_hits) in [0,1]
    is_track_query,        # (B, Nq) bool mask, True if query is track
    query_mask=None,       # (B, Nq) bool mask, True if valid
    key_mask=None,         # (B, Nk) bool mask, True if valid
    query_indices_in_key=None,
    particle_prob=None,
    eps=1e-8
    ):
    """
    β-weighted cross attention loss with padding mask support.
    """
    B, Nq, Nk = attn.shape
    assert(Nk == truth_cluster.shape[1])

    # --- cluster broadcast ---
    truth_cluster_q = truth_cluster.gather(1, query_indices_in_key)  # (B, Nq)
    q_cluster = truth_cluster_q.unsqueeze(2).expand(B, Nq, Nk)
    k_cluster = truth_cluster.unsqueeze(1).expand(B, Nq, Nk)
    pos_mask = (q_cluster == k_cluster)  # same cluster

    valid_query_mask = query_mask if query_mask is not None else torch.ones(B, Nq, dtype=torch.bool, device=attn.device)
    # -------------------------------
    # Detect duplicate queries per (batch, cluster)
    # - only consider queries with valid_query_mask == True
    # - for each cluster keep the first (lowest index) query, mark others as dead
    # dead_query_mask: (B, Nq) bool, True means "this is a duplicate (2nd+) query"
    # -------------------------------
    dead_query_mask = torch.zeros(B, Nq, dtype=torch.bool, device=attn.device)
    for b in range(B):
        qc = truth_cluster_q[b]         # (Nq,)
        vmask = valid_query_mask[b]     # (Nq,)
        # iterate in index order; record first occurrence of each cluster id
        seen = {}
        # go through only valid queries, in increasing index order
        valid_idx = torch.nonzero(vmask, as_tuple=False).flatten()
        for qi in valid_idx.tolist():
            cid = int(qc[qi].item())
            # If cluster id is padding-like (e.g. -1), skip marking (treat as no-cluster)
            # (Assumes padding cluster ids are negative; adjust if different)
            if cid < 0:
                continue
            if cid in seen:
                # this is second+ occurrence -> mark dead
                dead_query_mask[b, qi] = True
            else:
                seen[cid] = qi

    # -------------------------------
    # Remove pos_mask (truth matching) for dead queries so they do not get correct-target credit.
    # We still keep their attention values (so we can penalize them), but they must not match truth.
    # pos_mask: (B, Nq, Nk)
    # -------------------------------
    if dead_query_mask.any():
        pos_mask = pos_mask.clone()
        pos_mask[dead_query_mask.unsqueeze(2).expand_as(pos_mask)] = False


    # -------------------------------
    #  β-weighted truth adjacency
    # -------------------------------
    beta_k = beta.unsqueeze(1)  # (B, 1, Nk)
    with torch.no_grad():
        truth_adj = pos_mask.float() * beta_k

        # key mask がある場合 padding を 0 に
        if key_mask is not None:
            truth_adj = truth_adj * key_mask.unsqueeze(1)

        denom = truth_adj.sum(dim=-1, keepdim=True)
        # print("denom.min(), denom.max():", denom.min(), denom.max())
        zero_mask = denom < eps
        # print("zero_mask.sum():", zero_mask.sum())
        truth_adj = truth_adj / (denom + eps)

        # fallback for zero-sum
        if zero_mask.any():
            fallback = pos_mask.float()
            if key_mask is not None:
                fallback = fallback * key_mask.unsqueeze(1)
            fallback = fallback / (fallback.sum(dim=-1, keepdim=True) + eps)
            truth_adj[zero_mask.expand_as(truth_adj)] = fallback[zero_mask.expand_as(truth_adj)]

    # -------------------------------
    # Charged: Cross Entropy style
    # -------------------------------
    attn_pos = attn * pos_mask

    # key mask
    if key_mask is not None:
        attn_pos = attn_pos * key_mask.unsqueeze(1)

    sum_pos = attn_pos.sum(dim=-1)  # (B, Nq)

    track_pos = sum_pos[is_track_query & valid_query_mask]
    # print(track_pos)
    # loss_charged = -torch.log(track_pos + eps).mean()
    # loss_charged = -torch.log(track_pos.clamp(min=eps)).mean()
    track_pos_safe = torch.clamp(track_pos, min=eps)
    loss_charged = -torch.log(track_pos_safe).mean()


    # -------------------------------
    # Neutral: KL divergence
    # -------------------------------
    neutral_mask = ~is_track_query & valid_query_mask
    # print((~is_track_query).shape, (~is_track_query).nonzero().shape)
    # print(neutral_mask.shape, neutral_mask.nonzero().shape)

    attn_neutral = attn[neutral_mask]  # (Nn, Nk)
    truth_neutral = truth_adj[neutral_mask]  # (Nn, Nk)

    if key_mask is not None:
        key_mask_expanded = key_mask.unsqueeze(1).expand(B, Nq, Nk)
        key_mask_neutral = key_mask_expanded[neutral_mask]  # (Nn, Nk)

        attn_neutral = attn_neutral * key_mask_neutral
        truth_neutral = truth_neutral * key_mask_neutral

    if attn_neutral.numel() > 0:
        attn_neutral_safe = torch.clamp(attn_neutral, min=eps)
        loss_neutral = F.kl_div(attn_neutral_safe.log(), truth_neutral, reduction="batchmean")
        # loss_neutral = F.kl_div(attn_neutral.log(), truth_neutral, reduction="batchmean")
    else:
        loss_neutral = torch.tensor(0.0, device=attn.device)
    
    # -------------------------------
    # L1 penalty for dead (duplicate) queries
    # -------------------------------
    lambda_attn = 3.0
    if dead_query_mask.any():
        # dead_query_mask: (B, Nq) -> expand to (B, Nq, 1) to match attn shape
        dead_mask_exp = dead_query_mask.unsqueeze(-1).float()  # (B, Nq, 1)
        # L1 penalty: mean absolute attention over dead queries only
        loss_attn_dead = lambda_attn * torch.mean(torch.abs(attn * dead_mask_exp))
    else:
        loss_attn_dead = torch.tensor(0.0, device=attn.device)

    
    # -------------------------------
    # L1 penalty for invalid queries
    # -------------------------------
    if query_mask is not None:
        invalid_query = (~query_mask).float().unsqueeze(-1)   # (B, Nq, 1)
        loss_attn_pad = lambda_attn * torch.mean(torch.abs(attn * invalid_query))
    else:
        loss_attn_pad = torch.tensor(0.0, device=attn.device)
    



    # particle_prob : (B, Nq) with values in [0,1]

    # teacher signal
    # target = 1 for alive queries, 0 for padding & dead queries
    alive_mask = valid_query_mask & (~dead_query_mask)

    target_particle_prob = alive_mask.float()   # (B, Nq)
    
    # BCE loss
    loss_particle_prob = F.binary_cross_entropy(
        particle_prob.clamp(min=1e-6, max=1.0-1e-6),
        target_particle_prob
    )


    # -------------------------------
    # 合算
    # -------------------------------
    loss_charged = loss_charged * 100
    loss_neutral = loss_neutral * 100
    loss_attn_pad = loss_attn_pad * 100
    loss_particle_prob = loss_particle_prob * 100
    
    return loss_charged, loss_neutral, loss_attn_pad, loss_attn_dead, loss_particle_prob

def attention_loss(
    attn,                  # (B, Nq, Nk) softmax済み attention
    raw_score,             # (B, H, Nq, Nk) softmaxしていない raw な attention
    truth_cluster,         # (B, N_hits) int cluster ID
    beta,                  # (B, N_hits) in [0,1]
    is_track_query,        # (B, Nq) bool mask, True if query is track
    query_mask=None,       # (B, Nq) bool mask, True if valid
    key_mask=None,         # (B, Nk) bool mask, True if valid
    query_indices_in_key=None,
    particle_prob=None,
    eps=1e-8
    ):
    """
    β-weighted cross attention loss with padding mask support.
    """
    B, Nq, Nk = attn.shape
    assert(Nk == truth_cluster.shape[1])

    # --- cluster broadcast ---
    truth_cluster_q = truth_cluster.gather(1, query_indices_in_key)  # (B, Nq)
    q_cluster = truth_cluster_q.unsqueeze(2).expand(B, Nq, Nk)
    k_cluster = truth_cluster.unsqueeze(1).expand(B, Nq, Nk)
    pos_mask = (q_cluster == k_cluster)  # same cluster
    torch.set_printoptions(edgeitems=10000)
    print(q_cluster[0])
    print(k_cluster[0])
    print(pos_mask.shape)
    print(pos_mask[0])

    valid_query_mask = query_mask if query_mask is not None else torch.ones(B, Nq, dtype=torch.bool, device=attn.device)
    # -------------------------------
    # Detect duplicate queries per (batch, cluster)
    # - only consider queries with valid_query_mask == True
    # - for each cluster keep the first (lowest index) query, mark others as dead
    # dead_query_mask: (B, Nq) bool, True means "this is a duplicate (2nd+) query"
    # -------------------------------
    dead_query_mask = torch.zeros(B, Nq, dtype=torch.bool, device=attn.device)
    for b in range(B):
        qc = truth_cluster_q[b]         # (Nq,)
        vmask = valid_query_mask[b]     # (Nq,)
        # iterate in index order; record first occurrence of each cluster id
        seen = {}
        # go through only valid queries, in increasing index order
        valid_idx = torch.nonzero(vmask, as_tuple=False).flatten()
        for qi in valid_idx.tolist():
            cid = int(qc[qi].item())
            # If cluster id is padding-like (e.g. -1), skip marking (treat as no-cluster)
            # (Assumes padding cluster ids are negative; adjust if different)
            if cid < 0:
                continue
            if cid in seen:
                # this is second+ occurrence -> mark dead
                dead_query_mask[b, qi] = True
            else:
                seen[cid] = qi

    # -------------------------------
    # Remove pos_mask (truth matching) for dead queries so they do not get correct-target credit.
    # We still keep their attention values (so we can penalize them), but they must not match truth.
    # pos_mask: (B, Nq, Nk)
    # -------------------------------
    # if dead_query_mask.any():
    #     pos_mask = pos_mask.clone()
    #     pos_mask[dead_query_mask.unsqueeze(2).expand_as(pos_mask)] = False

    # -------------------------------
    # cross attention matching: Cross Entropy style
    # -------------------------------
    loss_charged = hitwise_clustering_ce_loss(raw_score, pos_mask, query_mask, key_mask)
    # loss_charged = hitwise_clustering_ce_loss_from_probs(attn, pos_mask, query_mask, key_mask)
    loss_neutral = torch.tensor(0.0, device=attn.device)
    
    # -------------------------------
    # L1 penalty for dead (duplicate) queries
    # -------------------------------
    lambda_attn = 3.0
    if dead_query_mask.any():
        # dead_query_mask: (B, Nq) -> expand to (B, Nq, 1) to match attn shape
        dead_mask_exp = dead_query_mask.unsqueeze(-1).float()  # (B, Nq, 1)
        # L1 penalty: mean absolute attention over dead queries only
        loss_attn_dead = lambda_attn * torch.mean(torch.abs(attn * dead_mask_exp))
    else:
        loss_attn_dead = torch.tensor(0.0, device=attn.device)

    
    # -------------------------------
    # L1 penalty for invalid queries
    # -------------------------------
    if query_mask is not None:
        invalid_query = (~query_mask).float().unsqueeze(-1)   # (B, Nq, 1)
        loss_attn_pad = lambda_attn * torch.mean(torch.abs(attn * invalid_query))
    else:
        loss_attn_pad = torch.tensor(0.0, device=attn.device)
    



    # particle_prob : (B, Nq) with values in [0,1]

    # teacher signal
    # target = 1 for alive queries, 0 for padding & dead queries
    alive_mask = valid_query_mask & (~dead_query_mask)

    target_particle_prob = alive_mask.float()   # (B, Nq)
    
    # BCE loss
    loss_particle_prob = F.binary_cross_entropy(
        particle_prob.clamp(min=1e-6, max=1.0-1e-6),
        target_particle_prob
    )


    # -------------------------------
    # 合算
    # -------------------------------
    # loss_charged = loss_charged * 100
    loss_charged = loss_charged
    loss_neutral = loss_neutral * 100
    loss_attn_pad = loss_attn_pad * 100
    loss_particle_prob = loss_particle_prob * 100
    
    return loss_charged, loss_neutral, loss_attn_pad, loss_attn_dead, loss_particle_prob


def clustering_loss(
    attn,                  # (B, Nq, Nk) softmax済み attention
    raw_score,             # (B, Nq, Nk) softmaxしていない rawな attention
    truth_cluster,         # (B, N_hits) int cluster ID
    beta,                  # (B, N_hits) in [0,1]
    is_track_query,        # (B, Nq) bool mask, True if query is track
    query_mask=None,       # (B, Nq) bool mask, True if valid
    key_mask=None,         # (B, Nk) bool mask, True if valid
    query_indices_in_key=None,
    particle_prob=None
    ):

    B, Nq, Nk = attn.shape


    loss_charged, loss_neutral, loss_attn_pad, loss_attn_dead, loss_particle_prob = attention_loss(
        attn, raw_score, truth_cluster, beta, is_track_query, query_mask=query_mask, key_mask=key_mask, query_indices_in_key=query_indices_in_key ,particle_prob=particle_prob)
    

    total_loss = loss_charged + loss_neutral + loss_attn_pad + loss_particle_prob



    components = dict(
        loss = total_loss / B,
        loss_E = 0,
        loss_Mag = 0,
        loss_Dir = 0,
        loss_pcl_prob = loss_particle_prob / B,
        loss_pid = 0,
        loss_charged = loss_charged / B,
        loss_neutral = loss_neutral / B,
        loss_attn_pad = loss_attn_pad / B,
        loss_attn_dead = loss_attn_dead / B,
    )

    return total_loss / B, components


def hitwise_clustering_ce_loss(
    logits,             # (B, H, Nq, Nk)
    truth_clustering,   # (B, Nq, Nk)
    query_mask,         # (B, Nq)
    key_mask,           # (B, Nk)
    eps=1e-12
    ):
    logits = logits.mean(dim=1)
    B, Nq, Nk = logits.shape

    if query_mask is not None:
        q_mask = torch.logical_not(query_mask.unsqueeze(2).expand(B, Nq, Nk))
        truth_clustering[q_mask] = False
    if key_mask is not None:
        k_mask = torch.logical_not(key_mask.unsqueeze(1).expand(B, Nq, Nk))
        truth_clustering[k_mask] = False

    # --------------------------------------------------
    # hit をバッチ軸に展開
    # --------------------------------------------------
    # (B, Nq, Nk) -> (B, Nk, Nq)
    logits = logits.permute(0, 2, 1)
    truth  = truth_clustering.permute(0, 2, 1)

    # (B, Nk, Nq) -> (B*Nk, Nq)
    logits = logits.reshape(B * Nk, Nq)
    truth  = truth.reshape(B * Nk, Nq)

    print(logits[0])
    print(truth[0])

    # --------------------------------------------------
    # mask 整形
    # --------------------------------------------------
    hit_mask   = key_mask.reshape(B * Nk)           # (B*Nk)
    query_mask = query_mask.unsqueeze(1)             # (B,1,Nq)
    query_mask = query_mask.expand(B, Nk, Nq)
    query_mask = query_mask.reshape(B * Nk, Nq)      # (B*Nk,Nq)

    # --------------------------------------------------
    # 無効 query を softmax 競合から除外
    # --------------------------------------------------
    logits = logits.masked_fill(query_mask == 0, -1e4)

    # --------------------------------------------------
    # log-softmax（query 方向）
    # --------------------------------------------------
    log_probs = F.log_softmax(logits, dim=-1)

    # --------------------------------------------------
    # cross entropy（soft label 対応）
    # --------------------------------------------------
    loss_per_hit = -(truth * log_probs).sum(dim=-1)  # (B*Nk)

    # --------------------------------------------------
    # 無効 hit を除外
    # --------------------------------------------------
    loss_per_hit = loss_per_hit * hit_mask

    # --------------------------------------------------
    # 正規化
    # --------------------------------------------------
    loss = loss_per_hit.sum() / hit_mask.sum().clamp_min(eps)

    return loss

def hitwise_clustering_ce_loss_from_probs(
    probs,              # (B, Nq, Nk), softmax over Nq
    truth_clustering,   # (B, Nq, Nk)
    query_mask,         # (B, Nq)
    key_mask,           # (B, Nk)
    eps=1e-12
    ):
    B, Nq, Nk = probs.shape

    # --------------------------------------------------
    # hit をサンプル軸に展開
    # --------------------------------------------------
    # (B, Nq, Nk) -> (B, Nk, Nq)
    probs = probs.permute(0, 2, 1)
    truth = truth_clustering.permute(0, 2, 1)

    # (B, Nk, Nq) -> (B*Nk, Nq)
    probs = probs.reshape(B * Nk, Nq)
    truth = truth.reshape(B * Nk, Nq)

    # --------------------------------------------------
    # mask 整形
    # --------------------------------------------------
    hit_mask = key_mask.reshape(B * Nk)               # (B*Nk)

    query_mask = query_mask.unsqueeze(1)              # (B,1,Nq)
    query_mask = query_mask.expand(B, Nk, Nq)
    query_mask = query_mask.reshape(B * Nk, Nq)

    # --------------------------------------------------
    # 無効 query を除外 → 再正規化
    # --------------------------------------------------
    probs = probs * query_mask
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

    # --------------------------------------------------
    # cross entropy（soft label 対応）
    # --------------------------------------------------
    log_probs = torch.log(probs.clamp_min(eps))
    loss_per_hit = -(truth * log_probs).sum(dim=-1)   # (B*Nk)

    # --------------------------------------------------
    # 無効 hit を除外
    # --------------------------------------------------
    loss_per_hit = loss_per_hit * hit_mask

    # --------------------------------------------------
    # 正規化
    # --------------------------------------------------
    loss = loss_per_hit.sum() / hit_mask.sum().clamp_min(eps)

    return loss





def lcr_hungarian_loss(pred_fourvec, true_fourvec, pred_pcl_prob, matched_idx, pred_logits, tgt_labels, attn, truth_mcid, wE=1.0, wMag=1.0, wDir=1.0, eps=1e-8):
    """
    pred_fourvec : (B, M, 4)   – weighted sums from LCR (M=K seeds)
    true_fourvec : (B, N, 4)   – truth particles (varying N<=M)
    matched_idx  : list of tuples [(row_idx, col_idx), ...] for each batch, from HungarianMatcher
    wE, wMag, wDir : エネルギー、大きさ、方向の重み
    Return       : scalar loss
    """
    batch_loss = 0.0
    loss_E = 0.0
    loss_Mag = 0.0
    loss_Dir = 0.0
    loss_pcl_prob = 0.0
    loss_pid = 0.0
    loss_attn = 0.0
    B = pred_fourvec.size(0)


    for b in range(B):
        P = pred_fourvec[b]    # (M, 4)
        T = true_fourvec[b]    # (N, 4)
        row, col = matched_idx[b]  # torch.Tensor

        # ΔE/E
        dE = torch.log(torch.abs(P[row, 0] - T[col, 0]) + 1)

        # 方向誤差・大きさ誤差
        p_pred = P[row, 1:]  # (matched_count, 3)
        p_true = T[col, 1:]  # (matched_count, 3)
        mag_pred = torch.norm(p_pred, dim=-1)
        mag_true = torch.norm(p_true, dim=-1)
        dMag = torch.abs(mag_pred - mag_true) / (mag_true + eps)

        p_pred_norm = p_pred / (mag_pred.unsqueeze(-1) + eps)
        p_true_norm = p_true / (mag_true.unsqueeze(-1) + eps)
        dDir = torch.sum((p_pred_norm - p_true_norm)**2, dim=-1)

        C = wE * dE + wMag * dMag + wDir * dDir
        batch_loss += C.sum() / len(row)
        loss_E += dE.sum() / len(row)
        loss_Mag += dMag.sum() / len(row)
        loss_Dir += dDir.sum() / len(row)

        # particle probability loss 
        target_particle = torch.zeros_like(pred_pcl_prob[b])
        target_particle[row] = 1
        loss_pcl_prob += F.binary_cross_entropy(pred_pcl_prob[b], target_particle)

        # pid loss
        pred_logit = pred_logits[b]
        tgt_label = tgt_labels[b]
        criterion = nn.CrossEntropyLoss()
        loss_pid += criterion(pred_logit[row], tgt_label[col])

        # Truth-based hit–seed assignment Cross Entropy
        # hit_to_truth = truth_mcid[b]
        # for i_seed, i_truth in zip(row, col):
        #     tgt_mask = (hit_to_truth == i_truth)  # hit ∈ true cluster
        #     attn_b = attn[b, i_seed]              # [N_hit]
        #     loss_attn += F.binary_cross_entropy(attn_b, tgt_mask.float())
        loss_attn = 0


    components = dict(
        loss = batch_loss / B + loss_pcl_prob / B + loss_pid / B + loss_attn / B,
        loss_E = loss_E / B,
        loss_Mag = loss_Mag / B,
        loss_Dir = loss_Dir / B,
        loss_pcl_prob = loss_pcl_prob / B,
        loss_pid = loss_pid / B,
        # loss_attn = loss_attn / B
    )

    return batch_loss / B + loss_pcl_prob / B + loss_pid / B, components
    # return batch_loss / B + loss_pcl_prob / B + loss_pid / B + loss_attn / B, components
