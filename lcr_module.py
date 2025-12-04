# lcr_module.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment   # Hungarian matching
import numpy as np

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
def truth_based_assignment(attn_w, hit_true_pid, K=5):
    # attn_w: [B, num_heads, N_seed, N_hit]
    # attn_mean = attn_w.mean(dim=1)  # → [B, Kmax, N_hit]
    attn_mean = attn_w

    # print(attn_mean.shape)

    # B, Kmax, N_hit = attn_mean.shape
    B = len(attn_mean)
    Kmax = attn_mean[0].shape[0]
    device = attn_mean[0].device

    seed_true_pid = []
    valid_mask = []

    for b in range(B):
        attn_b = attn_mean[b]  # [Kseed, N_hit]

        # --- Top-K hit assignment ---
        topk_vals, topk_idx = torch.topk(attn_b, K, dim=-1)  # [Kseed,K]
        # hit_pid_b = hit_true_pid[b].gather(0, topk_idx)      # [Kseed,K]
        hit_pid_b = hit_true_pid[b][topk_idx]                  # [Kseed,K]

        # 計算用に (-1 noise) 除去
        mask_valid = (hit_pid_b >= 0)

        seed_pid_b = torch.full((Kmax,), -1, dtype=torch.long, device=device)
        mask_out_b = torch.zeros_like(seed_pid_b, dtype=torch.bool)

        for s in range(Kmax):
            pid_list = hit_pid_b[s][mask_valid[s]]
            if pid_list.numel() > 0:
                seed_pid_b[s] = torch.mode(pid_list).values
                mask_out_b[s] = True

        seed_true_pid.append(seed_pid_b)
        valid_mask.append(mask_out_b)

    seed_true_pid = torch.stack(seed_true_pid)
    valid_mask = torch.stack(valid_mask)

    return seed_true_pid, valid_mask
"""
def truth_based_assignment(attn_w, hit_true_pid, K=5):
    B = len(attn_w)
    Kmax = attn_w[0].shape[0]
    device = attn_w[0].device

    indices = []
    row_list = []
    col_list = []

    for b in range(B):
        attn_b = attn_w[b]  # [Kseed, N_hit]
        true_pid_b = hit_true_pid[b]  # [N_hit]
        print(torch.unique(true_pid_b))
        
        # Top-K indices for each seed
        topk_vals, topk_idx = torch.topk(attn_b, K, dim=-1)   # [Kseed, K]
        hit_pid_b = true_pid_b[topk_idx]                      # [Kseed, K]

        # seedごとに、最頻PIDを持つhitを選択
        """
        for s in range(Kmax):
            pid_list = hit_pid_b[s]
            # valid_mask = (pid_list >= 0)
            # pid_list = pid_list[valid_mask]

            if pid_list.numel() == 0:
                continue

            pid_mode = torch.mode(pid_list).values.item()

            # そのPIDを持つ topk の位置を抽出
            pid_hit_mask = (hit_pid_b[s] == pid_mode)  # [K]
            if pid_hit_mask.any():
                # row: seed index, col: hit index
                selected_hit_idx = topk_idx[s][pid_hit_mask]
                row = torch.full_like(selected_hit_idx, s)

                print(row, selected_hit_idx)
                
                row_list.append(row)
                col_list.append(selected_hit_idx)
        """
        # --- 対処法2: attention confidence で seed を並び替え ---
        # 各seedの最大attention値 → confidenceとして評価
        seed_conf = attn_b.max(dim=-1).values  # [Kseed]
        sorted_seeds = torch.argsort(seed_conf, descending=True)  # confidence順に並べ替え

        assigned_hits = set()

        for s_idx in sorted_seeds:  # 変更: sorted_seeds を使用
            pid_list = hit_pid_b[s_idx]
            pid_mode = torch.mode(pid_list).values.item()

            # mode PID に一致するhit候補を抽出
            candidates = torch.where(pid_list == pid_mode)[0]

            assigned = False
            for c in candidates:
                hit_idx = topk_idx[s_idx, c].item()
                if hit_idx not in assigned_hits:
                    assigned_hits.add(hit_idx)

                    row_list.append(torch.tensor([s_idx], device=device))
                    col_list.append(torch.tensor([hit_idx], device=device))
                    assigned = True
                    print(s_idx, hit_idx)
                    break

            # 候補が全滅 or 割り当てられない場合はスキップ
            if not assigned:
                continue
            
        print(row_list, col_list)

        indices.append((torch.as_tensor(row_list, dtype=torch.long),
                        torch.as_tensor(col_list, dtype=torch.long)))

    # if len(row_list) == 0:
    #     return torch.tensor([]), torch.tensor([])

    # row = torch.cat(row_list, dim=0).to(device)
    # col = torch.cat(col_list, dim=0).to(device)

    # return row, col
    print(indices)
    return indices

def truth_based_assignment_safe(attn_w, hit_true_pid, K=5, debug=False):
    """
    attn_w: list of [Kseed, N_attn_padded] tensors (per batch)
    hit_true_pid: list of [N_hit] tensors (per batch)  (N_hit <= N_attn_padded maybe)
    Returns: indices list [(row_tensor, col_tensor), ...] per batch
    """
    B = len(attn_w)
    device = attn_w[0].device
    indices = []

    for b in range(B):
        attn_b = attn_w[b]          # [Kseed, N_attn_padded]
        Kseed, N_attn_padded = attn_b.shape
        hit_pid = hit_true_pid[b]   # [N_hit]
        N_hit = hit_pid.shape[0]

        # --- pad hit_pid to match N_attn_padded length ---
        if N_hit < N_attn_padded:
            pad = torch.full((N_attn_padded - N_hit,), -1, device=device, dtype=hit_pid.dtype)
            hit_pid_padded = torch.cat([hit_pid.to(device), pad], dim=0)
        elif N_hit > N_attn_padded:
            # this is suspicious but trim safely (log it)
            hit_pid_padded = hit_pid.to(device)[:N_attn_padded]
            if debug:
                print(f"[WARN] batch {b}: hit_true_pid longer ({N_hit}) than N_attn_padded ({N_attn_padded}), trimming.")
        else:
            hit_pid_padded = hit_pid.to(device)

        # top-K over padded attention
        topk_vals, topk_idx = torch.topk(attn_b, K, dim=-1)   # [Kseed, K]

        # safety check: ensure all indices in valid range
        if topk_idx.numel() > 0:
            max_idx = int(topk_idx.max().item())
            min_idx = int(topk_idx.min().item())
            if max_idx >= N_attn_padded or min_idx < 0:
                # invalid indices exist: log and skip problematic entries
                if debug:
                    print(f"[ERROR] batch {b}: topk_idx out of bounds (min {min_idx}, max {max_idx}, padded_len {N_attn_padded})")
                # clamp rather than crash (optional) or mask out invalid
                topk_idx_clamped = topk_idx.clamp(0, N_attn_padded - 1)
            else:
                topk_idx_clamped = topk_idx

        # now safe to index
        hit_pid_b = hit_pid_padded[topk_idx_clamped]  # [Kseed, K]

        row_list = []
        col_list = []
        assigned_hits = set()

        # prefer high-confidence seeds: sort seeds by max attention
        seed_conf = attn_b.max(dim=-1).values
        sorted_seeds = torch.argsort(seed_conf, descending=True)

        for s_idx in sorted_seeds.tolist():
            pid_list = hit_pid_b[s_idx]  # length K
            # ignore negative PID candidates
            valid_mask = pid_list >= 0
            if not valid_mask.any():
                continue
            # mode of candidate pids among valid ones
            valid_pids = pid_list[valid_mask]
            pid_mode = torch.mode(valid_pids).values.item()

            # candidate positions in topk list that match mode
            cand_pos = torch.nonzero((hit_pid_b[s_idx] == pid_mode), as_tuple=False).squeeze(-1)
            if cand_pos.numel() == 0:
                continue

            # pick first candidate whose global index not assigned yet (greedy)
            assigned = False
            for pos in cand_pos.tolist():
                hidx = int(topk_idx_clamped[s_idx, pos].item())  # index into padded hits
                # if hidx corresponds to padded region (> N_hit-1) then skip
                if hidx >= N_hit:
                    # this is a pad index -> skip
                    if debug:
                        print(f"[DEBUG] batch {b} seed {s_idx} candidate pos {pos} -> pad index {hidx} skip")
                    continue
                if hidx in assigned_hits:
                    continue
                assigned_hits.add(hidx)
                row_list.append(s_idx)
                col_list.append(hidx)
                assigned = True
                break
            # end for cand_pos

        # end for seeds

        if len(row_list) == 0:
            indices.append((torch.empty(0, dtype=torch.long, device=device),
                            torch.empty(0, dtype=torch.long, device=device)))
        else:
            indices.append((torch.tensor(row_list, dtype=torch.long, device=device),
                            torch.tensor(col_list, dtype=torch.long, device=device)))

    return indices

class AttentionMatcher:
    """
    Implements the matching using attention for set prediction (DETR style).
    """
    def __init__(self, cls_cost=1.0, bbox_cost=1.0, padding_idx=5):
        self.cls_cost = cls_cost
        self.bbox_cost = bbox_cost
        self.padding_idx = padding_idx

    @torch.no_grad()
    def __call__(self, truth_fourvec_mcids, seed_mcids, seed_trackness):
        """
        Args:
            truth_fourvec_mcids:    list of [num_keyes] LongTensor for each batch
            seed_mcids:             list of [num_queries] LongTensor for each batch
            seed_trackness:         list of [num_queries] LongTensor for each batch
        Returns:
            List of (index_pred, index_tgt) for each batch
        """
        # batch_size, num_queries, num_classes = pred_logits.shape
        batch_size = len(truth_fourvec_mcids)
        num_queries = len(seed_mcids)
        indices = []

        # seedのmcidと同一のidを持つtruth 4 vecのindexをmatchさせる
        # 縦横は要確認だけど
        # 
        for b in range(batch_size):
            out_prob = pred_logits[b]              # [num_queries, num_classes]
            out_bbox = pred_boxes[b]               # [num_queries, box_dim]
            tgt_lbl = tgt_labels[b]
            tgt_box = tgt_boxes[b]
            num_tgt = tgt_lbl.shape[0]
            # Indicator: 1 si vrai objet, 0 si padding
            if hasattr(self, 'padding_idx'):
                padding_idx = self.padding_idx
            else:
                padding_idx = num_classes
            if self.padding_idx is None:
                # Pas de padding, tous les objets sont vrais
                indicator = torch.ones(num_tgt, device=tgt_lbl.device, dtype=torch.float)
            else:
                # Avec padding
                indicator = (tgt_lbl != self.padding_idx).float()  # [num_targets]
            # Cost class: -p_sigma(i)(ci) si vrai objet, 0 sinon
            cost_class = -out_prob[:, tgt_lbl] * indicator  # [num_queries, num_targets]
            # Cost bbox: L1 distance si vrai objet, 0 sinon
            cost_bbox = torch.cdist(out_bbox, tgt_box, p=1) * indicator
            C = self.cls_cost * cost_class + self.bbox_cost * cost_bbox
            C = C.cpu().detach().numpy()
            if not np.all(np.isfinite(C)):
                print("Invalid entries in cost matrix C!")
                print("C:", C)
                print("NaNs:", np.isnan(C).sum(), "Infs:", np.isinf(C).sum())
                raise ValueError("Cost matrix contains NaN or Inf values.")

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.long),
                            torch.as_tensor(col_ind, dtype=torch.long)))
        return indices

class HungarianMatcher:
    """
    Implements the Hungarian matching for set prediction (DETR style).
    """
    def __init__(self, cls_cost=1.0, bbox_cost=1.0, padding_idx=5):
        self.cls_cost = cls_cost
        self.bbox_cost = bbox_cost
        self.padding_idx = padding_idx

    @torch.no_grad()
    def __call__(self, pred_logits, pred_boxes, tgt_labels, tgt_boxes, seed_padding_mask):
        """
        Args:
            pred_logits: [batch_size, num_queries, num_classes]
            pred_boxes:  [batch_size, num_queries, box_dim]
            tgt_labels:  list of [num_targets_i] LongTensor for each batch
            tgt_boxes:   list of [num_targets_i, box_dim] FloatTensor for each batch
            seed_padding_mask: [batch_size, num_queries] boolean 1 fot true , 0 for padding
        Returns:
            List of (index_pred, index_tgt) for each batch
        """
        batch_size, num_queries, num_classes = pred_logits.shape
        indices = []
        for b in range(batch_size):
            padding = seed_padding_mask[b]
            out_prob = pred_logits[b]              # [num_queries, num_classes]
            out_bbox = pred_boxes[b]               # [num_queries, box_dim]
            tgt_lbl = tgt_labels[b]
            tgt_box = tgt_boxes[b]
            num_tgt = tgt_lbl.shape[0]
            indicator = padding.float().unsqueeze(1)
            # Cost class: -p_sigma(i)(ci) si vrai objet, 0 sinon
            cost_class = -out_prob[:, tgt_lbl] * indicator  # [num_queries, num_targets]
            # Cost bbox: L1 distance si vrai objet, 0 sinon
            cost_bbox = torch.cdist(out_bbox, tgt_box, p=1) * indicator
            C = self.cls_cost * cost_class + self.bbox_cost * cost_bbox
            C = C.cpu().detach().numpy()
            if not np.all(np.isfinite(C)):
                print("Invalid entries in cost matrix C!")
                print("C:", C)
                print("NaNs:", np.isnan(C).sum(), "Infs:", np.isinf(C).sum())
                raise ValueError("Cost matrix contains NaN or Inf values.")

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.long),
                            torch.as_tensor(col_ind, dtype=torch.long)))
        return indices


class HungarianMatcher_origianl:
    """
    Implements the Hungarian matching for set prediction (DETR style).
    """
    def __init__(self, cls_cost=1.0, bbox_cost=1.0, padding_idx=5):
        self.cls_cost = cls_cost
        self.bbox_cost = bbox_cost
        self.padding_idx = padding_idx

    @torch.no_grad()
    def __call__(self, pred_logits, pred_boxes, tgt_labels, tgt_boxes):
        """
        Args:
            pred_logits: [batch_size, num_queries, num_classes]
            pred_boxes:  [batch_size, num_queries, box_dim]
            tgt_labels:  list of [num_targets_i] LongTensor for each batch
            tgt_boxes:   list of [num_targets_i, box_dim] FloatTensor for each batch
        Returns:
            List of (index_pred, index_tgt) for each batch
        """
        print(pred_logits.shape, pred_boxes.shape)
        batch_size, num_queries, num_classes = pred_logits.shape
        indices = []
        for b in range(batch_size):
            out_prob = pred_logits[b]              # [num_queries, num_classes]
            out_bbox = pred_boxes[b]               # [num_queries, box_dim]
            tgt_lbl = tgt_labels[b]
            tgt_box = tgt_boxes[b]
            num_tgt = tgt_lbl.shape[0]
            # Indicator: 1 si vrai objet, 0 si padding
            if hasattr(self, 'padding_idx'):
                padding_idx = self.padding_idx
            else:
                padding_idx = num_classes
            if self.padding_idx is None:
                # Pas de padding, tous les objets sont vrais
                indicator = torch.ones(num_tgt, device=tgt_lbl.device, dtype=torch.float)
            else:
                # Avec padding
                indicator = (tgt_lbl != self.padding_idx).float()  # [num_targets]
            # Cost class: -p_sigma(i)(ci) si vrai objet, 0 sinon
            cost_class = -out_prob[:, tgt_lbl] * indicator  # [num_queries, num_targets]
            # Cost bbox: L1 distance si vrai objet, 0 sinon
            cost_bbox = torch.cdist(out_bbox, tgt_box, p=1) * indicator
            C = self.cls_cost * cost_class + self.bbox_cost * cost_bbox
            C = C.cpu().detach().numpy()
            if not np.all(np.isfinite(C)):
                print("Invalid entries in cost matrix C!")
                print("C:", C)
                print("NaNs:", np.isnan(C).sum(), "Infs:", np.isinf(C).sum())
                raise ValueError("Cost matrix contains NaN or Inf values.")

            row_ind, col_ind = linear_sum_assignment(C)
            indices.append((torch.as_tensor(row_ind, dtype=torch.long),
                            torch.as_tensor(col_ind, dtype=torch.long)))
        return indices


# ------------------------------------------------------------
# Main Module: LCR
# ------------------------------------------------------------
class LCR_withPID(nn.Module):
    """
    Inputs
      hit_embed  : (B, N_hit, D)  – GravNet/OC embedding vectors
      hit_beta   : (B, N_hit)     – β scores (objectness)
      hit_feat   : (B, N_hit, F)  – features to aggregate (E, x, y, z, t, track, …)
      hit_mask   : (B, N_hit)     – 1 for valid, 0 for padding
    Outputs
      part_fourvec : (B, K, 4)
      part_cls     : (B, K, nClasses)
      attn_weights : (B, K, N_hit)  (optional: diagnostics)
    """
    def __init__(self, embed_dim_=4, embed_dim=128, num_heads=8, K=256,
                 n_classes=7, feat_dim=5):
        super().__init__()
        self.K = K
        # self.pre_proj = nn.Linear(4, embed_dim)             ################### embed_dim　は具体的に何をさしているのか
        # self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.pre_proj = nn.Linear(embed_dim_, embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.mha    = nn.MultiheadAttention(embed_dim,
                                            num_heads,
                                            batch_first=True)
        self.cls_head   = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )
        self.fourvec_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 4)   # E, px, py, pz など
        )
        self.mass_pred = nn.Linear(embed_dim, 1)

        self.F_TRACK_VEC   = slice(0, 3)   # px,py,pz
        self.F_E_CALO      = 3             # scalar
        self.F_IS_TRACK    = 5             # scalar flag

    """""
    def forward(self, hit_embed, hit_beta, hit_feat, hit_mask=None):
        B, N, D = hit_embed.shape
        device = hit_embed.device

        # Seed selection: β top-K (per batch sample)
        idx = hit_beta.topk(self.K, dim=1).indices              # (B,K)
        batch_idx = torch.arange(B, device=device)[:, None]
        seeds = hit_embed[batch_idx, idx]                       # (B,K,D)

        # Prepare MHA tensors
        Q = self.proj_q(seeds)          # (B,K,D)
        K = hit_embed                   # (B,N,D)
        V = hit_embed                   # (B,N,D)

        # Build key-padding mask for invalid hits
        key_padding = None
        if hit_mask is not None:
            key_padding = ~hit_mask.bool()   # (B,N) True → ignore

        # Cross-attention: seeds as queries
        attn_out, attn_w = self.mha(Q, K, V,
                                    key_padding_mask=key_padding)
        # attn_out : (B,K,D),  attn_w : (B,K,N)

        # --- Aggregation to particle features -----------------
        # ① Classification
        part_cls = self.cls_head(attn_out)                      # (B,K,C)

        # ② Four-vector : weight hit_feats with attention map
        #     hit_feat   : (B,N,F)  (F >=4, assume first 4 dims are (E,x,y,z))
        w = attn_w                                           # (B,K,N)
        four_in = hit_feat[..., :4]                          # (B,N,4)
        part_fourvec = torch.bmm(w, four_in)                 # (B,K,4)
        # Optionally refine with small MLP
        part_fourvec = part_fourvec + self.fourvec_head(part_fourvec)

        return part_fourvec, part_cls, attn_w
    """""
    def forward(self, hit_embed, hit_beta, hit_feat, hit_mask=None):
        # (1) seed 抽出と cross-attention は以前と同じ
        # …
        B, N, D = hit_embed.shape
        device = hit_embed.device

        # Seed selection: β top-K (per batch sample)
        idx = hit_beta.topk(self.K, dim=1).indices              # (B,K)
        batch_idx = torch.arange(B, device=device)[:, None]
        seeds = hit_embed[batch_idx, idx]                       # (B,K,D)

        # Prepare MHA tensors
        seeds = self.pre_proj(seeds)
        Q = self.proj_q(seeds)          # (B,K,D)
        K = self.pre_proj(hit_embed)                   # (B,N,D)
        V = self.pre_proj(hit_embed)                   # (B,N,D)

        # Build key-padding mask for invalid hits
        key_padding = None
        if hit_mask is not None:
            key_padding = ~hit_mask.bool()   # (B,N) True → ignore

        attn_out, w = self.mha(Q, K, V,
                               key_padding_mask=key_padding)  # w:(B,K,N)

        # --- 集約 ----------------------------------------------------------
        # マスク
        is_trk   = hit_feat[..., self.F_IS_TRACK:self.F_IS_TRACK+1]      # (B,N,1)
        pxpy_pz  = hit_feat[..., self.F_TRACK_VEC]                 # (B,N,3)
        E_calo   = hit_feat[..., self.F_E_CALO:self.F_E_CALO+1]         # (B,N,1)

        # 1) track momentum (vector sum)
        p_vec = torch.bmm(w, pxpy_pz * is_trk)                # (B,K,3)

        # 2) track |p| and provisional mass hypothesis m (learnable)
        p_abs = torch.linalg.vector_norm(p_vec, dim=-1, keepdim=True)  # (B,K,1)
        # m_pred を small MLP で回帰
        # m_pred = F.softplus(nn.Linear(attn_out.size(-1), 1)(attn_out)) # (B,K,1)
        m_pred = F.softplus(self.mass_pred(attn_out))  # (B,K,1)
        E_trk  = torch.sqrt(torch.clamp(p_abs**2 + m_pred**2, min=1e-6))

        # 3) calorimeter energy (scalar sum)
        E_calo_sum = torch.bmm(w, E_calo * (1 - is_trk))      # (B,K,1)

        # 4) 生 4-vector
        four_raw = torch.cat([E_trk + E_calo_sum, p_vec], dim=-1)  # (B,K,4)

        # 5) MLP で微修正（calibration / leakage 補正）
        four_corr = self.fourvec_head(four_raw) + four_raw     # residual
        # ---------------------------------------------------------------

        # 分類ヘッドは以前と同じ
        part_cls = self.cls_head(attn_out)

        return four_corr, part_cls, w


class LCR_withClass(nn.Module):
    """
    Inputs
      hit_embed  : (B, N_hit, D)  – GravNet/OC embedding vectors
      hit_beta   : (B, N_hit)     – β scores (objectness)
      hit_feat   : (B, N_hit, F)  – features to aggregate (E, x, y, z, t, track, …)
      hit_mask   : (B, N_hit)     – 1 for valid, 0 for padding
    Outputs
      part_fourvec : (B, K, 4)
      part_cls     : (B, K, nClasses)
      attn_weights : (B, K, N_hit)  (optional: diagnostics)
    """
    def __init__(self, embed_dim_=4, embed_dim=128, num_heads=8, K=256,
                 n_classes=7, feat_dim=5):
        super().__init__()
        self.K = K
        # self.pre_proj = nn.Linear(4, embed_dim)             ################### embed_dim　は具体的に何をさしているのか
        # self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.pre_proj = nn.Linear(embed_dim_, embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.mha    = nn.MultiheadAttention(embed_dim,
                                            num_heads,
                                            batch_first=True)
        self.cls_head   = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )
        self.fourvec_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 4)   # E, px, py, pz など
        )
        self.mass_pred = nn.Linear(embed_dim, 1)

        self.F_TRACK_VEC   = slice(0, 3)   # px,py,pz
        self.F_E_CALO      = 3             # scalar
        self.F_IS_TRACK    = 5             # scalar flag

    """""
    def forward(self, hit_embed, hit_beta, hit_feat, hit_mask=None):
        B, N, D = hit_embed.shape
        device = hit_embed.device

        # Seed selection: β top-K (per batch sample)
        idx = hit_beta.topk(self.K, dim=1).indices              # (B,K)
        batch_idx = torch.arange(B, device=device)[:, None]
        seeds = hit_embed[batch_idx, idx]                       # (B,K,D)

        # Prepare MHA tensors
        Q = self.proj_q(seeds)          # (B,K,D)
        K = hit_embed                   # (B,N,D)
        V = hit_embed                   # (B,N,D)

        # Build key-padding mask for invalid hits
        key_padding = None
        if hit_mask is not None:
            key_padding = ~hit_mask.bool()   # (B,N) True → ignore

        # Cross-attention: seeds as queries
        attn_out, attn_w = self.mha(Q, K, V,
                                    key_padding_mask=key_padding)
        # attn_out : (B,K,D),  attn_w : (B,K,N)

        # --- Aggregation to particle features -----------------
        # ① Classification
        part_cls = self.cls_head(attn_out)                      # (B,K,C)

        # ② Four-vector : weight hit_feats with attention map
        #     hit_feat   : (B,N,F)  (F >=4, assume first 4 dims are (E,x,y,z))
        w = attn_w                                           # (B,K,N)
        four_in = hit_feat[..., :4]                          # (B,N,4)
        part_fourvec = torch.bmm(w, four_in)                 # (B,K,4)
        # Optionally refine with small MLP
        part_fourvec = part_fourvec + self.fourvec_head(part_fourvec)

        return part_fourvec, part_cls, attn_w
    """""
    def forward(self, hit_embed, hit_beta, hit_feat, hit_mask=None):
        # (1) seed 抽出と cross-attention は以前と同じ
        # …
        B, N, D = hit_embed.shape
        device = hit_embed.device

        # Seed selection: β top-K (per batch sample)
        idx = hit_beta.topk(self.K, dim=1).indices              # (B,K)
        batch_idx = torch.arange(B, device=device)[:, None]
        seeds = hit_embed[batch_idx, idx]                       # (B,K,D)

        # Prepare MHA tensors
        seeds = self.pre_proj(seeds)
        Q = self.proj_q(seeds)          # (B,K,D)
        K = self.pre_proj(hit_embed)                   # (B,N,D)
        V = self.pre_proj(hit_embed)                   # (B,N,D)

        # Build key-padding mask for invalid hits
        key_padding = None
        if hit_mask is not None:
            key_padding = ~hit_mask.bool()   # (B,N) True → ignore

        attn_out, w = self.mha(Q, K, V,
                               key_padding_mask=key_padding)  # w:(B,K,N)

        # --- 集約 ----------------------------------------------------------
        # マスク
        is_trk   = hit_feat[..., self.F_IS_TRACK:self.F_IS_TRACK+1]      # (B,N,1)
        pxpy_pz  = hit_feat[..., self.F_TRACK_VEC]                 # (B,N,3)
        E_calo   = hit_feat[..., self.F_E_CALO:self.F_E_CALO+1]         # (B,N,1)

        # 1) track momentum (vector sum)
        p_vec = torch.bmm(w, pxpy_pz * is_trk)                # (B,K,3)

        # 2) track |p| and provisional mass hypothesis m (learnable)
        p_abs = torch.linalg.vector_norm(p_vec, dim=-1, keepdim=True)  # (B,K,1)
        # m_pred を small MLP で回帰
        # m_pred = F.softplus(nn.Linear(attn_out.size(-1), 1)(attn_out)) # (B,K,1)
        m_pred = F.softplus(self.mass_pred(attn_out))  # (B,K,1)
        E_trk  = torch.sqrt(torch.clamp(p_abs**2 + m_pred**2, min=1e-6))

        # 3) calorimeter energy (scalar sum)
        E_calo_sum = torch.bmm(w, E_calo * (1 - is_trk))      # (B,K,1)

        # 4) 生 4-vector
        four_raw = torch.cat([E_trk + E_calo_sum, p_vec], dim=-1)  # (B,K,4)

        # 5) MLP で微修正（calibration / leakage 補正）
        four_corr = self.fourvec_head(four_raw) + four_raw     # residual
        # ---------------------------------------------------------------

        # 分類ヘッドは以前と同じ
        part_cls = self.cls_head(attn_out)

        return four_corr, part_cls, w

class LCR(nn.Module):
    """
    Inputs
      hit_embed  : (B, N_hit, D)  – GravNet/OC embedding vectors
      hit_beta   : (B, N_hit)     – β scores (objectness)
      hit_feat   : (B, N_hit, F)  – features to aggregate (E, x, y, z, t, track, …)
      hit_mask   : (B, N_hit)     – 1 for valid, 0 for padding
    Outputs
      part_fourvec : (B, K, 4)
      part_cls     : (B, K, nClasses)
      attn_weights : (B, K, N_hit)  (optional: diagnostics)
    """
    def __init__(self, embed_dim_=4, embed_dim=128, num_heads=8, K=256, feat_dim=5):
        super().__init__()
        self.K = K
        # self.pre_proj = nn.Linear(4, embed_dim)             ################### embed_dim　は具体的に何をさしているのか
        # self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.pre_proj = nn.Linear(embed_dim_, embed_dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(feat_dim, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, embed_dim),
        #     nn.LayerNorm(embed_dim)
        # )
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.mha    = nn.MultiheadAttention(embed_dim,
                                            num_heads,
                                            batch_first=True)
        self.fourvec_head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 4)   # E, px, py, pz など
        )
        self.mass_pred = nn.Linear(embed_dim, 1)

        # self.F_TRACK_VEC   = slice(0, 3)   # px,py,pz
        self.F_TRACK_VEC   = slice(7, 10)   # px,py,pz
        self.F_E_CALO      = 0             # scalar
        self.F_IS_TRACK    = 5             # scalar flag

        self.beta_threshold = 0.9

    """""
    def forward(self, hit_embed, hit_beta, hit_feat, hit_mask=None):
        B, N, D = hit_embed.shape
        device = hit_embed.device

        # Seed selection: β top-K (per batch sample)
        idx = hit_beta.topk(self.K, dim=1).indices              # (B,K)
        batch_idx = torch.arange(B, device=device)[:, None]
        seeds = hit_embed[batch_idx, idx]                       # (B,K,D)

        # Prepare MHA tensors
        Q = self.proj_q(seeds)          # (B,K,D)
        K = hit_embed                   # (B,N,D)
        V = hit_embed                   # (B,N,D)

        # Build key-padding mask for invalid hits
        key_padding = None
        if hit_mask is not None:
            key_padding = ~hit_mask.bool()   # (B,N) True → ignore

        # Cross-attention: seeds as queries
        attn_out, attn_w = self.mha(Q, K, V,
                                    key_padding_mask=key_padding)
        # attn_out : (B,K,D),  attn_w : (B,K,N)

        # --- Aggregation to particle features -----------------
        # ① Classification
        part_cls = self.cls_head(attn_out)                      # (B,K,C)

        # ② Four-vector : weight hit_feats with attention map
        #     hit_feat   : (B,N,F)  (F >=4, assume first 4 dims are (E,x,y,z))
        w = attn_w                                           # (B,K,N)
        four_in = hit_feat[..., :4]                          # (B,N,4)
        part_fourvec = torch.bmm(w, four_in)                 # (B,K,4)
        # Optionally refine with small MLP
        part_fourvec = part_fourvec + self.fourvec_head(part_fourvec)

        return part_fourvec, part_cls, attn_w
    """""
    def forward(self, hit_embed, hit_beta, hit_feat, hit_mask=None):
        # (1) seed 抽出と cross-attention は以前と同じ
        # …
        B, N, D = hit_embed.shape
        device = hit_embed.device

        # Seed selection: β top-K (per batch sample)
        # idx = hit_beta.topk(self.K, dim=1).indices              # (B,K)
        # batch_idx = torch.arange(B, device=device)[:, None]
        # seeds = hit_embed[batch_idx, idx]                       # (B,K,D)
        seed_mask = (hit_beta >= self.beta_threshold)  # (B, N)
        if hit_mask is not None:
            seed_mask = seed_mask & hit_mask.bool()  # valid hit かつ β閾値以上
        max_seeds = seed_mask.sum(dim=1).max().item()  # int
        seeds_padded = torch.zeros(B, max_seeds, D, device=device)
        seed_padding_mask = torch.ones(B, max_seeds, dtype=torch.bool, device=device)
        
        for b in range(B):
            selected = hit_embed[b][seed_mask[b]]  # (n_seed_b, D)
            n_seed = selected.size(0)
            if n_seed > 0:
                seeds_padded[b, :n_seed] = selected
                seed_padding_mask[b, :n_seed] = False  # False=有効, True=無効
        
        seeds = seeds_padded  # (B, Kmax, D) へ



        # Prepare MHA tensors
        seeds = self.pre_proj(seeds)
        Q = self.proj_q(seeds)          # (B,K,D)
        K = self.pre_proj(hit_embed)                   # (B,N,D)
        V = self.pre_proj(hit_embed)                   # (B,N,D)

        # Build key-padding mask for invalid hits
        key_padding = None
        if hit_mask is not None:
            key_padding = ~hit_mask.bool()   # (B,N) True → ignore

        attn_out, w = self.mha(Q, K, V,
                               key_padding_mask=key_padding)  # w:(B,K,N)

        # --- 集約 ----------------------------------------------------------
        # マスク
        is_trk   = hit_feat[..., self.F_IS_TRACK:self.F_IS_TRACK+1]      # (B,N,1)
        pxpy_pz  = hit_feat[..., self.F_TRACK_VEC]                 # (B,N,3)
        E_calo   = hit_feat[..., self.F_E_CALO:self.F_E_CALO+1]         # (B,N,1)

        # 1) track momentum (vector sum)
        p_vec = torch.bmm(w, pxpy_pz * is_trk)                # (B,K,3)

        # 2) track |p| and provisional mass hypothesis m (learnable)
        p_abs = torch.linalg.vector_norm(p_vec, dim=-1, keepdim=True)  # (B,K,1)
        # m_pred を small MLP で回帰
        # m_pred = F.softplus(nn.Linear(attn_out.size(-1), 1)(attn_out)) # (B,K,1)
        m_pred = F.softplus(self.mass_pred(attn_out))  # (B,K,1)
        E_trk  = torch.sqrt(torch.clamp(p_abs**2 + m_pred**2, min=1e-6))

        # 3) calorimeter energy (scalar sum)
        E_calo_sum = torch.bmm(w, E_calo * (1 - is_trk))      # (B,K,1)

        # 4) 生 4-vector
        four_raw = torch.cat([E_trk + E_calo_sum, p_vec], dim=-1)  # (B,K,4)

        # 5) MLP で微修正（calibration / leakage 補正）
        four_corr = self.fourvec_head(four_raw) + four_raw     # residual
        # ---------------------------------------------------------------

        return four_corr, w


class CrossAttnBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, Q, K, V, key_padding_mask=None, attn_mask=None):
        attn_out, attn_w = self.mha(Q, K, V, key_padding_mask=key_padding_mask, attn_mask=attn_mask, need_weights=True)
        Q = self.norm1(Q + attn_out)                # residual
        ffn_out = self.ffn(Q)
        out = self.norm2(Q + ffn_out)               # residual
        return out, attn_w


class LCR_Block(nn.Module):
    def __init__(self, embed_dim_=4, embed_dim=128, num_heads=8, num_layers=4, feat_dim=5):
        super().__init__()
        self.pre_proj = nn.Linear(embed_dim_, embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim)

        # Cross-attention blocks
        self.layers = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.mass_pred = nn.Linear(embed_dim, 1)
        self.fourvec_head = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 4)
        )

        self.F_TRACK_VEC   = slice(7, 10)  # px,py,pz
        self.F_E_CALO      = 0
        self.F_IS_TRACK    = 5
        self.beta_threshold = 0.9

    def forward(self, hit_embed, hit_beta, hit_feat, hit_mask=None):
        B, N, D = hit_embed.shape
        device = hit_embed.device

        # --- seed 選択 (可変長 + パディング) ---
        seed_mask = (hit_beta >= self.beta_threshold)
        if hit_mask is not None:
            seed_mask = seed_mask & hit_mask.bool()
        max_seeds = seed_mask.sum(dim=1).max().item()

        seeds_padded = torch.zeros(B, max_seeds, D, device=device)
        seed_padding_mask = torch.ones(B, max_seeds, dtype=torch.bool, device=device)
        for b in range(B):
            selected = hit_embed[b][seed_mask[b]]
            n_seed = selected.size(0)
            if n_seed > 0:
                seeds_padded[b, :n_seed] = selected
                seed_padding_mask[b, :n_seed] = False

        seeds = self.pre_proj(seeds_padded)
        Q = self.proj_q(seeds)
        K = self.pre_proj(hit_embed)
        V = self.pre_proj(hit_embed)

        # --- stacked cross-attention ---
        attn_w_all = []
        for layer in self.layers:
            Q_new, attn_w = layer(Q, K, V, key_padding_mask=(~hit_mask.bool() if hit_mask is not None else None))
            # Q = Q + Q_new
            Q = F.layer_norm(Q + Q_new, Q.shape[-1:])
            attn_w_all.append(attn_w)

        attn_out = Q  # 最終出力 (B, Kmax, D)
        w = attn_w_all[-1]  # 最終層の attention map を返す

        # --- 物理量の集約 (以前と同じ) ---
        is_trk   = hit_feat[..., self.F_IS_TRACK:self.F_IS_TRACK+1]
        pxpy_pz  = hit_feat[..., self.F_TRACK_VEC]
        E_calo   = hit_feat[..., self.F_E_CALO:self.F_E_CALO+1]

        p_vec = torch.bmm(w, pxpy_pz * is_trk)
        p_abs = torch.linalg.vector_norm(p_vec, dim=-1, keepdim=True)
        m_pred = F.softplus(self.mass_pred(attn_out))
        E_trk  = torch.sqrt(torch.clamp(p_abs**2 + m_pred**2, min=1e-6))
        E_calo_sum = torch.bmm(w, E_calo * (1 - is_trk))

        four_raw = torch.cat([E_trk + E_calo_sum, p_vec], dim=-1)
        four_corr = self.fourvec_head(four_raw) + four_raw

        return four_corr, w

class LCR_Block_modifiedOutput(nn.Module):
    def __init__(self, embed_dim_=4, embed_dim=128, num_heads=8, num_layers=4, feat_dim=5, num_particle_classes=5):
        super().__init__()
        # ここのparameter数を増やしたほうがいいかも
        self.pre_proj = nn.Linear(embed_dim_, embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim)

        self.k_embed = nn.Linear(embed_dim, embed_dim)
        self.q_embed = nn.Linear(embed_dim, embed_dim)
        self.v_embed = nn.Linear(embed_dim, embed_dim)

        # Cross-attention blocks
        self.layers = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # self.mass_pred = nn.Linear(embed_dim, 1)
        # self.fourvec_head = nn.Sequential(
        #     nn.LayerNorm(4),
        #     nn.Linear(4, 4)
        # )
        self.fourvec_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4)
        )
        self.particle_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

        # for particle identification
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_particle_classes)
        )


        self.F_TRACK_VEC   = slice(7, 10)  # px,py,pz
        self.F_E_CALO      = 0
        self.F_IS_TRACK    = 5
        self.beta_threshold = 0.9

    def forward(self, hit_embed, hit_beta, hit_feat, hit_mask=None):
        B, N, D = hit_embed.shape
        device = hit_embed.device

        # --- seed 選択 (可変長 + パディング) ---
        seed_mask = (hit_beta >= self.beta_threshold)
        if hit_mask is not None:
            seed_mask = seed_mask & hit_mask.bool()
        max_seeds = seed_mask.sum(dim=1).max().item()

        seeds_padded = torch.zeros(B, max_seeds, D, device=device)
        seed_padding_mask = torch.ones(B, max_seeds, dtype=torch.bool, device=device)
        for b in range(B):
            selected = hit_embed[b][seed_mask[b]]
            n_seed = selected.size(0)
            if n_seed > 0:
                seeds_padded[b, :n_seed] = selected
                seed_padding_mask[b, :n_seed] = False

        seeds = self.pre_proj(seeds_padded)
        Q = self.proj_q(seeds)
        K = self.pre_proj(hit_embed)
        V = self.pre_proj(hit_embed)

        # --- stacked cross-attention ---
        attn_w_all = []
        for layer in self.layers:
            Q_new, attn_w = layer(Q, K, V, key_padding_mask=(~hit_mask.bool() if hit_mask is not None else None))
            # Q = Q + Q_new
            Q = F.layer_norm(Q + Q_new, Q.shape[-1:])
            attn_w_all.append(attn_w)

        attn_out = Q  # 最終出力 (B, Kmax, D)
        w = attn_w_all[-1]  # 最終層の attention map を返す

        four_raw = self.fourvec_head(attn_out)
        E = F.softplus(four_raw[..., :1]) + 1e-6
        p = four_raw[..., 1:]
        four_corr = torch.cat([E, p], dim=-1)

        # --- ② particle / non-particle 判定 ---
        particle_logits = self.particle_head(attn_out).squeeze(-1)
        particle_prob = torch.sigmoid(particle_logits)

        particle_prob_mask = torch.sigmoid(10 * (particle_prob - 0.5)).unsqueeze(-1)  # soft mask ∈ (0,1)
        four_corr_filtered = four_corr * particle_prob_mask

        particle_cls_logits = self.classifier_head(attn_out)
        # particle_cls_logits = self.classifier_head(attn_out).softmax(-1)

        return four_corr, particle_prob, particle_cls_logits, attn_w_all

    """ # particle candidateのself attentionを含んだforward
    def forward(self, hit_embed, hit_beta, hit_feat, hit_mask=None):
        B, N, D = hit_embed.shape
        device = hit_embed.device

        # --- seed 選択 (可変長 + パディング) ---
        seed_mask = (hit_beta >= self.beta_threshold)
        if hit_mask is not None:
            seed_mask = seed_mask & hit_mask.bool()
        max_seeds = seed_mask.sum(dim=1).max().item()

        seeds_padded = torch.zeros(B, max_seeds, D, device=device)
        seed_padding_mask = torch.ones(B, max_seeds, dtype=torch.bool, device=device)
        for b in range(B):
            selected = hit_embed[b][seed_mask[b]]
            n_seed = selected.size(0)
            if n_seed > 0:
                seeds_padded[b, :n_seed] = selected
                seed_padding_mask[b, :n_seed] = False

        seeds = self.pre_proj(seeds_padded)
        Q = self.proj_q(seeds)
        K = self.pre_proj(hit_embed)
        V = self.pre_proj(hit_embed)

        # --- stacked cross-attention ---
        attn_w_all = []
        for layer in self.layers:
            Q_new, attn_w = layer(Q, K, V, key_padding_mask=(~hit_mask.bool() if hit_mask is not None else None))
            # Q = Q + Q_new
            Q = F.layer_norm(Q + Q_new, Q.shape[-1:])
            attn_w_all.append(attn_w)

        attn_out = Q  # 最終出力 (B, Kmax, D)
        w = attn_w_all[-1]  # 最終層の attention map を返す

        four_raw = self.fourvec_head(attn_out)
        E = F.softplus(four_raw[..., :1]) + 1e-6
        p = four_raw[..., 1:]
        four_corr = torch.cat([E, p], dim=-1)

        self_layer = self.layers[-1]  # 最終層の CrossAttention を Self Attention にも利用する場合
        Q_self, self_w = self_layer(attn_out, attn_out, attn_out)
        attn_out = attn_out + Q_self

        mean_self_w = self_w.mean(dim=1)  # ヘッド平均 (B, N_seed, N_seed)
        max_indices = mean_self_w.argmax(dim=-1)
        self_mask = (max_indices != torch.arange(mean_self_w.size(1), device=mean_self_w.device).unsqueeze(0))
        four_corr = four_corr.masked_fill(self_mask.unsqueeze(-1), 0.0)

        # --- ② particle / non-particle 判定 ---
        particle_logits = self.particle_head(attn_out).squeeze(-1)
        particle_prob = torch.sigmoid(particle_logits)

        particle_prob_mask = torch.sigmoid(10 * (particle_prob - 0.5)).unsqueeze(-1)  # soft mask ∈ (0,1)
        four_corr_filtered = four_corr * particle_prob_mask

        return four_corr_filtered, particle_prob, attn_w_all
        """


class LCR_Block_modifiedOutput_moreParameters(nn.Module):
    def __init__(self, embed_dim_=17, embed_dim=128, num_heads=8, num_layers=4, feat_dim=5, num_particle_classes=5):
        super().__init__()
        self.k_embed = nn.Sequential(
            nn.Linear(embed_dim_, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.q_embed = nn.Sequential(
            nn.Linear(embed_dim_, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.v_embed = nn.Sequential(
            nn.Linear(embed_dim_, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Cross-attention blocks
        self.layers = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # self.mass_pred = nn.Linear(embed_dim, 1)
        # self.fourvec_head = nn.Sequential(
        #     nn.LayerNorm(4),
        #     nn.Linear(4, 4)
        # )
        self.fourvec_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4)
        )
        self.particle_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

        # for particle identification
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_particle_classes)
        )


        self.F_TRACK_VEC   = slice(7, 10)  # px,py,pz
        self.F_E_CALO      = 0
        self.F_IS_TRACK    = 5
        self.beta_threshold = 0.9

    def forward(self, hit_embed, hit_mask=None):
        B, N, D = hit_embed.shape
        device = hit_embed.device

        # --- seed 選択 (可変長 + パディング) ---
        seed_mask = (hit_embed[:,:,0] >= self.beta_threshold)
        if hit_mask is not None:
            seed_mask = seed_mask & hit_mask.bool()
        max_seeds = seed_mask.sum(dim=1).max().item()

        seeds_padded = torch.zeros(B, max_seeds, D, device=device)
        seed_padding_mask = torch.ones(B, max_seeds, dtype=torch.bool, device=device)
        for b in range(B):
            selected = hit_embed[b][seed_mask[b]]
            n_seed = selected.size(0)
            if n_seed > 0:
                seeds_padded[b, :n_seed] = selected
                seed_padding_mask[b, :n_seed] = False

        Q = self.q_embed(seeds_padded)
        K = self.k_embed(hit_embed)
        V = self.v_embed(hit_embed)

        # --- stacked cross-attention ---
        attn_w_all = []
        for layer in self.layers:
            Q_new, attn_w = layer(Q, K, V, key_padding_mask=(~hit_mask.bool() if hit_mask is not None else None))
            # Q = Q + Q_new
            Q = F.layer_norm(Q + Q_new, Q.shape[-1:])
            attn_w_all.append(attn_w)

        attn_out = Q  # 最終出力 (B, Kmax, D)
        w = attn_w_all[-1]  # 最終層の attention map を返す

        four_raw = self.fourvec_head(attn_out)
        E = F.softplus(four_raw[..., :1]) + 1e-6
        p = four_raw[..., 1:]
        four_corr = torch.cat([E, p], dim=-1)

        # --- ② particle / non-particle 判定 ---
        particle_logits = self.particle_head(attn_out).squeeze(-1)
        particle_prob = torch.sigmoid(particle_logits)

        # particle_prob_mask = torch.sigmoid(10 * (particle_prob - 0.5)).unsqueeze(-1)  # soft mask ∈ (0,1)
        # four_corr_filtered = four_corr * particle_prob_mask

        particle_cls_logits = self.classifier_head(attn_out)
        # particle_cls_logits = self.classifier_head(attn_out).softmax(-1)

        return four_corr, particle_prob, particle_cls_logits, attn_w_all, seed_padding_mask

class LCR_Block_modifiedOutput_moreParameters_trackQuery(nn.Module):
    def __init__(self, embed_dim_=17, embed_dim=128, num_heads=8, num_layers=8, feat_dim=5, num_particle_classes=5):
        super().__init__()
        self.k_embed = nn.Sequential(
            nn.Linear(embed_dim_, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.q_embed = nn.Sequential(
            nn.Linear(embed_dim_, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.v_embed = nn.Sequential(
            nn.Linear(embed_dim_, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Cross-attention blocks
        self.layers = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # self.mass_pred = nn.Linear(embed_dim, 1)
        # self.fourvec_head = nn.Sequential(
        #     nn.LayerNorm(4),
        #     nn.Linear(4, 4)
        # )
        self.fourvec_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4)
        )
        # self.fourvec_head_neutral = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, 4)
        # )
        self.particle_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

        # for particle identification
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_particle_classes)
        )


        self.F_TRACK_VEC   = slice(7, 10)  # px,py,pz
        self.F_E_CALO      = 0
        self.F_IS_TRACK    = 5
        self.beta_threshold = 0.9

    def forward(self, hit_embed, query, hit_mask=None):
        B, N, D = hit_embed.shape
        device = hit_embed.device

        Q = self.q_embed(query)
        K = self.k_embed(hit_embed)
        V = self.v_embed(hit_embed)

        # --- stacked cross-attention ---
        attn_w_all = []
        for layer in self.layers:
            Q_new, attn_w = layer(Q, K, V, key_padding_mask=(~hit_mask.bool() if hit_mask is not None else None))
            # Q = Q + Q_new
            Q = F.layer_norm(Q + Q_new, Q.shape[-1:])
            attn_w_all.append(attn_w)

        attn_out = Q  # 最終出力 (B, Kmax, D)
        w = attn_w_all[-1]  # 最終層の attention map を返す

        four_raw = self.fourvec_head(attn_out)
        E = F.softplus(four_raw[..., :1]) + 1e-6
        p = four_raw[..., 1:]
        four_corr = torch.cat([E, p], dim=-1)

        # --- ② particle / non-particle 判定 ---
        particle_logits = self.particle_head(attn_out).squeeze(-1)
        particle_prob = torch.sigmoid(particle_logits)

        # particle_prob_mask = torch.sigmoid(10 * (particle_prob - 0.5)).unsqueeze(-1)  # soft mask ∈ (0,1)
        # four_corr_filtered = four_corr * particle_prob_mask

        particle_cls_logits = self.classifier_head(attn_out)
        # particle_cls_logits = self.classifier_head(attn_out).softmax(-1)

        return four_corr, particle_prob, particle_cls_logits, attn_w_all


"""
class LCR_Block_modifiedOutput(nn.Module):
    def __init__(self, embed_dim_=4, embed_dim=128, num_heads=8, num_layers=4, feat_dim=5):
        super().__init__()
        self.pre_proj = nn.Linear(embed_dim_, embed_dim)
        self.proj_q = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads

        # Cross + Self Attention のstack
        self.cross_layers = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.self_layers = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        # Cross-attention blocks
        self.layers = nn.ModuleList([
            CrossAttnBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])

        self.mass_pred = nn.Linear(embed_dim, 1)
        # 4-vector 予測ヘッド
        self.fourvec_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4)
        )
        # particle / non-particle 判定ヘッド
        self.particle_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1)
        )

        self.F_TRACK_VEC   = slice(7, 10)  # px,py,pz
        self.F_E_CALO      = 0
        self.F_IS_TRACK    = 5
        self.beta_threshold = 0.9

    def forward(self, hit_embed, hit_beta, hit_feat, hit_mask=None):
        B, N, D = hit_embed.shape
        device = hit_embed.device

        # --- seed 選択 (可変長 + パディング) ---
        seed_mask = (hit_beta >= self.beta_threshold)
        if hit_mask is not None:
            seed_mask = seed_mask & hit_mask.bool()
        max_seeds = max(1, seed_mask.sum(dim=1).max().item())

        seeds_padded = torch.zeros(B, max_seeds, D, device=device)
        seed_padding_mask = torch.ones(B, max_seeds, dtype=torch.bool, device=device)
        for b in range(B):
            selected = hit_embed[b][seed_mask[b]]
            n_seed = selected.size(0)
            if n_seed > 0:
                seeds_padded[b, :n_seed] = selected
                seed_padding_mask[b, :n_seed] = False

        seeds = self.pre_proj(seeds_padded)
        Q = self.proj_q(seeds)
        K = self.pre_proj(hit_embed)
        V = self.pre_proj(hit_embed)

        ""
        # --- stacked cross-attention ---
        attn_w_all = []
        for layer in self.layers:
            Q_new, attn_w = layer(Q, K, V, key_padding_mask=(~hit_mask.bool() if hit_mask is not None else None))
            # Q = Q + Q_new
            Q = F.layer_norm(Q + Q_new, Q.shape[-1:])
            attn_w_all.append(attn_w)

        attn_out = Q  # 最終出力 (B, Kmax, D)
        w = attn_w_all[-1]  # 最終層の attention map を返す

        # --- 物理量の集約 (以前と同じ) ---
        is_trk   = hit_feat[..., self.F_IS_TRACK:self.F_IS_TRACK+1]
        pxpy_pz  = hit_feat[..., self.F_TRACK_VEC]
        E_calo   = hit_feat[..., self.F_E_CALO:self.F_E_CALO+1]

        p_vec = torch.bmm(w, pxpy_pz * is_trk)
        p_abs = torch.linalg.vector_norm(p_vec, dim=-1, keepdim=True)
        m_pred = F.softplus(self.mass_pred(attn_out))
        E_trk  = torch.sqrt(torch.clamp(p_abs**2 + m_pred**2, min=1e-6))
        E_calo_sum = torch.bmm(w, E_calo * (1 - is_trk))

        four_raw = torch.cat([E_trk + E_calo_sum, p_vec], dim=-1)
        four_corr = self.fourvec_head(four_raw) + four_raw

        return four_corr, w
        ""

        # --- track/neutral mask (④) ---
        is_trk_hit = hit_feat[..., self.F_IS_TRACK:self.F_IS_TRACK+1]  # (B, N, 1)
        is_trk_hit = is_trk_hit.bool()

        attn_w_all = []
        for i, (cross_layer, self_layer) in enumerate(zip(self.cross_layers, self.self_layers)):
            # (④) neutral seed は track hit を見ないように attention maskを作成
            # mask: True = 無視
            seed_is_neutral = (hit_beta < self.beta_threshold).unsqueeze(-1)  # (B, N_seed, 1)
            neutral_ignore_track = (seed_is_neutral & is_trk_hit.transpose(1, 2))
            attn_mask = neutral_ignore_track.float() * -1e9  # attention weightを抑制
            # attn_mask = attn_mask[0]
            B, N_q, N_k = attn_mask.shape
            if N_q == 0 or N_k == 0:
                attn_mask = None
            else:
                attn_mask = neutral_ignore_track.float() * -1e9
                attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
                attn_mask = attn_mask.view(B * self.num_heads, N_q, N_k)
                attn_mask = attn_mask[:, :Q.size(1), :]

            # --- Cross Attention ---
            Q, attn_w = cross_layer(Q, K, V, key_padding_mask=(~hit_mask.bool() if hit_mask is not None else None), attn_mask=attn_mask)
            attn_w_all.append(attn_w)

            # --- Self Attention (③) ---
            # cluster内部で重複抑制（自分が最大相関でないseedはattenされる）
            Q_self, self_w = self_layer(Q, Q, Q)
            Q = Q + Q_self  # residualで更新

        attn_out = Q  # (B, N_seed, D)

        # --- ① 4-vector 予測 ---
        four_raw = self.fourvec_head(attn_out)
        E = F.softplus(four_raw[..., :1]) + 1e-6
        p = four_raw[..., 1:]
        four_corr = torch.cat([E, p], dim=-1)

        # --- ② particle / non-particle 判定 ---
        particle_logits = self.particle_head(attn_out).squeeze(-1)
        particle_prob = torch.sigmoid(particle_logits)

        particle_prob_mask = torch.sigmoid(10 * (particle_prob - 0.5)).unsqueeze(-1)  # soft mask ∈ (0,1)
        four_corr_filtered = four_corr * particle_prob_mask

        return four_corr_filtered, particle_prob, attn_w_all
"""



# ------------------------------------------------------------
# Dummy usage example
# ------------------------------------------------------------
if __name__ == "__main__":
    B_, N_hit, D_, F_ = 2, 6000, 128, 6
    K_ = 128

    hit_embed = torch.randn(B_, N_hit, D_)
    hit_beta  = torch.sigmoid(torch.randn(B_, N_hit))
    hit_feat  = torch.randn(B_, N_hit, F_)
    hit_mask  = torch.ones(B_, N_hit).bool()   # no padding in this toy

    model = LCR(embed_dim=D_, num_heads=8, K=K_, n_classes=7, feat_dim=4)
    part_fourvec, part_cls, attn_w = model(hit_embed, hit_beta,
                                           hit_feat, hit_mask)

    print("Output four-vectors:", part_fourvec.shape)  # (B,K,4)
    print("Output class-logits:", part_cls.shape)      # (B,K,7)

    # --- Dummy truth & loss demo -----------------------------
    true_fourvec = torch.randn(B_, 40, 4)    # at most 40 true particles
    true_cls     = torch.randint(0, 7, (B_, 40))

    loss = hungarian_set_loss(part_fourvec, part_cls,
                              true_fourvec, true_cls)
    loss.backward()
    print("Hungarian loss:", loss.item())
