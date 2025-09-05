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

        return four_corr, w



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
