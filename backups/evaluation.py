import numpy as np
import torch

def cluster_purity_efficiency(labels_pred, labels_true, m, n):
    # m: pred cluster id, n: true cluster id
    pred_hits = (labels_pred == m)
    true_hits = (labels_true == n)
    inter = (pred_hits & true_hits).sum()
    purity = inter / max(pred_hits.sum(), 1)
    efficiency = inter / max(true_hits.sum(), 1)
    f1 = 0.0 if (purity+efficiency)==0 else 2*purity*efficiency/(purity+efficiency)
    return float(purity), float(efficiency), float(f1)

def event_hit_metrics(labels_pred, labels_true, matches, weights=None):
    # weights: 各真粒子の重み（例：真ヒット数 or 真エネルギー）
    per_pair = []
    for m, n in matches:
        p, e, f1 = cluster_purity_efficiency(labels_pred, labels_true, m, n)
        per_pair.append((p,e,f1))
    per_pair = np.array(per_pair) if len(per_pair)>0 else np.zeros((0,3))
    if weights is None or len(per_pair)==0:
        return dict(purity=per_pair[:,0].mean() if len(per_pair) else 0.0,
                    efficiency=per_pair[:,1].mean() if len(per_pair) else 0.0,
                    f1=per_pair[:,2].mean() if len(per_pair) else 0.0)
    w = np.asarray(weights[:len(per_pair)])  # 長さ合わせ
    w = w / w.sum() if w.sum()>0 else np.ones_like(w)/len(w)
    return dict(purity=(per_pair[:,0]*w).sum(),
                efficiency=(per_pair[:,1]*w).sum(),
                f1=(per_pair[:,2]*w).sum())

def event_fourvec_metrics(P_pred, P_true, matches, weight_true_E=False):
    # P_*: (K,4) with (E,px,py,pz)
    # 1) ペア単位の誤差
    err = []
    for m, n in matches:
        E_pred, p_pred = P_pred[m,0], P_pred[m,1:]
        E_true, p_true = P_true[n,0], P_true[n,1:]
        mae_E  = torch.abs(E_pred - E_true)
        mape_E = mae_E / (torch.clamp(E_true, min=1e-6))
        mae_p  = torch.abs(torch.norm(p_pred, p=2) - torch.norm(p_true, p=2))
        # 方向差（ΔR相当）：単純化してコサイン類似度から角度に変換
        cosang = torch.clamp(torch.dot(p_pred, p_true) /
                             (torch.norm(p_pred, p=2)*torch.norm(p_true, p=2) + 1e-6),
                             -1.0, 1.0)
        ang = torch.arccos(cosang)  # radians
        err.append([mae_E.item(), mape_E.item(), mae_p.item(), ang.item()])
    err = np.array(err) if len(err)>0 else np.zeros((0,4))
    # 2) イベント合計4ベクトルの差
    sum_pred = P_pred.sum(dim=0)
    sum_true = P_true.sum(dim=0)
    ev_mae = torch.abs(sum_pred - sum_true).cpu().numpy()  # (4,)

    if len(err)==0:
        per_pair = dict(mae_E=0.0, mape_E=0.0, mae_p=0.0, ang=0.0)
    else:
        if weight_true_E:
            # 真エネルギーで重み付け
            w = np.array([P_true[n,0].item() for _,n in matches])
            w = w / (w.sum() + 1e-9)
            per_pair = dict(mae_E=(err[:,0]*w).sum(),
                            mape_E=(err[:,1]*w).sum(),
                            mae_p=(err[:,2]*w).sum(),
                            ang=(err[:,3]*w).sum())
        else:
            per_pair = dict(mae_E=err[:,0].mean(),
                            mape_E=err[:,1].mean(),
                            mae_p=err[:,2].mean(),
                            ang=err[:,3].mean())
    return per_pair, dict(ev4_mae_E=ev_mae[0], ev4_mae_px=ev_mae[1], ev4_mae_py=ev_mae[2], ev4_mae_pz=ev_mae[3])
