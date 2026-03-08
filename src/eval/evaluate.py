"""Evaluation: mAP + CMC@1/5/10 with optional k-reciprocal re-ranking."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .reranking import k_reciprocal_rerank


@torch.no_grad()
def extract_features(
    model,
    loader: DataLoader,
    device: torch.device,
    fp16: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract features for all samples in loader.

    Returns:
        (feats, pids, cam_ids) — all on CPU.
    """
    feats_list, pid_list, cam_list = [], [], []
    model.eval()

    for imgs, pids, cam_ids, view_ids in tqdm(loader, desc="Extracting", leave=False):
        imgs = imgs.to(device)
        cam_ids = cam_ids.to(device)
        view_ids = view_ids.to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
            feat = model.extract_features(imgs, cam_ids, view_ids)
        feats_list.append(feat.cpu().float())
        pid_list.append(pids)
        cam_list.append(cam_ids.cpu())

    return (
        torch.cat(feats_list, dim=0),
        torch.cat(pid_list, dim=0),
        torch.cat(cam_list, dim=0),
    )


def compute_metrics(
    dist_mat: torch.Tensor,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
    query_cams: torch.Tensor,
    gallery_cams: torch.Tensor,
    max_rank: int = 10,
) -> dict[str, float]:
    """Compute mAP and CMC@{1,5,10} with same-camera exclusion.

    Args:
        dist_mat: (Q, G) distance matrix (lower = more similar).

    Returns:
        Dict with keys: mAP, rank1, rank5, rank10.
    """
    Q = query_pids.shape[0]
    all_ap, all_cmc = [], []

    for q in range(Q):
        qpid = query_pids[q].item()
        qcam = query_cams[q].item()

        # Exclude same-camera same-id (standard ReID protocol)
        valid = ~((gallery_pids == qpid) & (gallery_cams == qcam))
        dist_q = dist_mat[q][valid]
        gpids = gallery_pids[valid]

        sorted_idx = dist_q.argsort()
        matched = (gpids[sorted_idx] == qpid)

        if not matched.any():
            continue

        # CMC
        cmc = matched.cumsum(0).clamp(max=1).float()
        all_cmc.append(cmc[:max_rank])

        # AP
        n_pos = matched.sum().item()
        pos_ranks = torch.where(matched)[0].float() + 1  # 1-indexed
        precision_at_k = torch.arange(1, n_pos + 1, dtype=torch.float) / pos_ranks
        ap = precision_at_k.mean().item()
        all_ap.append(ap)

    if not all_ap:
        return {"mAP": 0.0, "rank1": 0.0, "rank5": 0.0, "rank10": 0.0}

    cmc_avg = torch.stack(all_cmc).mean(dim=0)
    return {
        "mAP": float(sum(all_ap) / len(all_ap) * 100),
        "rank1": float(cmc_avg[0] * 100),
        "rank5": float(cmc_avg[4] * 100) if max_rank >= 5 else 0.0,
        "rank10": float(cmc_avg[9] * 100) if max_rank >= 10 else 0.0,
    }


def evaluate(
    model,
    query_loader: DataLoader,
    gallery_loader: DataLoader,
    device: torch.device,
    fp16: bool = True,
    use_rerank: bool = True,
    k1: int = 20,
    k2: int = 6,
    lambda_: float = 0.3,
) -> dict[str, float]:
    """Full evaluation pipeline: extract → distance → re-rank → metrics."""
    q_feats, q_pids, q_cams = extract_features(model, query_loader, device, fp16)
    g_feats, g_pids, g_cams = extract_features(model, gallery_loader, device, fp16)

    # Cosine distance (FP32)
    q_feats = F.normalize(q_feats, p=2, dim=-1)
    g_feats = F.normalize(g_feats, p=2, dim=-1)
    dist = 1.0 - torch.matmul(q_feats, g_feats.T)

    base_metrics = compute_metrics(dist, q_pids, g_pids, q_cams, g_cams)

    if not use_rerank:
        return base_metrics

    rr_dist = k_reciprocal_rerank(
        q_feats.to(device), g_feats.to(device), k1=k1, k2=k2, lambda_=lambda_
    ).cpu()
    rr_metrics = compute_metrics(rr_dist, q_pids, g_pids, q_cams, g_cams)

    print(
        f"\n  [w/o rerank] mAP={base_metrics['mAP']:.1f}%  R1={base_metrics['rank1']:.1f}%"
        f"\n  [w/  rerank] mAP={rr_metrics['mAP']:.1f}%  R1={rr_metrics['rank1']:.1f}%"
    )
    return rr_metrics
