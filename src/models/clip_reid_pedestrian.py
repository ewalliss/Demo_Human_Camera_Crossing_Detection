"""CLIPReIDPedestrianModel — ViT-B/16 + SIE + OLP for pedestrian re-ID.

Two forward modes controlled by stage flag:
  stage=1 → returns (img_feat, text_feat) for SupConLoss
  stage=2 → returns (img_feat, cls_score) for ID + Triplet + I2T losses
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

from .olp_head import OLPHead
from .prompt_learner import PromptLearner
from .sie_layer import SIELayer


class TextEncoder(nn.Module):
    """Wraps CLIP's text transformer to accept pre-built prompt embeddings."""

    def __init__(self, clip_model: CLIPModel) -> None:
        super().__init__()
        self.transformer = clip_model.text_model.encoder
        self.final_ln = clip_model.text_model.final_layer_norm
        self.text_proj = clip_model.text_projection
        self.pos_embedding = clip_model.text_model.embeddings.position_embedding

    def forward(self, prompt_emb: torch.Tensor) -> torch.Tensor:
        """Encode prompt embeddings to L2-normalised text features.

        Args:
            prompt_emb: shape (N, seq_len, embed_dim)

        Returns:
            Text features, shape (N, 512), L2-normalised.
        """
        seq_len = prompt_emb.shape[1]
        pos_ids = torch.arange(seq_len, device=prompt_emb.device).unsqueeze(0)
        hidden = prompt_emb + self.pos_embedding(pos_ids)

        # Causal mask: each token attends to previous tokens only
        causal_mask = torch.full(
            (seq_len, seq_len), float("-inf"), device=hidden.device
        ).triu(diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        out = self.transformer(
            inputs_embeds=hidden,
            attention_mask=None,
            causal_attention_mask=causal_mask,
        ).last_hidden_state  # (N, seq_len, D)

        # Extract EOT (last non-padding) position = index -1
        eot_feat = out[:, -1, :]                     # (N, D_text)
        eot_feat = self.final_ln(eot_feat)
        feat = self.text_proj(eot_feat)              # (N, 512)
        return F.normalize(feat, p=2, dim=-1)


class CLIPReIDPedestrianModel(nn.Module):
    """CLIP ViT-B/16 + SIE + OLP pedestrian re-identification model.

    Args:
        num_pids:    Number of training identities.
        num_cams:    Number of distinct camera IDs.
        num_views:   Number of viewpoint bins.
        olp_k:       Top-k patches for OLP head.
        clip_name:   HuggingFace model id.
        template:    Prompt template (must contain 'person').
        n_ctx:       Learnable context tokens per identity.
    """

    EMBED_DIM = 512      # CLIP projection output
    PATCH_DIM = 768      # ViT-B/16 hidden dimension

    def __init__(
        self,
        num_pids: int,
        num_cams: int = 30,
        num_views: int = 4,
        olp_k: int = 16,
        clip_name: str = "openai/clip-vit-base-patch16",
        template: str = "a photo of a X X X X person",
        n_ctx: int = 4,
    ) -> None:
        super().__init__()

        clip = CLIPModel.from_pretrained(clip_name)

        # Image encoder — full ViT-B/16 vision model
        self.image_encoder = clip.vision_model
        self.visual_proj = clip.visual_projection  # 768 → 512

        # Text path
        self.prompt_learner = PromptLearner(num_pids, clip, template, n_ctx)
        self.text_encoder = TextEncoder(clip)

        # SIE — injected after visual projection
        self.sie = SIELayer(num_cams=num_cams, num_views=num_views, embed_dim=self.EMBED_DIM)

        # OLP — fuses [CLS] + patch features
        self.olp = OLPHead(patch_dim=self.PATCH_DIM, out_dim=self.EMBED_DIM, k=olp_k)

        # ID classifier head
        self.bn = nn.BatchNorm1d(self.EMBED_DIM)
        self.classifier = nn.Linear(self.EMBED_DIM, num_pids, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        self.num_pids = num_pids

    # ── Stage control ─────────────────────────────────────────────────────────

    def freeze_for_stage1(self) -> None:
        """Stage 1: freeze encoders, train only PromptLearner."""
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.visual_proj.parameters():
            p.requires_grad = False
        for p in self.sie.parameters():
            p.requires_grad = False
        for p in self.olp.parameters():
            p.requires_grad = False
        for p in self.bn.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = False
        # PromptLearner remains trainable
        for p in self.prompt_learner.parameters():
            p.requires_grad = True

    def freeze_for_stage2(self) -> None:
        """Stage 2: freeze text path, train image encoder + SIE + OLP + heads."""
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        for p in self.prompt_learner.parameters():
            p.requires_grad = False
        for p in self.image_encoder.parameters():
            p.requires_grad = True
        for p in self.visual_proj.parameters():
            p.requires_grad = True
        for p in self.sie.parameters():
            p.requires_grad = True
        for p in self.olp.parameters():
            p.requires_grad = True
        for p in self.bn.parameters():
            p.requires_grad = True
        for p in self.classifier.parameters():
            p.requires_grad = True

    # ── Forward ───────────────────────────────────────────────────────────────

    def encode_image(
        self,
        pixel_values: torch.Tensor,
        cam_ids: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract normalised image features via ViT + SIE + OLP.

        Returns:
            (global_feat, fused_feat) both shape (N, 512), L2-normalised.
        """
        vision_out = self.image_encoder(pixel_values=pixel_values, output_hidden_states=False)
        # pooler_output: (N, 768) raw CLS; last_hidden_state: (N, 197, 768)
        cls_raw = vision_out.pooler_output          # (N, 768)
        patch_tokens = vision_out.last_hidden_state[:, 1:, :]  # (N, 196, 768)

        # Project CLS to 512-d
        global_feat = self.visual_proj(cls_raw)     # (N, 512)
        global_feat = F.normalize(global_feat, p=2, dim=-1)

        # SIE conditioning (cam + view)
        global_feat = self.sie(global_feat, cam_ids, view_ids)
        global_feat = F.normalize(global_feat, p=2, dim=-1)

        # OLP: fuse global CLS + top-k patch features
        fused_feat = self.olp(global_feat, patch_tokens)  # already normalised inside OLP

        return global_feat, fused_feat

    def encode_text(self, pids: torch.Tensor | None = None) -> torch.Tensor:
        """Encode learnable prompts for given pids.

        Returns:
            Text features, shape (N_pids, 512), L2-normalised.
        """
        prompt_emb = self.prompt_learner(pids)     # (N, seq_len, D)
        return self.text_encoder(prompt_emb)       # (N, 512)

    def forward(
        self,
        pixel_values: torch.Tensor,
        pids: torch.Tensor,
        cam_ids: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass returning both image and text features + logits.

        Returns dict with keys:
            img_feat:  (N, 512) global visual feature (SIE-conditioned)
            fused_feat: (N, 512) OLP-fused visual feature
            text_feat: (N, 512) text feature for unique pids in batch
            cls_score: (N, num_pids) classification logits
            batch_pids: unique pid tensor used for text encoding
        """
        global_feat, fused_feat = self.encode_image(pixel_values, cam_ids, view_ids)

        # BN + classifier on fused feature
        feat_bn = self.bn(fused_feat)
        cls_score = self.classifier(feat_bn)

        # Text features for unique pids in batch
        unique_pids = pids.unique()
        text_feat = self.encode_text(unique_pids)

        return {
            "img_feat": global_feat,
            "fused_feat": fused_feat,
            "text_feat": text_feat,
            "cls_score": cls_score,
            "batch_pids": unique_pids,
        }

    @torch.no_grad()
    def extract_features(
        self,
        pixel_values: torch.Tensor,
        cam_ids: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Inference-only: extract normalised fused features."""
        _, fused = self.encode_image(pixel_values, cam_ids, view_ids)
        return fused
