# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
from pathlib import Path
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.configs.third_delta import (
    LEGACY_THIRD_PARAMETER_PREFIXES,
    THIRD_PARAMETER_PREFIXES,
    validate_third_checkpoint_config,
)
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from gr00t.model.modules.third_delta import THIRD_QUERY_COUNT, third_prediction_metrics


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True
    supports_rtc = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        config.validate_phase_config()

        if config.use_action_query_gate:
            if not config.use_third_view_aux_loss:
                raise ValueError("use_action_query_gate requires use_third_view_aux_loss=True")
            if config.num_learnable_queries <= 0:
                raise ValueError("use_action_query_gate requires num_learnable_queries > 0")

        gate_config = dict(
            use_action_query_gate=config.use_action_query_gate,
            query_gate_init_prob=config.query_gate_init_prob,
        )

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                **gate_config,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                **gate_config,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        if config.use_third_view_aux_loss:
            if config.num_learnable_queries <= 0:
                raise ValueError("Third-view supervision requires num_learnable_queries > 0")
            self.learnable_queries = nn.Parameter(
                torch.empty(config.num_learnable_queries, self.input_embedding_dim)
            )
            self.query_head = nn.Sequential(
                nn.Linear(self.model.config.output_dim, config.query_ffn_hidden_dim),
                nn.GELU(),
                nn.Linear(config.query_ffn_hidden_dim, config.backbone_embedding_dim),
            )
            self.initialize_third_view_aux_parameters()

        if config.use_phase_aux_loss:
            self.phase_queries = nn.Parameter(torch.empty(4, self.input_embedding_dim))
            self.phase_head = nn.Sequential(
                nn.RMSNorm(self.model.inner_dim, eps=1e-6),
                nn.Flatten(start_dim=1),
                nn.Linear(4 * self.model.inner_dim, config.phase_mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(
                    config.phase_mlp_hidden_dim, self.action_horizon * config.phase_num_classes
                ),
            )
            self.initialize_phase_aux_parameters()

        if config.third_loss and config.third_framework == "queries":
            self.third_queries = nn.Parameter(
                torch.empty(THIRD_QUERY_COUNT, self.input_embedding_dim)
            )
            self.third_query_head = nn.Sequential(
                nn.Linear(self.model.config.output_dim, config.query_ffn_hidden_dim),
                nn.GELU(),
                nn.Linear(config.query_ffn_hidden_dim, config.backbone_embedding_dim),
            )
            self.initialize_third_parameters()

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Pin the time-sampling Beta to CPU/fp32 explicitly. The action head can
        # be instantiated under a meta / no_init_weights default-device context
        # (e.g. nested from_pretrained). A Beta built from bare Python floats
        # would then place its concentration tensors on the meta device (or in
        # the active default dtype, e.g. bf16). With validate_args enabled that
        # already fails here in __init__ (Beta's internal .item() check cannot
        # run on meta); even with validation off, sample_time would later raise
        # or return garbage. Explicit device/dtype here makes the sampler depend
        # only on the config, not on the construction-time device/dtype context,
        # so the noise schedule is identical across SDPA/FA2/FA4 and meta vs.
        # real-device loads. config is the canonical source for these values.
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def initialize_third_view_aux_parameters(self) -> None:
        """Explicitly initialize parameters that are absent from legacy checkpoints."""
        nn.init.normal_(self.learnable_queries, mean=0.0, std=0.02)
        for module in self.query_head.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def initialize_third_parameters(self) -> None:
        """Initialize all third-view query parameters when absent from a base checkpoint."""
        nn.init.normal_(self.third_queries, mean=0.0, std=0.02)
        for module in self.third_query_head.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def initialize_phase_aux_parameters(self) -> None:
        """Initialize the complete phase branch when loading a base checkpoint."""
        nn.init.normal_(self.phase_queries, mean=0.0, std=0.02)
        for module in self.phase_head.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _phase_outputs(self, query_states: torch.Tensor, action_input: BatchFeature) -> dict:
        logits = self.phase_head(query_states).reshape(
            query_states.shape[0], self.action_horizon, self.config.phase_num_classes
        )
        if "phase_labels" not in action_input:
            raise ValueError("Phase supervision requires phase_labels for every training sample")
        labels = action_input.phase_labels
        if labels.dtype != torch.int64 or labels.shape != logits.shape[:2]:
            raise ValueError("phase_labels must be int64 with shape [B, action_horizon]")
        valid_labels = labels != -100
        if (
            (labels[valid_labels] < 0) | (labels[valid_labels] >= self.config.phase_num_classes)
        ).any():
            raise ValueError("phase_labels contains an out-of-range class ID")
        valid = valid_labels & action_input.action_mask.bool().any(dim=-1)
        targets = labels.masked_fill(~valid, -100)
        frame_losses = F.cross_entropy(
            logits.float().reshape(-1, self.config.phase_num_classes),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)
        count = valid.sum()
        phase_loss = frame_losses.sum() / count.clamp_min(1)
        accuracy = ((logits.argmax(dim=-1) == labels) & valid).sum().float() / count.clamp_min(1)
        return {
            "phase_logits": logits,
            "phase_loss": phase_loss,
            "phase_accuracy": accuracy.detach(),
            "phase_valid_frames": count,
        }

    def _dit_self_attention_mask(
        self, hidden_states: torch.Tensor, num_state_tokens: int
    ) -> torch.Tensor | None:
        """Restrict auxiliary context reads while allowing actions to read every token."""
        if self.config.third_loss and self.config.third_framework == "queries":
            # State, queries and actions all read each other bidirectionally.
            return None
        if self.config.use_phase_aux_loss:
            batch_size, seq_len = hidden_states.shape[:2]
            context_end = num_state_tokens + 4
            mask = hidden_states.new_zeros((batch_size, seq_len, seq_len))
            mask[:, :context_end, context_end:] = -torch.inf
            return mask
        if not self.config.use_third_view_aux_loss:
            return None

        batch_size, seq_len = hidden_states.shape[:2]
        query_end = num_state_tokens + self.config.num_learnable_queries
        mask = hidden_states.new_zeros((batch_size, seq_len, seq_len))
        mask[:, :num_state_tokens, num_state_tokens:] = torch.finfo(mask.dtype).min
        mask[:, num_state_tokens:query_end, query_end:] = torch.finfo(mask.dtype).min
        return mask

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
            if self.config.use_third_view_aux_loss:
                self.learnable_queries.requires_grad_(False)
                self.query_head.requires_grad_(False)
            if self.config.use_phase_aux_loss:
                self.phase_queries.requires_grad_(False)
                self.phase_head.requires_grad_(False)
            if self.config.third_loss and self.config.third_framework == "queries":
                self.third_queries.requires_grad_(False)
                self.third_query_head.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
                if self.config.use_third_view_aux_loss:
                    self.query_head.eval()
                if self.config.use_phase_aux_loss:
                    self.phase_head.eval()
                if self.config.third_loss and self.config.third_framework == "queries":
                    self.third_query_head.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _action_backbone_output(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """Extract and left-pad user-1 before the trainable VL postprocessor."""
        if not self.config.use_third_view_aux_loss:
            return BatchFeature(data=dict(backbone_output))

        if "first_user_end" not in action_input:
            return BatchFeature(
                data={
                    "backbone_features": backbone_output.backbone_features.detach(),
                    "backbone_attention_mask": backbone_output.backbone_attention_mask,
                    "image_mask": backbone_output.image_mask,
                }
            )

        boundaries = action_input.first_user_end.long()
        max_length = int(boundaries.max())

        def pack_prefix(tensor: torch.Tensor) -> torch.Tensor:
            if torch.is_floating_point(tensor):
                tensor = tensor.detach()
            output = tensor.new_zeros((tensor.shape[0], max_length, *tensor.shape[2:]))
            for row, boundary in enumerate(boundaries.tolist()):
                output[row, max_length - boundary :] = tensor[row, :boundary]
            return output

        return BatchFeature(
            data={
                "backbone_features": pack_prefix(backbone_output.backbone_features),
                "backbone_attention_mask": pack_prefix(backbone_output.backbone_attention_mask),
                "image_mask": pack_prefix(backbone_output.image_mask),
            }
        )

    def _visual_outputs(
        self,
        model_output: torch.Tensor,
        num_state_tokens: int,
        backbone_output: BatchFeature,
    ) -> dict:
        # Pool every query after the DiT's final normalization/time modulation
        # and output projection, then predict in the visual merger's feature space.
        query_end = num_state_tokens + self.config.num_learnable_queries
        evp = model_output[:, num_state_tokens:query_end].mean(dim=1)
        ep = self.query_head(evp).float()
        ev = backbone_output.third_view_visual_target.detach().float()
        valid = backbone_output.third_view_visual_valid_mask.bool()
        cosine_per_sample = 1 - F.cosine_similarity(ep, ev, dim=-1, eps=1e-8)
        smooth_l1_per_sample = F.smooth_l1_loss(ep, ev, beta=1.0, reduction="none").mean(dim=-1)
        count = valid.sum()
        # State dropout does not affect eligibility. An empty target batch keeps
        # a connected zero loss, including the prediction MLP, for backward/DDP.
        cosine_loss = cosine_per_sample.masked_fill(~valid, 0.0).sum() / count.clamp_min(1)
        smooth_l1_loss = smooth_l1_per_sample.masked_fill(~valid, 0.0).sum() / count.clamp_min(1)
        return {
            "visual_cosine_loss": cosine_loss,
            "visual_smooth_l1_loss": smooth_l1_loss,
            "visual_loss": cosine_loss + smooth_l1_loss,
            "visual_valid_samples": count,
        }

    def _third_outputs(
        self, model_output: torch.Tensor, num_state_tokens: int, backbone_output: BatchFeature
    ) -> dict:
        if "third_prediction_target" not in backbone_output:
            raise ValueError("DiT third supervision requires third-view teacher targets")
        query_states = model_output[:, num_state_tokens : num_state_tokens + THIRD_QUERY_COUNT]
        prediction = self.third_query_head(query_states).float()
        target = backbone_output.third_prediction_target.detach().float()
        if prediction.shape != target.shape:
            raise ValueError("Third query predictions and targets must have shape [B, 8, D]")
        smooth_l1_loss = F.smooth_l1_loss(prediction, target, beta=1.0)
        cosine = F.cosine_similarity(prediction, target, dim=-1, eps=1e-8).mean()
        cosine_loss = 1 - cosine
        metrics = third_prediction_metrics(
            prediction,
            target,
            backbone_output.third_second_target.detach(),
            backbone_output.third_teacher_target.detach(),
            self.config.third_prediction_target,
        )
        metrics.update(
            {
                "third_loss": smooth_l1_loss + 0.1 * cosine_loss,
                "third_smooth_l1_loss": smooth_l1_loss,
                "third_cosine_loss": cosine_loss,
                "third_valid_groups": target.new_tensor(target.shape[0] * THIRD_QUERY_COUNT),
            }
        )
        return metrics

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        full_backbone_output = backbone_output
        backbone_output = self._action_backbone_output(backbone_output, action_input)
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Learned queries receive no action position embedding.
        if self.config.third_loss and self.config.third_framework == "queries":
            query_features = self.third_queries.unsqueeze(0).expand(actions.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, query_features, action_features), dim=1)
        elif self.config.use_third_view_aux_loss:
            query_features = self.learnable_queries.unsqueeze(0).expand(actions.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, query_features, action_features), dim=1)
        elif self.config.use_phase_aux_loss:
            query_features = self.phase_queries.unsqueeze(0).expand(actions.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, query_features, action_features), dim=1)
        else:
            sa_embs = torch.cat((state_features, action_features), dim=1)
        self_attention_mask = self._dit_self_attention_mask(sa_embs, state_features.shape[1])
        vl_attn_mask = backbone_output.backbone_attention_mask
        dit_kwargs = {}
        if self.config.use_phase_aux_loss:
            dit_kwargs["unmodulated_prefix_length"] = state_features.shape[1] + 4
        if self.config.use_action_query_gate:
            dit_kwargs = dict(
                query_token_range=(
                    state_features.shape[1],
                    state_features.shape[1] + self.config.num_learnable_queries,
                ),
                return_gate_stats=True,
            )

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_result = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                self_attention_mask=self_attention_mask,
                **dit_kwargs,
            )
        else:
            model_result = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                self_attention_mask=self_attention_mask,
                **dit_kwargs,
            )

        model_output = model_result[0]
        gate_stats = model_result[2] if self.config.use_action_query_gate else {}
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        fm_loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        visual_outputs = {}
        if self.config.use_third_view_aux_loss:
            visual_outputs = self._visual_outputs(
                model_output, state_features.shape[1], full_backbone_output
            )
            loss = (
                self.config.fm_loss_weight * fm_loss
                + self.config.visual_loss_weight * visual_outputs["visual_loss"]
            )
        else:
            loss = fm_loss

        third_outputs = {}
        if self.config.third_loss and self.config.third_framework == "queries":
            third_outputs = self._third_outputs(
                model_output, state_features.shape[1], full_backbone_output
            )

        phase_outputs = {}
        if self.config.use_phase_aux_loss:
            # Last complete block, before the shared action output normalization/projection.
            query_start = state_features.shape[1]
            query_states = model_result[1][-1][:, query_start : query_start + 4]
            phase_outputs = self._phase_outputs(query_states, action_input)
            loss = (
                self.config.fm_loss_weight * fm_loss
                + self.config.phase_loss_weight * phase_outputs["phase_loss"]
            )

        output = {
            "loss": loss,
            "fm_loss": fm_loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
        output.update(visual_outputs)
        output.update(gate_stats)
        output.update(phase_outputs)
        output.update(third_outputs)
        return output

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self._action_backbone_output(backbone_output, action_input)
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            required = ("action_horizon", "rtc_overlap_steps", "rtc_frozen_steps", "rtc_ramp_rate")
            if options is None or any(key not in options for key in required):
                raise ValueError(f"RTC requires sampling options {required}")
            horizon, overlap, frozen = (options[key] for key in required[:3])
            ramp_rate = options["rtc_ramp_rate"]
            if any(type(value) is not int for value in (horizon, overlap, frozen)):
                raise ValueError("RTC horizons and steps must be integers")
            if not 0 <= frozen <= overlap <= horizon <= actions.shape[1] or horizon <= 0:
                raise ValueError(
                    "RTC requires 0 <= frozen <= overlap <= effective horizon <= model horizon"
                )
            previous = action_input["action"]
            if (
                previous.ndim != 3
                or previous.shape[0] != batch_size
                or previous.shape[1] < horizon
                or previous.shape[2] != self.action_dim
            ):
                raise ValueError(
                    "RTC action shape must match batch, padded action dimension, and effective horizon"
                )
            if not torch.isfinite(previous).all():
                raise ValueError("RTC actions must be finite")
            if (
                not isinstance(ramp_rate, (int, float))
                or not math.isfinite(ramp_rate)
                or ramp_rate <= 0
            ):
                raise ValueError("RTC ramp rate must be finite and positive")

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            if self.config.third_loss and self.config.third_framework == "queries":
                query_features = self.third_queries.unsqueeze(0).expand(batch_size, -1, -1)
                sa_embs = torch.cat((state_features, query_features, action_features), dim=1)
            elif self.config.use_third_view_aux_loss:
                query_features = self.learnable_queries.unsqueeze(0).expand(batch_size, -1, -1)
                sa_embs = torch.cat((state_features, query_features, action_features), dim=1)
            elif self.config.use_phase_aux_loss:
                query_features = self.phase_queries.unsqueeze(0).expand(batch_size, -1, -1)
                sa_embs = torch.cat((state_features, query_features, action_features), dim=1)
            else:
                sa_embs = torch.cat((state_features, action_features), dim=1)
            self_attention_mask = self._dit_self_attention_mask(sa_embs, state_features.shape[1])

            # Run model forward.
            dit_kwargs = {}
            if self.config.use_phase_aux_loss:
                dit_kwargs["unmodulated_prefix_length"] = state_features.shape[1] + 4
            if self.config.use_action_query_gate:
                dit_kwargs["query_token_range"] = (
                    state_features.shape[1],
                    state_features.shape[1] + self.config.num_learnable_queries,
                )
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                    self_attention_mask=self_attention_mask,
                    **dit_kwargs,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    self_attention_mask=self_attention_mask,
                    **dit_kwargs,
                )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    model_name = config.model_name
    # Fine-tuning may resolve a cached/gated Cosmos model ID to its local
    # snapshot directory.  The directory has the same Qwen3-VL architecture
    # even though its path no longer contains the original Hub model ID.
    if (
        "nvidia/Cosmos-Reason2" in model_name
        or "Qwen/Qwen3-VL" in model_name
        or Path(model_name).is_dir()
    ):
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Validate the architecture/target and initialize a complete new third branch."""
        if any(key.startswith("third_delta_") for key in kwargs):
            raise ValueError("Legacy third_delta_* options are not supported; use third_* options")
        config_kwargs = {
            key: kwargs[key]
            for key in (
                "cache_dir",
                "force_download",
                "local_files_only",
                "token",
                "revision",
                "subfolder",
            )
            if key in kwargs
        }
        checkpoint_config, _ = cls.config_class.get_config_dict(
            pretrained_model_name_or_path, **config_kwargs
        )
        requested_config = kwargs.get("config")
        validate_third_checkpoint_config(
            checkpoint_config,
            kwargs.get("third_framework", getattr(requested_config, "third_framework", None)),
            kwargs.get("third_loss", getattr(requested_config, "third_loss", None)),
            kwargs.get(
                "third_prediction_target",
                getattr(requested_config, "third_prediction_target", None),
            ),
            kwargs.get("third_target", getattr(requested_config, "third_target", None)),
        )
        return_loading_info = kwargs.pop("output_loading_info", False)
        model, loading_info = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, output_loading_info=True, **kwargs
        )
        validate_third_checkpoint_config(
            checkpoint_config,
            model.config.third_framework,
            model.config.third_loss,
            model.config.third_prediction_target,
            model.config.third_target,
        )
        all_prefixes = tuple(p for prefixes in THIRD_PARAMETER_PREFIXES.values() for p in prefixes)
        incompatible = [
            name
            for name in loading_info.get("unexpected_keys", [])
            if name.startswith(all_prefixes + LEGACY_THIRD_PARAMETER_PREFIXES)
        ]
        if incompatible:
            raise RuntimeError(
                f"Checkpoint contains incompatible, mixed or legacy third parameters: {incompatible}"
            )
        if model.config.third_loss:
            prefixes = THIRD_PARAMETER_PREFIXES[model.config.third_framework]
            third_keys = {name for name, _ in model.named_parameters() if name.startswith(prefixes)}
            missing = set(loading_info.get("missing_keys", [])) & third_keys
            mismatched = [
                item
                for item in loading_info.get("mismatched_keys", [])
                if (item if isinstance(item, str) else item[0]) in third_keys
            ]
            if mismatched or (
                missing and (missing != third_keys or checkpoint_config.get("third_loss"))
            ):
                raise RuntimeError(
                    "Third checkpoint has partially missing or mismatched parameters"
                )
            if missing:
                third_module = (
                    model.backbone
                    if model.config.third_framework in ("mlp", "transformer")
                    else model.action_head
                )
                third_module.initialize_third_parameters()
                logger.info("Third branch absent from base checkpoint; fully initialized")
                # These parameters were explicitly initialized; pipeline strict checks
                # must still reject every other missing or mismatched checkpoint key.
                loading_info["missing_keys"] = [
                    key for key in loading_info["missing_keys"] if key not in third_keys
                ]
        return (model, loading_info) if return_loading_info else model

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            use_third_view_aux_loss=config.use_third_view_aux_loss,
            third_loss=config.third_loss,
            third_framework=config.third_framework,
            third_target=config.third_target,
            third_prediction_target=config.third_prediction_target,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
            use_third_view_aux_loss=config.use_third_view_aux_loss,
            third_loss=config.third_loss,
            third_framework=config.third_framework,
            third_prediction_target=config.third_prediction_target,
            num_learnable_queries=config.num_learnable_queries,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if not isinstance(x, torch.Tensor):
                return x
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)
        if self.config.third_loss:
            third_outputs = (
                backbone_outputs
                if self.config.third_framework in ("mlp", "transformer")
                else action_outputs
            )
            if "third_loss" not in third_outputs:
                raise ValueError(
                    "Third-view training requires teacher targets and an auxiliary loss"
                )
            action_outputs["loss"] = (
                self.config.fm_loss_weight * action_outputs["fm_loss"]
                + self.config.third_loss_weight * third_outputs["third_loss"]
            )
            if self.config.third_framework in ("mlp", "transformer"):
                action_outputs.update(
                    {k: v for k, v in third_outputs.items() if k.startswith("third_")}
                )
        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
