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

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from enum import Enum
import json
import math
from pathlib import Path

import torch
from transformers import PretrainedConfig

from gr00t.configs.third_delta import (
    normalize_third_config,
    reject_legacy_third_config,
    third_architecture_version,
)

from . import register_model_config


@dataclass
class Gr00tN1d7Config(PretrainedConfig):
    """Unified configuration for Gr00tN1d7 model with backbone and action head.

    Gr00tN1d7 uses the Cosmos-Reason2-2B (Qwen3-VL architecture) VLM backbone,
    replacing the Eagle backbone used in Gr00tN1d6.
    """

    # Model identification
    model_type: str = "Gr00tN1d7"
    model_dtype: str = "bfloat16"  # Use bfloat16 for Flash Attention compatibility

    # Backbone configuration
    model_name: str = "nvidia/Cosmos-Reason2-2B"
    backbone_model_type: str = "qwen"
    model_revision: str | None = None
    tune_top_llm_layers: int = 0  # Number of top LLM layers to tune
    backbone_embedding_dim: int = 2048  # project_to_dim; must match Cosmos-Reason2-2B hidden size
    tune_llm: bool = False
    tune_visual: bool = False
    select_layer: int = 12
    reproject_vision: bool = False
    use_flash_attention: bool = True
    load_bf16: bool = False  # Enable BF16 loading
    backbone_trainable_params_fp32: bool = True

    ### Processing parameters
    image_crop_size: tuple[int, int] | None = (230, 230)
    image_target_size: tuple[int, int] | None = (256, 256)

    shortest_image_edge: int | None = None
    crop_fraction: float | None = None

    random_rotation_angle: int | None = None
    color_jitter_params: dict[str, float] | None = None
    use_albumentations_transforms: bool = True
    letter_box_transform: bool = False
    # Extra augmentation config (mask-based and others).
    extra_augmentation_config: dict | None = None
    formalize_language: bool = True
    apply_sincos_state_encoding: bool = (
        False  # Global flag to enable per-embodiment sin/cos encoding
    )
    use_percentiles: bool = True
    use_relative_action: bool = False

    # Action head configuration parameters
    max_state_dim: int = 132  # Default from state_shape
    max_action_dim: int = 132  # Default from action_shape
    action_horizon: int = 40
    hidden_size: int = 1024
    input_embedding_dim: int = 1536

    # State history: number of consecutive state timesteps fed to the state encoder
    state_history_length: int = 1

    # Global parameters
    add_pos_embed: bool = True
    attn_dropout: float = 0.2
    use_vlln: bool = True
    max_seq_len: int = 1024
    use_alternate_vl_dit: bool = True  # True for AlternateVLDiT, False for DiT
    attend_text_every_n_blocks: int = 2

    diffusion_model_cfg: dict = field(
        default_factory=lambda: {
            "positional_embeddings": None,
            "num_layers": 16,
            "num_attention_heads": 32,
            "attention_head_dim": 48,
            "norm_type": "ada_norm",
            "dropout": 0.2,
            "final_dropout": True,
            "output_dim": 1024,
            "interleave_self_attention": True,
        }
    )

    # Flow matching parameters
    num_inference_timesteps: int = 4
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000

    # Training parameters
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    tune_vlln: bool = True

    # Select LLM-input feature prediction or eight final DiT supervision queries.
    third_loss: bool = False
    # Derived from the explicitly selected framework and saved with checkpoints.
    third_version: int = 1
    # OmegaConf structured configs require str; allowed values are validated below.
    third_framework: str = "mlp"
    third_prediction_target: str = "third"
    third_target: str = "visual_encoder"
    third_loss_weight: float = 0.5

    # Third-view representation auxiliary objective. This is intentionally
    # opt-in so models without auxiliary supervision keep their architecture.
    # Pool all final DiT queries and align the MLP prediction with the detached
    # mean third-view merger output (before the LLM), using cosine + SmoothL1.
    # Enabling this also applies block self-attention: state -> state,
    # queries -> state/queries, actions -> all tokens, in training and inference.
    use_third_view_aux_loss: bool = False
    use_action_query_gate: bool = False
    query_gate_init_prob: float = 0.5
    third_view_key: str = "third_view"
    num_learnable_queries: int = 8
    fm_loss_weight: float = 1.0
    visual_loss_weight: float = 0.1
    query_ffn_hidden_dim: int = 4096

    # Four time-independent context queries predict phases aligned with the action chunk.
    use_phase_aux_loss: bool = False
    phase_label_key: str = "phase"
    phase_num_classes: int = 3
    phase_mlp_hidden_dim: int = 1024
    phase_loss_weight: float = 1.0

    # State augmentation parameters
    state_dropout_prob: float = 0.8  # State dropout probability
    state_noise_std: float = 0.0  # Gaussian noise std on retained normalized state inputs
    hand_state_noise_std: float = 0.0  # Additional Gaussian noise on normalized hand states
    hand_state_dropout_prob: float = 0.0  # Additional probability of zeroing both hand inputs
    exclude_state: bool = False  # Zero out all state inputs (ablation)
    use_mean_std: bool = False  # Use mean/std normalization instead of min/max

    # Multi-embodiment parameters
    max_num_embodiments: int = 32

    def __init__(self, **kwargs):
        reject_legacy_third_config(kwargs)
        if "third_version" in kwargs and kwargs["third_version"] != third_architecture_version(
            kwargs.get("third_framework", "mlp")
        ):
            raise ValueError("Requested third version does not match selected framework")
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Ensures that all dataclass defaults (including those using default_factory)
        # are explicitly assigned to the instance, even if dataclasses initialization or subclassing
        # (PretrainedConfig) interferes with normal default injection.
        for f in self.__dataclass_fields__.values():
            if not hasattr(self, f.name):
                if f.default is not MISSING:
                    setattr(self, f.name, f.default)
                elif getattr(f, "default_factory", MISSING) is not MISSING:
                    setattr(self, f.name, f.default_factory())

        if not math.isfinite(self.state_noise_std) or self.state_noise_std < 0:
            raise ValueError(f"state_noise_std must be finite and >= 0, got {self.state_noise_std}")
        if not math.isfinite(self.hand_state_noise_std) or self.hand_state_noise_std < 0:
            raise ValueError("hand_state_noise_std must be finite and >= 0")
        if not math.isfinite(self.hand_state_dropout_prob) or not (
            0 <= self.hand_state_dropout_prob <= 1
        ):
            raise ValueError("hand_state_dropout_prob must be finite and in [0, 1]")
        self.validate_phase_config()

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        reject_legacy_third_config(kwargs)
        normalized = normalize_third_config(
            config_dict,
            kwargs.get("third_framework"),
            kwargs.get("third_loss"),
            kwargs.get("third_prediction_target"),
            kwargs.get("third_target"),
        )
        if "third_version" in kwargs and kwargs["third_version"] != normalized["third_version"]:
            raise ValueError("Requested delta version does not match the selected framework")
        result = super().from_dict(normalized, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config.validate_phase_config()
        return result

    def validate_phase_config(self) -> None:
        if self.third_loss and (
            self.use_phase_aux_loss or self.use_third_view_aux_loss or self.use_action_query_gate
        ):
            raise ValueError(
                "third_loss is incompatible with phase, third-view loss and query gates"
            )
        self.third_version = third_architecture_version(self.third_framework)
        if self.third_prediction_target not in ("third", "delta"):
            raise ValueError("third_prediction_target must be third or delta")
        if self.third_target != "visual_encoder":
            raise ValueError("third_target currently supports only visual_encoder")
        if not math.isfinite(self.third_loss_weight) or self.third_loss_weight < 0:
            raise ValueError("third_loss_weight must be finite and >= 0")
        if self.third_loss and self.third_framework == "queries":
            if self.num_learnable_queries != 8 or self.query_ffn_hidden_dim != 4096:
                raise ValueError(
                    "DiT third supervision requires 8 queries and MLP hidden width 4096"
                )
            if (
                not self.diffusion_model_cfg.get("interleave_self_attention", False)
                or self.diffusion_model_cfg.get("num_layers", 0) < 2
            ):
                raise ValueError("Delta queries require at least two interleaved attention blocks")
        if self.use_phase_aux_loss:
            if self.use_third_view_aux_loss or self.use_action_query_gate:
                raise ValueError(
                    "Phase supervision is incompatible with third-view loss and query gates"
                )
            if (
                not self.diffusion_model_cfg.get("interleave_self_attention", False)
                or self.diffusion_model_cfg.get("num_layers", 0) < 2
            ):
                raise ValueError("Phase queries require at least two interleaved attention blocks")
        if not isinstance(self.phase_num_classes, int) or self.phase_num_classes < 2:
            raise ValueError("phase_num_classes must be an integer >= 2")
        if not isinstance(self.phase_mlp_hidden_dim, int) or self.phase_mlp_hidden_dim < 1:
            raise ValueError("phase_mlp_hidden_dim must be a positive integer")
        if not self.phase_label_key:
            raise ValueError("phase_label_key must not be empty")
        if not math.isfinite(self.phase_loss_weight) or self.phase_loss_weight < 0:
            raise ValueError("phase_loss_weight must be finite and >= 0")

    def to_filtered_dict(self, exclude_augment: bool = True) -> dict:
        """Return a dictionary representation of this config, optionally excluding augmentation keys."""
        if is_dataclass(self):
            cfg = asdict(self)
        else:
            cfg = dict(self.__dict__)

        if exclude_augment:
            exclude_keys = {
                "random_rotation_angle",
                "color_jitter_params",
                "use_albumentations_transforms",
                "formalize_language",
                "image_crop_size",
                "image_target_size",
                "shortest_image_edge",
                "crop_fraction",
            }
            cfg = {k: v for k, v in cfg.items() if k not in exclude_keys}

        return cfg

    def to_filtered_json(self, exclude_augment: bool = True, **kwargs) -> str:
        """Return a JSON string of this config, optionally excluding augmentation keys."""

        def default(o):
            if isinstance(o, (Path, torch.dtype, torch.device)):
                return str(o)
            if isinstance(o, Enum):
                return o.value
            return str(o)

        return json.dumps(
            self.to_filtered_dict(exclude_augment),
            indent=2,
            default=default,
            **kwargs,
        )


register_model_config("Gr00tN1d7", Gr00tN1d7Config)
