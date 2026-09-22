"""Register the Locomanip MY_G1 data contract with Isaac-GR00T N1.7."""

from __future__ import annotations

import json
from pathlib import Path

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

CONFIG_PATH = Path(__file__).with_name("my_g1_features.json")
DATA_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
TRAINING = DATA_CONFIG["training_modalities"]

MY_G1_CONFIG = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=TRAINING["video"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=TRAINING["state"]),
    "action": ModalityConfig(
        delta_indices=list(range(DATA_CONFIG["action_horizon"])),
        modality_keys=TRAINING["action"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation[TRAINING["action_representation"]],
                type=ActionType[TRAINING["action_type"]],
                format=ActionFormat[TRAINING["action_format"]],
            )
            for _ in TRAINING["action"]
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=TRAINING["language"]),
}

register_modality_config(MY_G1_CONFIG, embodiment_tag=EmbodimentTag.MY_G1)
