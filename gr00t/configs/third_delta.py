"""Architecture and checkpoint metadata for third-view supervision."""

THIRD_VERSIONS = {"mlp": 1, "queries": 2, "transformer": 3}
THIRD_PARAMETER_PREFIXES = {
    "mlp": ("backbone.third_head.",),
    "queries": ("action_head.third_queries", "action_head.third_query_head."),
    "transformer": ("backbone.third_head.",),
}
LEGACY_THIRD_PARAMETER_PREFIXES = (
    "backbone.third_delta_head.",
    "action_head.delta_queries",
    "action_head.delta_query_head.",
)


def reject_legacy_third_config(config: dict) -> None:
    legacy = [key for key in config if key.startswith("third_delta_")]
    if legacy:
        raise ValueError(
            f"Legacy third_delta_* configuration is unsupported: {legacy}. "
            "Start a new run from a base model with the third_* interface."
        )


def third_architecture_version(framework: str) -> int:
    if framework not in THIRD_VERSIONS:
        raise ValueError("third_framework must be mlp, queries or transformer")
    return THIRD_VERSIONS[framework]


def checkpoint_third_framework(config: dict) -> str | None:
    reject_legacy_third_config(config)
    if not config.get("third_loss", False):
        return None
    framework = config.get("third_framework", "mlp")
    version = third_architecture_version(framework)
    if config.get("third_version", version) != version:
        raise ValueError("Third checkpoint framework and architecture version disagree")
    if config.get("third_prediction_target", "third") not in ("third", "delta"):
        raise ValueError("third_prediction_target must be third or delta")
    if config.get("third_target", "visual_encoder") != "visual_encoder":
        raise ValueError("third_target supports visual_encoder only")
    return framework


def validate_third_checkpoint_config(
    config: dict,
    framework: str | None = None,
    enabled: bool | None = None,
    prediction_target: str | None = None,
    target: str | None = None,
) -> str | None:
    saved = checkpoint_third_framework(config)
    if framework is not None:
        third_architecture_version(framework)
    if prediction_target is not None and prediction_target not in ("third", "delta"):
        raise ValueError("third_prediction_target must be third or delta")
    if target is not None and target != "visual_encoder":
        raise ValueError("third_target supports visual_encoder only")
    if saved is not None and (
        enabled is False
        or (framework is not None and framework != saved)
        or (
            prediction_target is not None
            and prediction_target != config.get("third_prediction_target", "third")
        )
        or (target is not None and target != config.get("third_target", "visual_encoder"))
    ):
        raise RuntimeError(
            "Checkpoint third supervision architecture/target differs from requested settings. "
            "Use the saved settings or start a new run from a base model."
        )
    return saved


def normalize_third_config(
    config: dict,
    framework: str | None = None,
    enabled: bool | None = None,
    prediction_target: str | None = None,
    target: str | None = None,
) -> dict:
    saved = validate_third_checkpoint_config(config, framework, enabled, prediction_target, target)
    normalized = dict(config)
    framework = framework or saved or config.get("third_framework", "mlp")
    normalized["third_framework"] = framework
    normalized["third_version"] = third_architecture_version(framework)
    normalized["third_prediction_target"] = prediction_target or config.get(
        "third_prediction_target", "third"
    )
    normalized["third_target"] = target or config.get("third_target", "visual_encoder")
    if enabled is not None:
        normalized["third_loss"] = enabled
    return normalized
