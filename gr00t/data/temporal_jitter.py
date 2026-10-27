"""Configuration validation for training image frame offsets."""


def validate_jiter(jiter: tuple[int, int] | None) -> tuple[int, int] | None:
    """Validate bounds, accepting lists when loading a saved YAML configuration."""
    if jiter is None:
        return None
    if (
        not isinstance(jiter, (tuple, list))
        or len(jiter) != 2
        or any(type(bound) is not int for bound in jiter)
    ):
        raise ValueError("jiter requires exactly two integer frame offsets: a b")
    a, b = jiter
    if a >= b:
        raise ValueError("jiter requires a < b")
    return (a, b)
