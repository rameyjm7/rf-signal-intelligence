import os
from pathlib import Path

import pytest

_POLICY_ENV = "INTEGRATION_ARTIFACT_POLICY"
_SKIP_IF_MISSING = "skip_if_missing"
_REQUIRE = "require"


def require_paths(label: str, paths: list[Path]) -> None:
    """Skip or fail integration tests based on missing artifacts and policy."""
    missing = [str(path) for path in paths if not path.exists()]
    if not missing:
        return

    policy = os.getenv(_POLICY_ENV, _SKIP_IF_MISSING).strip().lower()
    message = (
        f"Missing {label} artifacts: {', '.join(missing)}. "
        f"Set {_POLICY_ENV}=require to fail instead of skip."
    )

    if policy == _REQUIRE:
        pytest.fail(message)

    pytest.skip(message)
