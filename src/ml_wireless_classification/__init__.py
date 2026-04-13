"""ML Wireless Signal Classification package."""

def create_app() -> None:
    """Compatibility shim for legacy console entry points."""
    from ml_wireless_classification.__main__ import main as _main
    _main()


def main() -> None:
    """CLI entry point exposed for console_scripts."""
    from ml_wireless_classification.__main__ import main as _main
    _main()


__all__ = ["main", "create_app"]
