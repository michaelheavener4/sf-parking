"""Compatibility entrypoint for the authoritative V4 tournament."""
try:
    from scripts.train_event_driven_v4 import main
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    from train_event_driven_v4 import main

if __name__ == "__main__":
    raise SystemExit(main())
