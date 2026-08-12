import os
import sys


# Windkit reads these settings before falling back to interactive configuration.
os.environ.setdefault("WINDKIT_NAME", "test")
os.environ.setdefault("WINDKIT_EMAIL", "test@dtu.dk")
os.environ.setdefault("WINDKIT_INSTITUTION", "DTU Wind")


def _disable_interactive_matplotlib():
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")
    # Avoids a bunch warnings on the test run
    plt.show = lambda *args, **kwargs: None


def pytest_addoption(parser):
    parser.addoption(
        "--no-show",
        action="store_true",
        default=False,
        help="disable interactive Matplotlib windows during tests",
    )


def pytest_configure(config):
    if config.getoption("--no-show"):
        _disable_interactive_matplotlib()
