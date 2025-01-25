"""
Supports `python -m ...` invocation of the CLI app as an alternative to invoking the command directly.
"""

from .cli import app

if __name__ == "__main__":
    app()
