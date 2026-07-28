"""Command-line interface for the UX analyzer."""

import typer

from ux_analyzer import __version__

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """Run UX analyzer commands."""


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(f"uxa {__version__}")
