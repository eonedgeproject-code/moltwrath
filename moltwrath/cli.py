"""MoltWrath CLI — command line interface."""

import click
import uvicorn

from moltwrath.utils.config import get_settings


@click.group()
def main():
    """🧠 MoltWrath — AI Agent Orchestration Framework"""
    pass


@main.command()
@click.option("--host", default=None, help="API host")
@click.option("--port", default=None, type=int, help="API port")
@click.option("--reload", is_flag=True, help="Enable hot reload")
def serve(host: str | None, port: int | None, reload: bool):
    """Start the MoltWrath API server."""
    settings = get_settings()
    uvicorn.run(
        "moltwrath.api.app:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@main.command()
def dashboard():
    """Open the MoltWrath dashboard."""
    import webbrowser
    settings = get_settings()
    url = f"http://localhost:{settings.api_port}"
    click.echo(f"🧠 Opening dashboard at {url}")
    webbrowser.open(url)


@main.command()
def info():
    """Show framework info."""
    click.echo("🧠 MoltWrath v0.1.0")
    click.echo("   AI Agent Orchestration Framework")
    click.echo("   Python + FastAPI + Multi-Agent Swarms")


if __name__ == "__main__":
    main()
