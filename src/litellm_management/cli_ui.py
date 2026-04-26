"""Rich-based CLI UI helpers."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from time import monotonic, sleep

from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text


class FeatureResultSummary(BaseModel):
    """Summary displayed after a feature finishes."""

    available_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)


class ModelTestResultRow(BaseModel):
    """One row in the per-model test result table."""

    model_name: str = Field(min_length=1)
    status: str = Field(min_length=1)
    response: str


class CliConsole:
    """Presentation layer for interactive CLI output."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def show_feature_header(self, feature_name: str, endpoint_url: str) -> None:
        """Show feature title and endpoint context."""
        body = Text()
        body.append(feature_name, style="bold white")
        body.append("\n")
        body.append(endpoint_url, style="cyan")

        self._console.print(
            Panel(
                body,
                title="[bold cyan]LiteLLM Management[/bold cyan]",
                border_style="cyan",
            )
        )

    @contextmanager
    def status(self, message: str) -> Iterator[None]:
        """Show a spinner while a blocking operation runs."""
        with self._console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
            yield

    def show_success(self, message: str) -> None:
        """Show a concise success line."""
        self._console.print(f"[green]✓[/green] {message}")

    def show_error(self, message: str) -> None:
        """Show a plain error line."""
        self._console.print(f"[red]Error:[/red] {message}")

    def show_empty(self, message: str) -> None:
        """Show a neutral empty-state line."""
        self._console.print(f"[yellow]{message}[/yellow]")

    def test_models_progress(self, model_ids: list[str]) -> None:
        """Render simulated per-model testing progress."""
        progress = Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[cyan]Testing models[/cyan]"),
            TextColumn("[bold]{task.completed}/{task.total}[/bold]"),
            BarColumn(bar_width=None),
            TextColumn("[white]{task.fields[model_id]}[/white]"),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        task_id = progress.add_task(
            "Testing models",
            total=len(model_ids),
            model_id="",
        )

        with Live(progress, console=self._console, refresh_per_second=12):
            for model_id in model_ids:
                start_time = monotonic()
                progress.update(task_id, model_id=model_id)

                while monotonic() - start_time < 1:
                    progress.refresh()
                    sleep(0.05)

                progress.advance(task_id)

    def show_results(self, summary: FeatureResultSummary) -> None:
        """Show final result summary."""
        table = Table.grid(padding=(0, 1))
        table.add_column(justify="left")
        table.add_column(justify="right")
        table.add_row("[green]✓[/green]", f"{summary.available_count} available")
        table.add_row("[red]✗[/red]", f"{summary.failed_count} failed")
        table.add_row("[cyan]⏱[/cyan]", f"Duration: {summary.duration_seconds:.1f}s")

        self._console.print(
            Panel(
                table,
                title="[bold cyan]Results[/bold cyan]",
                border_style="cyan",
            )
        )

    def show_model_results_table(self, rows: Sequence[ModelTestResultRow]) -> None:
        """Show per-model test results."""
        table = Table(
            title="Model results",
            header_style="bold cyan",
            border_style="cyan",
        )
        table.add_column("Model", style="white", no_wrap=True)
        table.add_column("Status", style="yellow")
        table.add_column("Response", style="white")

        for row in rows:
            table.add_row(row.model_name, row.status, row.response)

        self._console.print(table)
