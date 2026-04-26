"""Rich-based CLI UI helpers."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from pydantic import BaseModel, Field
from rich.console import Console
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
    duration_seconds: float = Field(ge=0)
    response: str


class ModelTestingProgress:
    """Live progress controller for per-model testing."""

    def __init__(self, progress: Progress, task_id: int) -> None:
        self._progress = progress
        self._task_id = task_id

    def update_current_model(self, model_id: str) -> None:
        """Set the model currently being tested."""
        self._progress.update(self._task_id, model_id=model_id)

    def advance(self) -> None:
        """Mark one model test as completed."""
        self._progress.advance(self._task_id)


class CliConsole:
    """Presentation layer for interactive CLI output."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def show_feature_header(
        self,
        feature_name: str,
        description: str,
        endpoint_url: str,
        test_prompt: str,
    ) -> None:
        """Show feature title and endpoint context."""
        body = Table.grid(padding=(0, 1))
        body.add_column(style="bold cyan", no_wrap=True)
        body.add_column(style="white")
        body.add_row("Name", feature_name)
        body.add_row("Description", description)
        body.add_row("Endpoint", f"[cyan]{endpoint_url}[/cyan]")
        body.add_row("Test prompt", f"[italic]\"{test_prompt}\"[/italic]")

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

    @contextmanager
    def test_models_progress(self, model_count: int) -> Iterator[ModelTestingProgress]:
        """Render live per-model testing progress."""
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
            total=model_count,
            model_id="",
        )

        with progress:
            yield ModelTestingProgress(progress=progress, task_id=task_id)

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
            padding=(1, 1),
        )
        table.add_column("Model", style="white", no_wrap=True)
        table.add_column("Status")
        table.add_column("Duration", style="cyan", justify="right")
        table.add_column("Response", style="white")

        for row in rows:
            status_style = self._get_status_style(row.status)
            response_style = "red" if row.status == "failed" else "white"
            table.add_row(
                row.model_name,
                Text(row.status, style=status_style),
                f"{row.duration_seconds:.2f}s",
                Text(row.response, style=response_style),
            )

        self._console.print(table)

    def _get_status_style(self, status: str) -> str:
        if status == "ok":
            return "green"

        if status == "failed":
            return "red"

        return "yellow"
