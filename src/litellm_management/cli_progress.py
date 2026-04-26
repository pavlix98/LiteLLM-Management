"""CLI progress rendering helpers."""

import sys
from typing import TextIO


class SpinnerProgress:
    """Renders a single-line spinner progress indicator."""

    _frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, output: TextIO = sys.stdout) -> None:
        self._output = output
        self._frame_index = 0
        self._last_line_length = 0

    def render(self, message: str) -> None:
        """Render progress on the current terminal line."""
        frame = self._frames[self._frame_index]
        self._frame_index = (self._frame_index + 1) % len(self._frames)

        line = f"{frame} {message}"
        padding = " " * max(0, self._last_line_length - len(line))
        self._output.write(f"\r{line}{padding}")
        self._output.flush()
        self._last_line_length = len(line)

    def finish(self, message: str) -> None:
        """Render a final message and end the progress line."""
        padding = " " * max(0, self._last_line_length - len(message))
        self._output.write(f"\r{message}{padding}\n")
        self._output.flush()
        self._last_line_length = 0
