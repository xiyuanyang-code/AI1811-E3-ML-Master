"""
Real-time Control Panel for Monitoring ML Agent Execution

Displays live status of Envisioner and Executor instances
"""

import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TaskID, SpinnerColumn, TextColumn
from datetime import datetime


@dataclass
class ExecutorState:
    """State of a single executor"""
    executor_id: str
    node_id: str
    status: str  # "idle", "running", "completed", "failed"
    current_task: str = ""
    start_time: float = 0.0
    progress: float = 0.0


@dataclass
class EnvisionerState:
    """State of the Envisioner"""
    current_phase: str = "idle"
    current_step: int = 0
    total_steps: int = 0
    iteration: int = 0
    budget: int = 0
    selected_node: str = ""
    expanding_node: str = ""
    best_metric: Optional[float] = None
    tree_size: int = 0
    stats: Dict[str, int] = field(default_factory=dict)


class ControlPanel:
    """
    Real-time control panel for monitoring ML agent execution

    Features:
    - Live display of Envisioner phase and progress
    - Active executor count and status
    - Current metrics and best results
    - Real-time log streaming
    """

    def __init__(self):
        self.console = Console()
        self.envisioner_state = EnvisionerState()
        self.executors: Dict[str, ExecutorState] = {}
        self.logs: list = []  # Recent logs
        self.max_logs = 10
        self.start_time = time.time()
        self.is_running = False
        self.display_thread: Optional[threading.Thread] = None

        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=20),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            expand=False,
        )
        self.overall_task: Optional[TaskID] = None

    def start(self):
        """Start the live display"""
        self.is_running = True
        self.overall_task = self.progress.add_task("Overall Progress", total=100)

        # Start display thread
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()

    def stop(self):
        """Stop the live display"""
        self.is_running = False
        if self.display_thread:
            self.display_thread.join(timeout=2)

    def update_envisioner(
        self,
        phase: str = "",
        step: int = 0,
        total_steps: int = 0,
        iteration: int = 0,
        budget: int = 0,
        selected_node: str = "",
        expanding_node: str = "",
        best_metric: Optional[float] = None,
        tree_size: int = 0,
        **stats
    ):
        """Update Envisioner state"""
        if phase:
            self.envisioner_state.current_phase = phase
        self.envisioner_state.current_step = step
        self.envisioner_state.total_steps = total_steps
        self.envisioner_state.iteration = iteration
        self.envisioner_state.budget = budget
        if selected_node:
            self.envisioner_state.selected_node = selected_node
        if expanding_node:
            self.envisioner_state.expanding_node = expanding_node
        if best_metric is not None:
            self.envisioner_state.best_metric = best_metric
        self.envisioner_state.tree_size = tree_size
        self.envisioner_state.stats.update(stats)

        # Update progress
        if total_steps > 0:
            progress_pct = int((step / total_steps) * 100)
            self.progress.update(self.overall_task, completed=progress_pct)

    def add_executor(
        self,
        executor_id: str,
        node_id: str,
        status: str = "idle",
        task: str = ""
    ):
        """Add or update an executor"""
        if executor_id not in self.executors:
            self.executors[executor_id] = ExecutorState(
                executor_id=executor_id,
                node_id=node_id,
                status=status,
                current_task=task,
                start_time=time.time()
            )
        else:
            exec_state = self.executors[executor_id]
            exec_state.node_id = node_id
            exec_state.status = status
            if task:
                exec_state.current_task = task
            if status == "running":
                exec_state.start_time = time.time()

    def remove_executor(self, executor_id: str):
        """Remove an executor"""
        if executor_id in self.executors:
            del self.executors[executor_id]

    def add_log(self, message: str, level: str = "INFO"):
        """Add a log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] [{level}] {message}")
        if len(self.logs) > self.max_logs:
            self.logs.pop(0)

    def _build_layout(self) -> Layout:
        """Build the dashboard layout"""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="logs", size=15),
        )

        layout["body"].split_row(
            Layout(name="envisioner", ratio=2),
            Layout(name="executors", ratio=1),
        )

        return layout

    def _build_header(self) -> Panel:
        """Build header panel"""
        elapsed = time.time() - self.start_time
        elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

        header_text = Text()
        header_text.append("E³-ML-Master Control Panel", style="bold cyan")
        header_text.append(f"  |  Elapsed: {elapsed_str}", style="dim")
        header_text.append(f"  |  Active Executors: {len([e for e in self.executors.values() if e.status == 'running'])}",
                          style="bold yellow")

        return Panel(header_text, style="on blue")

    def _build_envisioner_panel(self) -> Panel:
        """Build Envisioner status panel"""
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Property", style="cyan", width=20)
        table.add_column("Value", style="white")

        # Phase info
        phase_color = {
            "idle": "dim",
            "SELECTION": "blue",
            "EXPANSION": "yellow",
            "SIMULATION": "green",
            "BACKPROPAGATION": "magenta"
        }.get(self.envisioner_state.current_phase.upper(), "white")

        table.add_row("Current Phase", f"[{phase_color}]{self.envisioner_state.current_phase}[/{phase_color}]")
        table.add_row("Step", f"{self.envisioner_state.current_step}/{self.envisioner_state.total_steps}")
        table.add_row("Iteration", f"{self.envisioner_state.iteration}/{self.envisioner_state.budget}")
        table.add_row("Tree Size", str(self.envisioner_state.tree_size))

        # Node info
        if self.envisioner_state.selected_node:
            table.add_row("Selected Node", f"[cyan]{self.envisioner_state.selected_node}[/cyan]")
        if self.envisioner_state.expanding_node:
            table.add_row("Expanding Node", f"[yellow]{self.envisioner_state.expanding_node}[/yellow]")

        # Metric
        if self.envisioner_state.best_metric is not None:
            table.add_row("Best Metric", f"[bold green]{self.envisioner_state.best_metric:.6f}[/bold green]")

        # Stats
        if self.envisioner_state.stats:
            stats_str = ", ".join([f"{k}: {v}" for k, v in self.envisioner_state.stats.items()])
            table.add_row("Statistics", stats_str)

        # Progress
        progress_panel = Panel(
            self._render_progress(),
            title="[bold]Overall Progress[/bold]",
            border_style="cyan"
        )

        # Combine table and progress
        layout = Layout()
        layout.split_column(
            Layout(Panel(table, title="[bold]Envisioner Status[/bold]", border_style="magenta"), ratio=2),
            Layout(progress_panel, ratio=1),
        )

        return layout

    def _build_executors_panel(self) -> Panel:
        """Build Executors status panel"""
        table = Table(show_header=True, header_style="bold yellow", box=None)
        table.add_column("Executor", style="cyan", width=8)
        table.add_column("Status", style="white", width=10)
        table.add_column("Node", style="white", width=8)
        table.add_column("Task", style="dim", width=20)

        if not self.executors:
            table.add_row("[dim]No active executors[/dim]", "", "", "")
        else:
            for executor in self.executors.values():
                status_color = {
                    "idle": "dim",
                    "running": "green",
                    "completed": "blue",
                    "failed": "red"
                }.get(executor.status, "white")

                status_symbol = {
                    "idle": "○",
                    "running": "⟳",
                    "completed": "✓",
                    "failed": "✗"
                }.get(executor.status, "?")

                # Calculate elapsed time for running executors
                elapsed = ""
                if executor.status == "running":
                    elapsed_sec = int(time.time() - executor.start_time)
                    elapsed = f" ({elapsed_sec}s)"

                table.add_row(
                    f"[{status_color}]{status_symbol} {executor.executor_id[:6]}[/{status_color}]",
                    f"[{status_color}]{executor.status}{elapsed}[/{status_color}]",
                    executor.node_id[:8] if executor.node_id else "",
                    executor.current_task[:18] if executor.current_task else ""
                )

        return Panel(table, title="[bold]Executor Status[/bold]", border_style="yellow")

    def _build_logs_panel(self) -> Panel:
        """Build logs panel"""
        log_text = Text()
        for log in self.logs:
            log_text.append(log + "\n", style="dim")

        if not self.logs:
            log_text.append("[No logs yet]", style="dim")

        return Panel(log_text, title="[bold]Recent Logs[/bold]", border_style="dim", height=13)

    def _render_progress(self) -> Layout:
        """Render progress bar"""
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            expand=True,
        )

        # Copy the task state from main progress
        if self.overall_task is not None:
            task_info = self.progress._tasks[self.overall_task]
            progress.add_task(
                "MCTS Steps",
                total=task_info.total,
                completed=task_info.completed,
            )

        return progress

    def _display_loop(self):
        """Main display loop (runs in separate thread)"""
        with Live(console=self.console, refresh_per_second=2) as live:
            while self.is_running:
                layout = self._build_layout()

                layout["header"].update(self._build_header())
                layout["envisioner"].update(self._build_envisioner_panel())
                layout["executors"].update(self._build_executors_panel())
                layout["logs"].update(self._build_logs_panel())

                live.update(layout)
                time.sleep(0.5)


# Global control panel instance
_control_panel: Optional[ControlPanel] = None


def get_control_panel() -> ControlPanel:
    """Get or create the global control panel instance"""
    global _control_panel
    if _control_panel is None:
        _control_panel = ControlPanel()
    return _control_panel


def start_control_panel():
    """Start the global control panel"""
    panel = get_control_panel()
    panel.start()
    return panel


def stop_control_panel():
    """Stop the global control panel"""
    global _control_panel
    if _control_panel:
        _control_panel.stop()
        _control_panel = None
