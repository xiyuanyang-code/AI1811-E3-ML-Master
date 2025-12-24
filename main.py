"""
Main entry point for Framework

A simplified MCTS-based ML agent with Envisioner and Executor.
"""

import logging
import time
import os
import sys
import atexit
import dotenv
import shutil
import pandas as pd
from typing import Tuple
from sklearn import metrics
from omegaconf import OmegaConf

sys.path.append(os.getcwd())

from rich.status import Status
from rich.console import Console

from framework import Envisioner, Memory
from interpreter.interpreter_parallel import Interpreter
from utils.config_mcts import load_task_desc, prep_agent_workspace, load_cfg
from utils.control_panel import start_control_panel, stop_control_panel, get_control_panel

dotenv.load_dotenv(override=True)
logger = logging.getLogger("ml-master")
console = Console()

# Print API configuration
console.print(f"[dim]Using API_KEY: {os.environ.get('OPENAI_API_KEY', 'Not set')}[/dim]")
console.print(f"[dim]Using BASE_URL: {os.environ.get('BASE_URL', 'Not set')}[/dim]")


# ==================== Grading Functions ====================

class InvalidSubmissionError(Exception):
    pass


def prepare_for_metric(
    submission: pd.DataFrame, answers: pd.DataFrame
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Prepare submission and answers for metric calculation"""
    assert (
        "formation_energy_ev_natom" in answers.columns
    ), "Expected 'formation_energy_ev_natom' column in answers"
    assert (
        "bandgap_energy_ev" in answers.columns
    ), "Expected 'bandgap_energy_ev' column in answers"
    if "formation_energy_ev_natom" not in submission.columns:
        raise InvalidSubmissionError(
            "Expected 'formation_energy_ev_natom' column in submission"
        )
    if "bandgap_energy_ev" not in submission.columns:
        raise InvalidSubmissionError(
            "Expected 'bandgap_energy_ev' column in submission"
        )
    if len(submission) != len(answers):
        raise InvalidSubmissionError(
            f"Expected {len(answers)} rows in submission, got {len(submission)}"
        )

    true_labels_formation, true_labels_bandgap = (
        answers["formation_energy_ev_natom"],
        answers["bandgap_energy_ev"],
    )
    predictions_formation, predictions_bandgap = (
        submission["formation_energy_ev_natom"],
        submission["bandgap_energy_ev"],
    )

    return (
        true_labels_formation,
        true_labels_bandgap,
        predictions_formation,
        predictions_bandgap,
    )


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """Calculate RMSLE score for submission"""
    (
        true_labels_formation,
        true_labels_bandgap,
        predictions_formation,
        predictions_bandgap,
    ) = prepare_for_metric(submission, answers)

    rmsle_formation = (
        metrics.mean_squared_log_error(true_labels_formation, predictions_formation)
        ** 0.5
    )
    rmsle_bandgap = (
        metrics.mean_squared_log_error(true_labels_bandgap, predictions_bandgap) ** 0.5
    )

    return (rmsle_formation + rmsle_bandgap) / 2


def run_grading(ground_truth_file_path: str, submission_file_path: str) -> float:
    """Run grading on best submission"""
    if os.path.exists(submission_file_path):
        gt = pd.read_csv(ground_truth_file_path)
        subm = pd.read_csv(submission_file_path)
        score = grade(subm, gt)
        return score
    else:
        return float('nan')


# ==================== Logging Filter ====================


class VerboseFilter(logging.Filter):
    """Filter (remove) logs that have verbose attribute set to True"""

    def filter(self, record):
        return not (hasattr(record, "verbose") and record.verbose)


def run():
    """Main execution loop for Framework 2"""
    begin_time = time.time()

    # Load configuration
    cfg = load_cfg()

    # Setup logging
    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, cfg.log_level.upper()), format=log_format, handlers=[]
    )

    # Don't want info logs from httpx
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    # Save logs to files
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(cfg.log_dir / "ml-master.log")
    file_handler.setFormatter(logging.Formatter(log_format))

    verbose_file_handler = logging.FileHandler(cfg.log_dir / "ml-master.verbose.log")
    verbose_file_handler.setFormatter(logging.Formatter(log_format))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))

    logger.addHandler(file_handler)
    logger.addHandler(verbose_file_handler)
    logger.addHandler(console_handler)

    logger.info(f'Starting run "{cfg.exp_name}" using Framework 2')
    # Load task description
    task_desc = load_task_desc(cfg)

    # Prepare workspace
    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    # Cleanup function
    global_step = [0]  # Use list to allow modification in nested function

    def cleanup():
        if global_step[0] == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    # Create Interpreter

    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),
        cfg=cfg
    )

    # Create Memory (global memory system)
    memory = Memory(
        capacity=5000,
        save_path=str(cfg.log_dir / "memory.pkl")
    )

    # Create Envisioner (global agent with MCTS tree)
    with Status("Initializing Envisioner..."):
        envisioner = Envisioner(
            interpreter=interpreter,
            memory=memory,
            task_description=task_desc,
            model_name="deepseek-reasoner",  # For strategy generation
            feedback_model_name="deepseek-chat",  # For metric extraction
            exploration_constant=cfg.agent.decay.exploration_constant,
            max_executor_count=cfg.agent.search.parallel_search_num,
            max_node_expansions=5,  # Hardcoded default
        )

    # Initialize root node
    with Status("Initializing root node..."):
        envisioner.initialize_root()

    # Start control panel
    console.print("\n[bold green]Starting Control Panel...[/bold green]")
    control_panel = start_control_panel()
    control_panel.add_log("Control panel started", "INFO")
    control_panel.add_log(f"Experiment: {cfg.exp_name}", "INFO")
    control_panel.add_log(f"Total steps: {cfg.agent.steps}", "INFO")

    # Setup progress tracking
    total_steps = cfg.agent.steps
    budget_per_step = 3  # Number of MCTS iterations per step
    best_score = float('inf')  # Track best grading score
    best_score_history = []

    # Grading paths (for nomad2018-predict-transparent-conductors)
    ground_truth_path = "data/nomad2018-predict-transparent-conductors/prepared/private/test.csv"
    submission_path = f"{cfg.workspace_dir}/best_submission/submission.csv"

    # Main MCTS loop
    control_panel.add_log(f"Starting MCTS search with {total_steps} steps", "INFO")

    for step in range(total_steps):
        # Update control panel with current step
        control_panel.update_envisioner(
            step=step + 1,
            total_steps=total_steps,
            budget=budget_per_step,
            best_metric=envisioner.best_metric,
            tree_size=envisioner.get_statistics().get('tree_size', 0),
            **envisioner.stats
        )

        logger.info(f"=== Step {step + 1}/{total_steps} ===")
        control_panel.add_log(f"Starting step {step + 1}/{total_steps}", "INFO")

        try:
            # Execute one MCTS step (selection -> expansion -> simulation -> backpropagation)
            control_panel.update_envisioner(phase="MCTS_STEP")
            envisioner.mcts_step(budget=budget_per_step)

            # Get statistics
            stats = envisioner.get_statistics()
            logger.info(
                f"Step {step + 1} completed: "
                f"tree_size={stats['tree_size']}, "
                f"selections={stats['selections']}, "
                f"expansions={stats['expansions']}, "
                f"simulations={stats['simulations']}"
            )

            # Update control panel with stats
            control_panel.update_envisioner(
                best_metric=envisioner.best_metric,
                tree_size=stats['tree_size'],
                **stats
            )

            # Log memory stats
            memory_stats = stats.get('memory_stats', {})
            if memory_stats:
                logger.info(
                    f"Memory: total_entries={memory_stats.get('total_entries', 0)}, "
                    f"best_reward={memory_stats.get('best_reward', 'N/A')}"
                )
                control_panel.add_log(
                    f"Memory: {memory_stats.get('total_entries', 0)} entries, "
                    f"success: {memory_stats.get('success_count', 0)}",
                    "INFO"
                )

            # Grade best submission
            current_score = run_grading(ground_truth_path, submission_path)
            if not pd.isna(current_score):
                best_score_history.append(current_score)
                if current_score < best_score:
                    best_score = current_score
                    logger.info(
                        f"New best score: {current_score:.6f} at step {step + 1}"
                    )
                    control_panel.add_log(
                        f"[NEW BEST] Score: {current_score:.6f} at step {step + 1}",
                        "SUCCESS"
                    )
                else:
                    control_panel.add_log(f"Current score: {current_score:.6f}", "INFO")
                logger.info(f"Current score: {current_score:.6f}, Best score: {best_score:.6f}")

            # Save memory periodically
            if (step + 1) % 5 == 0:
                memory.save()
                logger.info(f"Memory saved at step {step + 1}")
                control_panel.add_log(f"Memory saved at step {step + 1}", "INFO")

            global_step[0] = step + 1

        except Exception as e:
            logger.error(f"Step {step + 1} failed: {e}")
            control_panel.add_log(f"Step {step + 1} failed: {e}", "ERROR")
            continue

    # Stop control panel before final summary
    stop_control_panel()
    console.print("\n[bold yellow]Control panel stopped - Generating final summary...[/bold yellow]\n")

    # Cleanup
    interpreter.cleanup_session(-1)

    # Get best node
    best_node = envisioner.get_best_node()
    if best_node:
        logger.info(
            f"Best node found: {best_node.id[:8]} with {best_node.visits} visits "
            f"and value {best_node.value:.4f}"
        )
        if best_node.strategy.code:
            logger.info(f"Best code length: {len(best_node.strategy.code)} chars")

    # Save final results
    memory.save()
    stats = envisioner.get_statistics()
    console.print(f"[bold cyan]Experiment:[/bold cyan] {cfg.exp_name}")
    console.print(f"[bold cyan]Total Steps:[/bold cyan] {total_steps}")
    console.print(f"[bold cyan]Tree Size:[/bold cyan] {stats['tree_size']} nodes")
    console.print(f"[bold cyan]Total Selections:[/bold cyan] {stats['selections']}")
    console.print(f"[bold cyan]Total Expansions:[/bold cyan] {stats['expansions']}")
    console.print(f"[bold cyan]Total Simulations:[/bold cyan] {stats['simulations']}")
    console.print(f"[bold cyan]Best Metric:[/bold cyan] {envisioner.best_metric:.6f if envisioner.best_metric else 'N/A'}")

    # Log grading summary
    if best_score_history:
        console.print(f"\n[bold yellow]Grading Summary:[/bold yellow]")
        console.print(f"[yellow]  Total graded steps: {len(best_score_history)}[/yellow]")
        console.print(f"[yellow]  Best score: {best_score:.6f}[/yellow]")
        console.print(f"[yellow]  Final score: {best_score_history[-1]:.6f}[/yellow]")
        console.print(f"[yellow]  Score history (last 10): {best_score_history[-10:]}[/yellow]")

        logger.info("=" * 60)
        logger.info("=== Grading Summary ===")
        logger.info(f"Total graded steps: {len(best_score_history)}")
        logger.info(f"Best score: {best_score:.6f}")
        logger.info(f"Final score: {best_score_history[-1]:.6f}")
        logger.info(f"Score history (last 10): {best_score_history[-10:]}")
        logger.info("=" * 60)

    elapsed_time = time.time() - begin_time
    elapsed_min = int(elapsed_time // 60)
    elapsed_sec = int(elapsed_time % 60)
    console.print(f"\n[bold green]Total runtime:[/bold green] {elapsed_min:02d}:{elapsed_sec:02d}")
    logger.info(f"Run completed in {elapsed_time:.2f} seconds")

    return best_node


if __name__ == "__main__":
    run()
