#!/usr/bin/env python3
"""
Parallel training experiment launcher for f32 vs bf16 initialization hypothesis test.

Usage:
    python launch_experiment.py --test-id 1 --num-pairs 16 --steps 1000 --start-seed 1601
    python launch_experiment.py --test-id 2 --num-pairs 2 --steps 10 --start-seed 1601  # Quick sanity check
"""

import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import typer

app = typer.Typer()


class ExperimentOrchestrator:
    """Manages lifecycle of parallel training runs."""

    def __init__(self, test_id: int, num_pairs: int, steps: int, start_seed: int):
        self.test_id = test_id
        self.num_pairs = num_pairs
        self.steps = steps
        self.start_seed = start_seed
        self.processes: List[subprocess.Popen] = []
        self.shutdown_requested = False

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle SIGINT/SIGTERM gracefully."""
        if self.shutdown_requested:
            typer.echo("\n⚠️  Force shutdown requested, exiting immediately")
            sys.exit(1)

        self.shutdown_requested = True
        typer.echo("\n" + "="*80)
        typer.echo("🛑 Shutdown signal received - stopping all workers gracefully...")
        typer.echo("="*80)
        self._cleanup()
        sys.exit(130)

    def _cleanup(self):
        """Terminate all worker processes."""
        if not self.processes:
            return

        typer.echo(f"Terminating {len(self.processes)} worker processes...")

        # Send SIGTERM to all
        for proc in self.processes:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass

        # Wait up to 10 seconds
        typer.echo("Waiting for clean shutdown (10s timeout)...")
        start = time.time()
        while time.time() - start < 10:
            if all(proc.poll() is not None for proc in self.processes):
                typer.echo("✓ All workers exited cleanly")
                return
            time.sleep(0.5)

        # Force kill stragglers
        typer.echo("⚠️  Timeout - force killing remaining processes")
        for proc in self.processes:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    def launch(self):
        """Launch all training runs."""
        log_dir = Path("../logs") / f"experiment_test{self.test_id:03d}"
        log_dir.mkdir(parents=True, exist_ok=True)

        typer.echo("="*80)
        typer.echo("F32 vs BF16 Initialization Experiment")
        typer.echo("="*80)
        typer.echo(f"Test ID: {self.test_id}")
        typer.echo(f"Matched pairs: {self.num_pairs}")
        typer.echo(f"Steps per run: {self.steps}")
        typer.echo(f"Seeds: {self.start_seed} to {self.start_seed + self.num_pairs - 1}")
        typer.echo(f"Total runs: {self.num_pairs * 2}")
        typer.echo("\nPress Ctrl-C to stop gracefully")
        typer.echo("="*80)
        typer.echo()

        # Launch all runs
        for i in range(self.num_pairs):
            seed = self.start_seed + i

            # f32→bf16 run
            self._launch_worker(
                init_method="f32",
                seed=seed,
                log_dir=log_dir
            )

            # pure bf16 run
            self._launch_worker(
                init_method="bf16",
                seed=seed,
                log_dir=log_dir
            )

            time.sleep(1)  # Stagger startup

        typer.echo(f"\n✓ Launched {len(self.processes)} workers")
        typer.echo(f"✓ Logs in: {log_dir}")
        typer.echo("\nWaiting for completion...\n")

        # Wait for all to complete
        try:
            for proc in self.processes:
                proc.wait()
        except KeyboardInterrupt:
            # Handled by signal handler
            pass

        if not self.shutdown_requested:
            typer.echo("\n" + "="*80)
            typer.echo("✓ All runs completed")
            typer.echo("="*80)

    def _launch_worker(self, init_method: str, seed: int, log_dir: Path):
        """Launch a single training worker."""
        run_id = f"test{self.test_id:03d}_{init_method}_seed{seed:04d}"
        log_file = log_dir / f"{run_id}.log"

        script = "train_f32_to_bf16.py" if init_method == "f32" else "train_pure_bf16.py"

        cmd = [
            sys.executable,  # Use same Python interpreter as launcher
            script,
            "--seed", str(seed),
            "--test-id", str(self.test_id),
            "--output", run_id,
            "--steps", str(self.steps),
        ]

        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).parent
            )

        self.processes.append(proc)
        typer.echo(f"  Started: {run_id} (PID {proc.pid})")


@app.command()
def main(
    test_id: int = typer.Option(..., help="Experiment test ID (encoded in all outputs)"),
    num_pairs: int = typer.Option(16, help="Number of matched f32/bf16 pairs to run"),
    steps: int = typer.Option(1000, help="Training steps per run"),
    start_seed: int = typer.Option(1601, help="Starting random seed (increments by 1)"),
):
    """Launch parallel f32 vs bf16 training experiment."""
    orchestrator = ExperimentOrchestrator(test_id, num_pairs, steps, start_seed)
    orchestrator.launch()


if __name__ == "__main__":
    app()
