"""
agents/training_agent.py

EmotionMirror Training Agent

An autonomous agent that runs the full ML experimentation loop:
  Train → Evaluate → Suggest → Train → ...

This implements a simple ReAct-style agent pattern:
  Reason: analyze current results
  Act: train the model / run evaluation
  Observe: read metrics from checkpoint
  Reason again: decide what to change
  Act again: generate new config → train → ...

Usage:
    # Run 3 automated experimentation rounds
    python agents/training_agent.py --rounds 3 --initial-config model/configs/baseline.yaml

    # Just run one round (train + evaluate + suggest next config)
    python agents/training_agent.py --rounds 1

This is the "agents" layer you asked about — it wraps your ML skills
into an autonomous loop that can run overnight and come back with
a report of what worked.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


AGENT_LOG_DIR = Path("agents/runs")
EXPERIMENTS_LOG = AGENT_LOG_DIR / "experiments.json"


class TrainingAgent:
    """
    A simple autonomous agent for ML experimentation.

    Tools available to this agent:
      - train_model(config_path) → runs model/train.py
      - evaluate_model(checkpoint_path) → runs evaluate_model skill
      - suggest_next_config(report, current_config) → runs suggest_hyperparams skill

    The agent's loop:
      for each round:
        1. Train with current config
        2. Evaluate the trained model
        3. Ask Claude to suggest the next config
        4. Save suggestion as next config
        5. Log everything
    """

    def __init__(self, initial_config_path: str, max_rounds: int = 3):
        self.current_config_path = Path(initial_config_path)
        self.max_rounds = max_rounds
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = AGENT_LOG_DIR / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

        print(f"\n{'='*60}")
        print(f"  EmotionMirror Training Agent")
        print(f"  Run ID: {self.run_id}")
        print(f"  Rounds: {self.max_rounds}")
        print(f"  Initial config: {self.current_config_path}")
        print(f"{'='*60}\n")

    # ─── Tools ─────────────────────────────────────────────────

    def train_model(self, config_path: str) -> Optional[str]:
        """
        Tool: runs model/train.py and returns the checkpoint path if successful.
        """
        print(f"\n[Agent] ACTION: train_model({config_path})")

        result = subprocess.run(
            [sys.executable, "model/train.py", "--config", config_path],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"[Agent] ⚠️  Training failed:\n{result.stderr[-2000:]}")
            return None

        # Find the checkpoint from the output
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        checkpoint_path = Path(cfg["output"]["checkpoint_dir"]) / f"{cfg['experiment_name']}_best.pt"
        if checkpoint_path.exists():
            print(f"[Agent] ✓ Training complete. Checkpoint: {checkpoint_path}")
            return str(checkpoint_path)

        print("[Agent] ⚠️  Checkpoint not found after training")
        return None

    def evaluate_model(self, checkpoint_path: str) -> Optional[dict]:
        """
        Tool: runs the evaluate_model skill and returns the structured report.
        """
        print(f"\n[Agent] ACTION: evaluate_model({checkpoint_path})")

        try:
            from agents.skills.evaluate_model import evaluate_model_skill
            report = evaluate_model_skill(checkpoint_path, str(self.run_dir))
            print(f"[Agent] OBSERVATION: test_accuracy={report['test_accuracy']:.3f}, "
                  f"macro_f1={report['macro_f1']:.3f}")
            return report
        except Exception as e:
            print(f"[Agent] ⚠️  Evaluation failed: {e}")
            return None

    def suggest_next_config(self, report: dict, current_config_path: str, round_num: int) -> Optional[str]:
        """
        Tool: calls Claude to suggest the next config to try.
        Returns path to the new config file.
        """
        print(f"\n[Agent] ACTION: suggest_next_config (round {round_num})")

        next_config_path = self.run_dir / f"config_round_{round_num + 1:02d}.yaml"

        try:
            from agents.skills.suggest_hyperparams import run_suggestion
            run_suggestion(
                current_config_path=current_config_path,
                eval_report_path=str(self.run_dir / f"{report['experiment_name']}_eval_report.json"),
                output_path=str(next_config_path),
                experiments_log_path=str(EXPERIMENTS_LOG),
            )
            print(f"[Agent] OBSERVATION: New config saved → {next_config_path}")
            return str(next_config_path)
        except Exception as e:
            print(f"[Agent] ⚠️  Suggestion failed: {e}")
            return None

    # ─── Main loop ────────────────────────────────────────────

    def run(self):
        """
        Main agent loop: Train → Evaluate → Suggest → repeat.
        """
        current_config = str(self.current_config_path)

        for round_num in range(1, self.max_rounds + 1):
            print(f"\n{'─'*60}")
            print(f"  Round {round_num}/{self.max_rounds}")
            print(f"{'─'*60}")

            # Step 1: Train
            checkpoint_path = self.train_model(current_config)
            if checkpoint_path is None:
                print("[Agent] Cannot continue — training failed")
                break

            # Step 2: Evaluate
            report = self.evaluate_model(checkpoint_path)
            if report is None:
                print("[Agent] Cannot continue — evaluation failed")
                break

            # Step 3: Log round
            round_result = {
                "round": round_num,
                "config": current_config,
                "checkpoint": checkpoint_path,
                "test_accuracy": report["test_accuracy"],
                "macro_f1": report["macro_f1"],
                "best_val_accuracy": report["best_val_accuracy"],
            }
            self.history.append(round_result)
            self._save_history()

            print(f"\n[Agent] Round {round_num} summary:")
            print(f"  Test accuracy: {report['test_accuracy']:.3f}")
            print(f"  Macro F1:      {report['macro_f1']:.3f}")

            # Step 4: Suggest (unless last round)
            if round_num < self.max_rounds:
                next_config = self.suggest_next_config(report, current_config, round_num)
                if next_config:
                    current_config = next_config
                else:
                    print("[Agent] Suggestion failed — repeating current config")

        # Final report
        self._print_final_report()

    def _save_history(self):
        history_path = self.run_dir / "agent_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def _print_final_report(self):
        print(f"\n{'='*60}")
        print("  Agent run complete")
        print(f"{'='*60}")
        print(f"\n{'Round':<8} {'Test Acc':>10} {'Macro F1':>10} {'Config'}")
        print("-" * 60)
        for r in self.history:
            print(f"  {r['round']:<6} {r['test_accuracy']:>10.3f} {r['macro_f1']:>10.3f}  {Path(r['config']).name}")

        if self.history:
            best = max(self.history, key=lambda x: x["test_accuracy"])
            print(f"\nBest result: Round {best['round']} — accuracy {best['test_accuracy']:.3f}")
            print(f"Best config: {best['config']}")
            print(f"Best checkpoint: {best['checkpoint']}")

        print(f"\nFull run artifacts: {self.run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EmotionMirror autonomous training agent")
    parser.add_argument("--rounds", type=int, default=3, help="Number of train-evaluate-suggest cycles")
    parser.add_argument("--initial-config", default="model/configs/baseline.yaml")
    args = parser.parse_args()

    agent = TrainingAgent(
        initial_config_path=args.initial_config,
        max_rounds=args.rounds,
    )
    agent.run()
