"""
agents/skills/suggest_hyperparams.py

AI Agent Skill: Hyperparameter Suggestion

Given a training history and current config, this skill calls Claude
to suggest the next config to try — simulating a lightweight Bayesian
optimization agent without needing Optuna or Ray Tune.

This teaches a key agent pattern: tool → observe → reason → suggest → loop.

Usage:
    python agents/skills/suggest_hyperparams.py \
        --history model/checkpoints/cnn_baseline_fer2013_eval_report.json \
        --current-config model/configs/baseline.yaml \
        --output model/configs/agent_suggested.yaml
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Optional


SYSTEM_PROMPT = """You are an expert ML engineer specializing in computer vision and emotion recognition.
You help optimize PyTorch training configs by analyzing training results and suggesting improvements.
Always respond with valid YAML only — no explanation, no markdown fences, just raw YAML."""


def suggest_next_config(
    current_config: dict,
    eval_report: dict,
    previous_experiments: Optional[list] = None,
) -> str:
    """
    Calls Claude to suggest the next config to try based on:
      - Current config
      - Evaluation report (metrics, confusion matrix, weak classes)
      - (Optional) History of previous experiments
    """
    import anthropic
    client = anthropic.Anthropic()

    context = {
        "current_config": current_config,
        "eval_report": eval_report,
        "previous_experiments": previous_experiments or [],
    }

    user_prompt = f"""Based on this training context, suggest the next config to try.

Context:
{json.dumps(context, indent=2)}

Rules for your suggestion:
1. Change at most 3 things from the current config — don't change everything at once
2. Focus on addressing the weakest emotion classes: {_get_weak_classes(eval_report)}
3. If val_acc is much lower than train_acc → increase regularization
4. If both are low → try higher learning rate or more capacity
5. Consider the confused emotion pairs: {_get_confused_pairs(eval_report)}

Output the complete modified YAML config only. No explanation."""

    response = anthropic.Anthropic().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return response.content[0].text


def _get_weak_classes(report: dict) -> list:
    """Returns emotion classes with F1 < 0.6."""
    weak = []
    for emotion, metrics in report.get("per_class", {}).items():
        if metrics.get("f1", 1.0) < 0.6:
            weak.append(f"{emotion} (F1={metrics['f1']:.2f})")
    return weak


def _get_confused_pairs(report: dict) -> list:
    """Returns top 3 most confused emotion pairs."""
    pairs = report.get("most_confused_pairs", [])[:3]
    return [f"{p['true']}→{p['predicted']} ({p['count']}x)" for p in pairs]


def run_suggestion(
    current_config_path: str,
    eval_report_path: str,
    output_path: str,
    experiments_log_path: Optional[str] = None,
):
    with open(current_config_path) as f:
        current_config = yaml.safe_load(f)

    with open(eval_report_path) as f:
        eval_report = json.load(f)

    previous_experiments = []
    if experiments_log_path and Path(experiments_log_path).exists():
        with open(experiments_log_path) as f:
            previous_experiments = json.load(f)

    print(f"[HyperparamSkill] Current accuracy: {eval_report.get('test_accuracy', '?'):.3f}")
    print(f"[HyperparamSkill] Weak classes: {_get_weak_classes(eval_report)}")
    print(f"[HyperparamSkill] Calling Claude for suggestion...")

    suggested_yaml = suggest_next_config(current_config, eval_report, previous_experiments)

    # Validate it's parseable YAML
    try:
        suggested_config = yaml.safe_load(suggested_yaml)
        # Auto-name the experiment
        run_num = len(previous_experiments) + 1
        suggested_config["experiment_name"] = f"agent_run_{run_num:02d}"
    except yaml.YAMLError as e:
        print(f"[HyperparamSkill] ⚠️  Claude returned invalid YAML: {e}")
        print("Raw output:")
        print(suggested_yaml)
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(suggested_config, f, default_flow_style=False)

    print(f"[HyperparamSkill] ✓ New config saved to {output_path}")
    print(f"[HyperparamSkill] Changes suggested:")

    _print_diff(current_config, suggested_config)

    # Log to experiments history
    if experiments_log_path:
        experiment_entry = {
            "run": run_num,
            "config": suggested_config,
            "based_on_accuracy": eval_report.get("test_accuracy"),
        }
        previous_experiments.append(experiment_entry)
        with open(experiments_log_path, "w") as f:
            json.dump(previous_experiments, f, indent=2)


def _print_diff(old: dict, new: dict, prefix: str = ""):
    """Recursively prints changed values between two dicts."""
    for key in set(list(old.keys()) + list(new.keys())):
        full_key = f"{prefix}.{key}" if prefix else key
        if key not in old:
            print(f"  + {full_key}: {new[key]}")
        elif key not in new:
            print(f"  - {full_key}")
        elif isinstance(old[key], dict) and isinstance(new[key], dict):
            _print_diff(old[key], new[key], full_key)
        elif old[key] != new[key]:
            print(f"  ~ {full_key}: {old[key]} → {new[key]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, help="Path to eval_report.json")
    parser.add_argument("--current-config", required=True, help="Path to current YAML config")
    parser.add_argument("--output", required=True, help="Path to save suggested config")
    parser.add_argument("--experiments-log", help="Path to experiments log JSON (optional)")
    args = parser.parse_args()

    run_suggestion(
        current_config_path=args.current_config,
        eval_report_path=args.history,
        output_path=args.output,
        experiments_log_path=args.experiments_log,
    )
