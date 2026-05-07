"""
agents/skills/evaluate_model.py

AI Agent Skill: Model Evaluation & Report Generation

This skill is called by an agent after training completes.
It automatically:
  1. Loads the best checkpoint
  2. Runs full test evaluation
  3. Generates a structured report (metrics, confusion matrix, per-class analysis)
  4. Asks Claude to write a human-readable summary with improvement suggestions

Usage (as an agent tool):
    from agents.skills.evaluate_model import evaluate_model_skill
    report = await evaluate_model_skill(checkpoint_path, config_path)

Or via CLI:
    python agents/skills/evaluate_model.py \
        --checkpoint model/checkpoints/cnn_baseline_fer2013_best.pt \
        --config model/configs/baseline.yaml
"""

import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Loads model and config from a saved checkpoint."""
    from model.model import build_model

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, checkpoint


def run_full_evaluation(checkpoint_path: str, output_dir: Optional[str] = None) -> dict:
    """
    Core evaluation function.

    Returns a structured report dict that can be:
      - Saved as JSON
      - Passed to Claude for narrative summary
      - Used to populate a dashboard
    """
    import torch.nn as nn
    from model.data.dataset import build_dataloaders, EMOTIONS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, checkpoint = load_model_from_checkpoint(checkpoint_path, device)

    _, _, test_loader = build_dataloaders(cfg)

    criterion = nn.CrossEntropyLoss()

    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Per-class metrics
    report = classification_report(all_labels, all_preds, target_names=EMOTIONS, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    # Most confused pairs
    cm_np = np.array(cm)
    np.fill_diagonal(cm_np, 0)
    confused_pairs = []
    for _ in range(5):
        i, j = np.unravel_index(cm_np.argmax(), cm_np.shape)
        confused_pairs.append({
            "true": EMOTIONS[i],
            "predicted": EMOTIONS[j],
            "count": int(cm_np[i, j]),
        })
        cm_np[i, j] = 0

    # Confidence analysis
    correct_mask = all_preds == all_labels
    mean_confidence_correct = float(all_probs.max(axis=1)[correct_mask].mean())
    mean_confidence_wrong = float(all_probs.max(axis=1)[~correct_mask].mean())

    structured_report = {
        "experiment_name": cfg["experiment_name"],
        "architecture": cfg["model"]["architecture"],
        "checkpoint": str(checkpoint_path),
        "trained_epochs": checkpoint["epoch"],
        "best_val_accuracy": float(checkpoint["val_accuracy"]),
        "test_accuracy": float((all_preds == all_labels).mean()),
        "test_loss": float(total_loss / len(test_loader)),
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro")),
        "weighted_f1": float(f1_score(all_labels, all_preds, average="weighted")),
        "per_class": {
            emotion: {
                "precision": round(report[emotion]["precision"], 3),
                "recall": round(report[emotion]["recall"], 3),
                "f1": round(report[emotion]["f1-score"], 3),
                "support": report[emotion]["support"],
            }
            for emotion in EMOTIONS
        },
        "confusion_matrix": cm,
        "most_confused_pairs": confused_pairs,
        "confidence_analysis": {
            "mean_when_correct": round(mean_confidence_correct, 3),
            "mean_when_wrong": round(mean_confidence_wrong, 3),
        },
        "dataset": cfg["data"]["dataset"],
        "image_size": cfg["data"]["image_size"],
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / f"{cfg['experiment_name']}_eval_report.json"
        with open(report_path, "w") as f:
            json.dump(structured_report, f, indent=2)
        print(f"[EvalSkill] Report saved to {report_path}")

    return structured_report


async def generate_ai_summary(report: dict, api_key: Optional[str] = None) -> str:
    """
    Calls Claude API to generate a human-readable analysis and
    actionable improvement suggestions based on the evaluation report.

    This is the 'AI agent' part — the agent evaluates its own work
    and suggests next steps autonomously.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""You are an ML engineer reviewing an emotion detection model's evaluation results.

Here is the evaluation report:
{json.dumps(report, indent=2)}

Please provide:
1. A brief 2-3 sentence summary of overall performance
2. The 2-3 most important weaknesses (focus on the most confused emotion pairs and lowest-F1 classes)
3. Three concrete, actionable improvement suggestions (e.g. specific augmentation strategies, architecture changes, data collection ideas)
4. A recommended next experiment to run

Be specific and technical. Format as markdown."""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def evaluate_model_skill(checkpoint_path: str, output_dir: str = "model/checkpoints") -> dict:
    """
    Synchronous entry point for use as an agent tool.
    Returns both the structured report and the AI summary.
    """
    print(f"[EvalSkill] Evaluating: {checkpoint_path}")
    report = run_full_evaluation(checkpoint_path, output_dir)

    print("\n[EvalSkill] Key metrics:")
    print(f"  Test accuracy:  {report['test_accuracy']:.3f}")
    print(f"  Macro F1:       {report['macro_f1']:.3f}")
    print(f"  Best val acc:   {report['best_val_accuracy']:.3f}")
    print("\n[EvalSkill] Most confused pairs:")
    for pair in report["most_confused_pairs"][:3]:
        print(f"  {pair['true']} → {pair['predicted']}: {pair['count']} times")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="model/checkpoints")
    parser.add_argument("--ai-summary", action="store_true", help="Generate Claude AI summary")
    args = parser.parse_args()

    report = evaluate_model_skill(args.checkpoint, args.output_dir)

    if args.ai_summary:
        import asyncio
        summary = asyncio.run(generate_ai_summary(report))
        print("\n" + "="*60)
        print("AI Analysis:")
        print("="*60)
        print(summary)

        summary_path = Path(args.output_dir) / f"{report['experiment_name']}_ai_analysis.md"
        with open(summary_path, "w") as f:
            f.write(f"# Model Evaluation Analysis\n\n")
            f.write(f"**Experiment:** {report['experiment_name']}\n")
            f.write(f"**Test accuracy:** {report['test_accuracy']:.3f}\n\n")
            f.write(summary)
        print(f"\nAI analysis saved to {summary_path}")
