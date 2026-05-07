# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EmotionMirror is a facial emotion detection system trained on FER2013. It classifies 7 emotions (angry, disgust, fear, happy, neutral, sad, surprise) from 48×48 face images, with a planned journaling/LLM layer on top.

## Repository Layout

```
model/
  model.py          — CNN + EfficientNet architectures, build_model() factory
  train.py          — main training script
  gradcam.py        — GradCAM implementation
  data/
    dataset.py      — FER2013Dataset, build_dataloaders()
  configs/
    baseline.yaml   — CustomCNN config
    efficientnet.yaml — EfficientNet-B0 config
  checkpoints/      — saved .pt and .onnx files (gitignored)
agents/
  training_agent.py — autonomous Train→Evaluate→Suggest loop
  skills/
    evaluate_model.py      — evaluation skill (metrics + optional AI summary)
    suggest_hyperparams.py — Claude API hyperparameter suggestion
scripts/
  download_data.py  — Kaggle FER2013 download
backend/            — FastAPI server (Phase 2, TBD)
frontend/           — React web app (Phase 3, TBD)
mobile/             — React Native / Expo (Phase 4, TBD)
```

## Setup

**1. Download FER2013 dataset** (requires Kaggle credentials at `~/.kaggle/kaggle.json`):

```bash
python download_data.py
```

Expected output: `model/data/fer2013/train/{angry,disgust,...}/` and `model/data/fer2013/test/`.

**2. Install dependencies** (no requirements.txt yet — inferred from imports):

```
torch torchvision albumentations scikit-learn matplotlib seaborn opencv-python pyyaml kaggle anthropic
```

## Commands

```bash
# Train with a config file
python train.py --config model/configs/baseline.yaml

# Evaluate a checkpoint
python evaluate_model.py --checkpoint model/checkpoints/baseline_best.pt --output-dir model/checkpoints

# Evaluate + generate Claude AI narrative summary
python evaluate_model.py --checkpoint model/checkpoints/baseline_best.pt --ai-summary

# Suggest next hyperparams based on evaluation results
python suggest_hyperparams.py \
  --history model/checkpoints/baseline_eval_report.json \
  --current-config model/configs/baseline.yaml \
  --output model/configs/next.yaml

# Run the autonomous training agent (train → evaluate → suggest, N rounds)
python training_agent.py --rounds 3 --initial-config model/configs/baseline.yaml
```

## Architecture

### Model layer (`model.py`)

Two architectures share the same interface (both output raw logits, both expose `get_feature_layer()` for GradCAM):

- **CustomCNN** — 4 conv blocks (3→32→64→128→256ch), 48×48 input, ~2.1M params. Fast, interpretable.
- **EfficientNetEmotion** — EfficientNet-B0 backbone + custom head, 224×224 input, ~4.0M params. Higher accuracy (~70% vs ~62%). Backbone frozen during warm-up epochs, then unfrozen for full fine-tuning.

`build_model(cfg)` is the factory — use this everywhere.

### Data layer (`dataset.py`)

- `FER2013Dataset` loads from folder structure (`data_dir/train/<emotion>/*.jpg`).
- `build_dataloaders(cfg)` carves a val set from the train split (last 10%), returns `(train_loader, val_loader, test_loader)`.
- Training uses `WeightedRandomSampler` to handle FER2013's class imbalance (happy has ~3× more samples than disgust).
- Train augmentations via Albumentations; val/test are resize+normalize only.
- Grayscale images are stacked to 3 channels so both models share the same input format.

### Training loop (`train.py`)

Config-driven via YAML. Key behaviors:

- EfficientNet backbone freezing: controlled by `model.freeze_backbone_epochs` in config; optimizer is **rebuilt** when backbone unfreezes (important — don't miss this when modifying).
- OneCycleLR steps per batch; CosineAnnealingLR/StepLR step per epoch.
- GradCAM visualizations saved every 5 epochs to the checkpoint dir.
- Best checkpoint saves the full config inside the `.pt` file — `evaluate_model.py` relies on this to reconstruct the model without a separate config arg.
- Exports to ONNX at the end if `output.export_onnx: true` in config.

### Autonomous agent layer (`training_agent.py` + skills)

ReAct-style loop: **Train → Evaluate → Suggest → repeat**

- `TrainingAgent` orchestrates the loop; each step is a "tool call" to a skill.
- `evaluate_model.py` produces a structured JSON report (per-class metrics, confusion matrix, most-confused pairs, confidence analysis).
- `suggest_hyperparams.py` calls the Claude API with the eval report and current config, gets back a new YAML config with ≤3 changes. Previous experiment history is accumulated in `agents/runs/experiments.json`.
- Agent runs are logged to `agents/runs/<timestamp>/`.

### GradCAM (`gradcam.py`)

Implemented from scratch using PyTorch forward/backward hooks on `model.get_feature_layer()`. Call `gradcam.remove_hooks()` when done to avoid memory leaks. Used both interactively and during training (every 5 epochs).

## Config Structure

YAML configs (in `model/configs/`) drive everything. Key sections:

- `experiment_name` — used for all artifact filenames
- `model.architecture` — `"custom_cnn"` or `"efficientnet_b0"`
- `model.freeze_backbone_epochs` — warm-up phase for EfficientNet
- `data.image_size` — 48 for CustomCNN, 224 for EfficientNet (also controls RGB mode)
- `training.scheduler` — `"cosine"`, `"one_cycle"`, or `"step"`
- `output.export_onnx` — whether to export after training

## Claude API Usage

Two skills call the Anthropic API:

- `evaluate_model.py::generate_ai_summary()` — generates narrative analysis of eval results
- `suggest_hyperparams.py::suggest_next_config()` — returns raw YAML (no markdown), parsed directly

Both use `claude-sonnet-4-20250514`. The `ANTHROPIC_API_KEY` env var must be set for the agent loop to work.

conversion with claude about the project (https://claude.ai/share/754e280f-c129-4cf2-b556-ccfaa6122ae9):

I am a full stack engineer and deep learning & AI enthusiastic. I have bsc in information systems and cognition science and interesting a lot in brain science and medical in general.
I want to develop by using AI agents a project that will touch in AI, DL and full stack worlds for increase and improve my knoledge and skills and also will help me to make my profile popular and interesting for recruiters and team leaders for potential offers in hi-tech in both domains. Give me some ideas for ai + machine learning + full stack project involves creating, training and testing a model (in computer vision or other DL categories) that would be usefull and practicle for users and I could deploy it as an interactive website and mobile app.

calude answer:
Here are 6 project ideas tailored to your profile. Click any card to see the full breakdown, then hit "Get full roadmap" for a deep dive.
A few thoughts on the selection strategy:
Why these specific projects hit hard with recruiters. Each one combines model training from scratch (not just API calls), a public dataset you can cite, and a deployable product — that trifecta is what separates "I used ChatGPT" from "I built an AI system." Recruiters at ML teams and full stack teams both immediately understand the scope.
My top recommendation for your profile specifically is either SkinSight or NeuroSight. Your cognitive science + information systems background gives you a genuine contextual edge when explaining the medical domain — most CV engineers don't have that, and it makes for a much better interview story than "I trained a classifier." You can frame it as domain-informed AI, not just a technical exercise.
On difficulty: GestureFlow and PostureCoach are the fastest to ship (2–4 weeks), which is valuable if you want something live soon. SkinSight and DocRadar are the most impressive on paper and the most likely to generate LinkedIn traction.
Cross-platform note: all six are designed to be deployable as both a web app and a React Native mobile app without rewriting your backend, which checks both full stack and mobile boxes for recruiters.

the option selected:

EmotionMirror

Real-time facial emotion detection & journaling

Intermediate
Why this project

Detect emotions from webcam in real-time. Add a journaling layer: track emotional state over time, show charts, and use an LLM to generate reflective prompts. Connects your cognition science background directly.
Model / ML

Custom CNN + AffectNet / FER2013 dataset
Stack

PyTorch · Flask · React · WebRTC
Deployment plan

Flask + React on Vercel/Render, React Native for mobile
Recruiter wow factor — Emotion AI is booming in HCI, mental health tech, and HR tech

my question:
Tell me more about building the EmotionMirror project — give me a detailed roadmap, architecture, and first steps.

the architecture file exists in the project and called: emotionmirror_architecture.svg

On the model choice. Start with FER2013 — it's small (35k images, fits in RAM), well-studied, and you'll get a first working model in a day. Once you have a baseline, add AffectNet data for the 7 emotions that matter (happy, sad, angry, disgust, fear, surprise, neutral). Don't try to be perfect at training — aim for ~65–70% validation accuracy on FER2013, which is roughly what EfficientNet-B0 fine-tuned achieves. The interesting recruiter story isn't "I got 80%" — it's "I analyzed where the model fails and why."
GradCAM is your differentiator. Most emotion detection demos online are just a label on a webcam feed. Showing the heatmap of what the model focused on (eyebrows for anger, mouth corners for happiness) is what turns this from a demo into a research-quality project. Implement it from scratch using PyTorch hooks — don't use a library, because implementing it yourself is what you'll talk about in interviews.
The journal + LLM layer ties your cognition science background in directly. When the system detects a pattern (e.g. high anxiety on weekday mornings), it calls the LLM API and generates a reflective prompt grounded in CBT-style journaling. This is a genuine product feature, not a gimmick — and it connects your two degrees to the technical work in a way most CV/ML portfolios completely miss.
For your first working version, skip mobile. Get the web app live in week 6, get a real URL, and start sharing it. Add React Native in weeks 7–8 once the backend is stable. Many people get stuck polishing before shipping — a live imperfect demo beats a perfect unreleased one every time.
Want me to go deeper on any specific part — the training code, the FastAPI WebSocket setup, the GradCAM implementation, or the React webcam streaming logic?
