"""
model/gradcam.py

GradCAM (Gradient-weighted Class Activation Mapping) implemented from scratch.

What it does:
  Given an image and a target class, GradCAM tells you WHICH spatial regions
  of the image were most important for that prediction.

How it works:
  1. Do a forward pass — save the activations from the last conv layer
  2. Do a backward pass for the target class — save the gradients at that same layer
  3. Global-average-pool the gradients to get channel importance weights
  4. Weighted sum of feature maps → raw heatmap
  5. ReLU + resize to input size → final heatmap

Why this layer:
  The last conv layer is the best trade-off between spatial resolution (still has
  spatial info) and semantic richness (deep enough to be meaningful).

Reference: Selvaraju et al., 2017 — https://arxiv.org/abs/1610.02391
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


class GradCAM:
    """
    GradCAM using forward and backward hooks.

    Usage:
        gradcam = GradCAM(model, target_layer=model.get_feature_layer())

        # Single image inference
        heatmap, pred_class, probs = gradcam(image_tensor, target_class=None)
        # target_class=None → use the predicted class

        # Overlay on original image
        overlay = gradcam.overlay(original_image_bgr, heatmap)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        """Forward hook: saves the feature map output."""
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        """Backward hook: saves the gradients flowing back through this layer."""
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        image: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Args:
            image: tensor of shape (1, C, H, W), already normalized
            target_class: class index to explain. None = use predicted class.

        Returns:
            heatmap: np.ndarray (H, W) in [0, 1] — the GradCAM map
            pred_class: int — the predicted class index
            probs: np.ndarray — softmax probabilities for all classes
        """
        self.model.eval()

        # --- Forward pass ---
        logits = self.model(image)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_class = int(np.argmax(probs))

        if target_class is None:
            target_class = pred_class

        # --- Backward pass for target class ---
        self.model.zero_grad()
        # Create a one-hot score for the target class
        score = logits[0, target_class]
        score.backward()

        # --- Compute GradCAM ---
        # gradients: (1, C, H, W) → channel-wise mean → (C,)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of feature maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)

        # ReLU: only keep positive contributions
        cam = F.relu(cam)

        # Resize to input image size
        h, w = image.shape[2], image.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class, probs

    def overlay(
        self,
        image_bgr: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlays the GradCAM heatmap on the original image.

        Args:
            image_bgr: (H, W, 3) uint8 BGR image (OpenCV format)
            heatmap: (H, W) float in [0, 1]
            alpha: heatmap opacity (0 = image only, 1 = heatmap only)
            colormap: OpenCV colormap (JET is standard for GradCAM)

        Returns:
            (H, W, 3) uint8 BGR overlay
        """
        # Resize heatmap to match image if needed
        if heatmap.shape != image_bgr.shape[:2]:
            heatmap = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))

        # Apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)

        # Blend
        overlay = cv2.addWeighted(image_bgr, 1 - alpha, colored_heatmap, alpha, 0)
        return overlay

    def remove_hooks(self):
        """Call this when done to avoid memory leaks."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass


def run_gradcam_on_batch(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    num_samples: int = 8,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Runs GradCAM on a batch and returns a grid of overlays.
    Used during training to visualize what the model is learning.

    Returns:
        grid: (H, W, 3) image grid (BGR)
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    from model.data.dataset import EMOTIONS

    gradcam = GradCAM(model, target_layer=model.get_feature_layer())

    images = images[:num_samples]
    labels = labels[:num_samples]

    fig, axes = plt.subplots(2, num_samples // 2, figsize=(16, 6))
    axes = axes.flatten()

    for i, (img, label) in enumerate(zip(images, labels)):
        img_input = img.unsqueeze(0)
        heatmap, pred_class, probs = gradcam(img_input)

        # De-normalize for display (approximate)
        img_display = img.permute(1, 2, 0).cpu().numpy()
        img_display = (img_display * 0.255 + 0.507).clip(0, 1)
        img_display_bgr = (img_display[:, :, ::-1] * 255).astype(np.uint8)

        overlay = gradcam.overlay(img_display_bgr, heatmap)
        overlay_rgb = overlay[:, :, ::-1]

        axes[i].imshow(overlay_rgb)
        gt = EMOTIONS[label.item()]
        pred = EMOTIONS[pred_class]
        color = "green" if gt == pred else "red"
        axes[i].set_title(f"GT: {gt}\nPred: {pred} ({probs[pred_class]:.0%})", color=color, fontsize=8)
        axes[i].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        print(f"[GradCAM] Saved visualization to {save_path}")
    plt.close()
    gradcam.remove_hooks()
