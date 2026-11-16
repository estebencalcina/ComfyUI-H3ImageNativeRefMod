"""Experimental native-reference cache for MiniMax H3-Image ComfyUI.

NOT compatible with MiniMax H3 VIDEO refmods. This stores BOTH a VAE latent and
Qwen vision pixels, so that conditioning uses precisely the two streams expected
by the H3-Image pipeline. This is a full latent, not a learned identity embedding.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import comfy.model_management
import comfy.utils
import folder_paths
import node_helpers
from safetensors.torch import load_file, save_file
from safetensors import safe_open

_FORMAT = "h3-image-native-multiref-v2"
_DIR = Path(folder_paths.models_dir) / "h3_image_native_refmods"
_DIR.mkdir(parents=True, exist_ok=True)


def _options():
    files = sorted(x.name for x in _DIR.glob("*.safetensors") if x.is_file())
    return files or ["NO_IDENTITIES_YET.safetensors"]


def _resize(image, width, height):
    """Match the official H3-Image resize operation, RGB BHWC."""
    image = image[..., :3]
    if image.shape[2] == width and image.shape[1] == height:
        return image
    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height,
                                      "lanczos", "disabled").movedim(1, -1)


def _short_edge(width, height, target):
    scale = target / min(width, height)
    return (max(32, round(width * scale / 32) * 32),
            max(32, round(height * scale / 32) * 32))


def _safe_name(name):
    name = "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in name).strip("_")
    if not name:
        raise ValueError("Identity name is empty after sanitization")
    return name[:80]


class SaveH3ImageNativeIdentity:
    """Build a reusable multi-view identity from 1-8 image references.

    Each reference is retained separately; this is NOT a learned/merged embedding.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_1": ("IMAGE",), "vae": ("VAE",),
            "identity_name": ("STRING", {"default": "Personaje_01"}),
            "reference_short_edge": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
        }, "optional": {f"image_{i}": ("IMAGE",) for i in range(2, 9)}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_file",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3-Image / Native RefMod (experimental)"

    def save(self, image_1, vae, identity_name, reference_short_edge=768, **other_images):
        images = [image_1] + [other_images.get(f"image_{i}") for i in range(2, 9)]
        images = [x for x in images if x is not None]
        name = _safe_name(identity_name)
        tensors, per_view = {}, []
        for j, image in enumerate(images):
            image = image[:1, :, :, :3]
            h, w = image.shape[1:3]
            # Save both native image VAE latents and a Qwen-readable pixel image.
            # Smaller references reduce disk/token cost but may reduce identity fidelity.
            wh = _short_edge(w, h, int(reference_short_edge))
            resized = _resize(image, *wh).clamp(0, 1)
            latent = vae.encode(resized)
            if latent.ndim == 4:
                latent = latent.unsqueeze(2)
            if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != 24 or latent.shape[2] != 1:
                raise ValueError(f"Unexpected H3-Image VAE shape on image {j+1}: {tuple(latent.shape)}")
            tensors[f"latent_{j}"] = latent.detach().cpu().contiguous()
            tensors[f"vision_{j}"] = (resized * 255).round().to(torch.uint8).cpu().contiguous()
            per_view.append({"latent_shape": list(latent.shape), "image_shape": list(resized.shape)})
        meta = {"format": _FORMAT, "vae": "MiniMax-H3-Image only", "count": str(len(images)),
                "views": json.dumps(per_view), "identity_name": name}
        dest = _DIR / f"{name}.safetensors"
        fd, tmp = tempfile.mkstemp(dir=str(_DIR), prefix=".h3img_", suffix=".tmp")
        os.close(fd)
        try:
            save_file(tensors, tmp, metadata=meta)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return (str(dest),)


class LoadH3ImageNativeIdentity:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"identity_file": (_options(),)}}
    RETURN_TYPES = ("H3IMAGE_NATIVE_IDENTITY", "IMAGE")
    RETURN_NAMES = ("identity", "first_reference_preview")
    FUNCTION = "load"
    CATEGORY = "MiniMax H3-Image / Native RefMod (experimental)"

    @classmethod
    def IS_CHANGED(cls, identity_file):
        path = _DIR / identity_file
        return path.stat().st_mtime_ns if path.is_file() else float("nan")

    def load(self, identity_file):
        if identity_file not in _options() or identity_file == "NO_IDENTITIES_YET.safetensors":
            raise ValueError("Save an 8-image identity first, then refresh the node / ComfyUI")
        path = (_DIR / identity_file).resolve()
        if path.parent != _DIR.resolve():
            raise ValueError("Invalid path")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
            if meta.get("format") != _FORMAT:
                raise ValueError("Wrong format: regenerate with 8-image creator; video RefMods/older prototype are not compatible")
            count = int(meta["count"])
        if not 1 <= count <= 8:
            raise ValueError(f"Invalid number of reference views: {count}")
        tensors = load_file(str(path), device="cpu")
        latents, visions = [], []
        for i in range(count):
            latent = tensors[f"latent_{i}"]
            vision = tensors[f"vision_{i}"]
            if latent.ndim != 5 or latent.shape[:3] != (1,24,1):
                raise ValueError(f"Invalid H3-Image latent #{i+1}: {tuple(latent.shape)}")
            if vision.ndim != 4 or vision.shape[0] != 1 or vision.shape[-1] != 3:
                raise ValueError(f"Invalid image #{i+1}")
            latents.append(latent)
            visions.append(vision.to(torch.float32).div(255.0))
        identity = {"latents": latents, "visions": visions, "format": _FORMAT, "name": meta.get("identity_name", "")}
        return (identity, visions[0])


class H3ImageNativeRefModConditioning:
    """Provide all stored views to both Qwen and DiT using native MiniMax refs.

    This reuses cached VAE latents, but Qwen still encodes the reference images.
    A true tiny 8-view identity embedding requires further model research.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "identity": ("H3IMAGE_NATIVE_IDENTITY",),
            "prompt": ("STRING", {"multiline": True,
                      "default": "A cinematic portrait of the same person in <Picture 1> through <Picture 8>, natural features."}),
            "width": ("INT", {"default": 1024, "min": 32, "max": 16384, "step": 32}),
            "height": ("INT", {"default": 1024, "min": 32, "max": 16384, "step": 32}),
        }}
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "condition"
    CATEGORY = "MiniMax H3-Image / Native RefMod (experimental)"

    def condition(self, clip, identity, prompt, width, height):
        if identity.get("format") != _FORMAT:
            raise ValueError("Expected native 8-view H3-Image identity")
        visions, latents = identity["visions"], identity["latents"]
        if len(visions) != len(latents) or not (1 <= len(visions) <= 8):
            raise ValueError("Identity reference count mismatch")
        # Official H3-Image path uses images=[...] and minimax_refs in CONDITIONING.
        tokens = clip.tokenize(prompt, images=visions)
        cond = clip.encode_from_tokens_scheduled(tokens)
        blocks = [{"kind": "image", "latent_h": int(z.shape[-2]),
                   "latent_w": int(z.shape[-1]), "latent": z}
                  for z in latents]
        cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": blocks})
        canvas = torch.zeros([1,24,1,height//16,width//16],
                             device=comfy.model_management.intermediate_device())
        return (cond, {"samples": canvas})


NODE_CLASS_MAPPINGS = {
    "SaveH3ImageNativeIdentity": SaveH3ImageNativeIdentity,
    "LoadH3ImageNativeIdentity": LoadH3ImageNativeIdentity,
    "H3ImageNativeRefModConditioning": H3ImageNativeRefModConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveH3ImageNativeIdentity": "Save H3-Image Native Identity (EXPERIMENTAL)",
    "LoadH3ImageNativeIdentity": "Load H3-Image Native Identity (EXPERIMENTAL)",
    "H3ImageNativeRefModConditioning": "H3-Image Native RefMod Conditioning (EXPERIMENTAL)",
}
