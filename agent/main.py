# -*- coding: utf-8 -*-
"""
US-JEPA + U-Net Ultrasound Denoising — FastAPI wrapper.

This is the SAME functional-dict pipeline from usjepa_unet_video_inference.py,
just wrapped in a single-image /denoise endpoint instead of a video loop.
Model is loaded ONCE at startup (not per-request).

Run locally:
    uvicorn main:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST "http://localhost:8000/denoise" \
         -F "file=@some_ultrasound_frame.jpg" \
         -o response.json
"""

import os
import io
import math
import base64
import time

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse


DECODER_CHECKPOINT_PATH = os.environ.get(
    "DECODER_CHECKPOINT_PATH", "C:/Users/User/Desktop/jepaproject/agent/models/checkpoint_best.pth"
)
PRETRAINED_ENCODER_PATH = os.environ.get(
    "PRETRAINED_ENCODER_PATH", "C:/Users/User/Desktop/jepaproject/agent/models/encoder_best.pth"
)

MODEL_CFG = {
    'img_size': 256,
    'img_channels': 1,
    'patch_size': 16,
    'embed_dim': 384,
    'encoder_depth': 8,
    'encoder_heads': 6,
    'freeze_encoder': True,
    'unet_channels': (256, 128, 64, 32),
    'decoder_dropout': 0.0,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = True

# Tiled inference (matches notebook Section 3)
TILE_SIZE = MODEL_CFG['img_size']
TILE_OVERLAP = 0.25

# Blend + unsharp postprocess (matches notebook Section 4)
BLEND_ALPHA = 0.3
UNSHARP_AMOUNT = 0.18

# Overlay/UI mask (matches notebook — scanner status bar / sidebar excluded from denoising)
AUTO_DETECT = False
MARGIN_TOP_FRAC = 0.09
MARGIN_BOTTOM_FRAC = 0.02
MARGIN_LEFT_FRAC = 0.03
MARGIN_RIGHT_FRAC = 0.14


# =============================================================================
# 2. FUNCTIONAL-DICT HELPERS (verbatim from the notebook)
# =============================================================================

def move_model_dict_to_device(d, device):
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = move_model_dict_to_device(v, device)
        return d
    elif isinstance(d, list):
        return [move_model_dict_to_device(v, device) for v in d]
    elif isinstance(d, nn.Module):
        return d.to(device)
    elif isinstance(d, nn.Parameter):
        d.data = d.data.to(device)
        return d
    else:
        return d


def load_functional_state(saved, target):
    if isinstance(target, dict):
        for k in target:
            load_functional_state(saved[k], target[k])
    elif isinstance(target, list):
        for s, t in zip(saved, target):
            load_functional_state(s, t)
    elif isinstance(target, nn.Module):
        target.load_state_dict(saved)
    elif isinstance(target, nn.Parameter):
        target.data.copy_(saved.to(target.device))


def set_module_tree_eval_or_train(d, train: bool):
    if isinstance(d, dict):
        for v in d.values():
            set_module_tree_eval_or_train(v, train)
    elif isinstance(d, list):
        for v in d:
            set_module_tree_eval_or_train(v, train)
    elif isinstance(d, nn.Module):
        d.train(train)


def gather_params(d):
    params = []
    if isinstance(d, dict):
        for v in d.values():
            params += gather_params(v)
    elif isinstance(d, list):
        for v in d:
            params += gather_params(v)
    elif isinstance(d, nn.Module):
        params += [p for p in d.parameters()]
    elif isinstance(d, nn.Parameter):
        params.append(d)
    return params


# =============================================================================
# 3. MODEL DEFINITION (verbatim from the notebook)
# =============================================================================

def init_patch_embed(in_channels=1, embed_dim=384, patch_size=16):
    return nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)


def patch_embed_forward(x, proj_layer):
    x = proj_layer(x)
    return rearrange(x, 'b d h w -> b (h w) d')


def init_transformer_block(embed_dim, n_heads, mlp_ratio=4.0, dropout=0.0):
    hidden = int(embed_dim * mlp_ratio)
    return nn.ModuleDict({
        'norm1': nn.LayerNorm(embed_dim),
        'attn': nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True),
        'norm2': nn.LayerNorm(embed_dim),
        'mlp': nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )
    })


def transformer_block_forward(x, layers):
    n = layers['norm1'](x)
    attn_out, _ = layers['attn'](n, n, n, need_weights=False)
    x = x + attn_out
    x = x + layers['mlp'](layers['norm2'](x))
    return x


def init_vit_encoder(img_size=224, patch_size=16, in_channels=1, embed_dim=384, depth=8, n_heads=6):
    patch_embed = init_patch_embed(in_channels, embed_dim, patch_size)
    grid_size = img_size // patch_size
    n_patches = grid_size ** 2
    cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
    blocks = [init_transformer_block(embed_dim, n_heads) for _ in range(depth)]
    norm = nn.LayerNorm(embed_dim)
    return {
        'patch_embed': patch_embed, 'cls_token': cls_token, 'pos_embed': pos_embed,
        'blocks': blocks, 'norm': norm, 'grid_size': grid_size, 'n_patches': n_patches,
    }


def vit_encoder_forward(x, encoder_layers, context_mask=None):
    tokens = patch_embed_forward(x, encoder_layers['patch_embed'])
    B, N, D = tokens.shape
    cls = encoder_layers['cls_token'].expand(B, -1, -1)
    tokens = torch.cat([cls, tokens], dim=1)
    tokens = tokens + encoder_layers['pos_embed']

    if context_mask is not None:
        keep = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), context_mask], dim=1)
        tokens_list = [tokens[b][keep[b]] for b in range(B)]
        max_len = max(t.shape[0] for t in tokens_list)
        padded = torch.zeros(B, max_len, D, device=x.device, dtype=tokens.dtype)
        for b, t in enumerate(tokens_list):
            padded[b, :t.shape[0]] = t
        tokens = padded

    for block in encoder_layers['blocks']:
        tokens = transformer_block_forward(tokens, block)
    return encoder_layers['norm'](tokens)


def init_residual_conv_block(channels, dropout=0.1):
    return {
        'conv1': nn.Conv2d(channels, channels, 3, padding=1),
        'norm1': nn.GroupNorm(8, channels),
        'conv2': nn.Conv2d(channels, channels, 3, padding=1),
        'norm2': nn.GroupNorm(8, channels),
        'drop': nn.Dropout2d(dropout),
    }


def residual_conv_block_forward(x, layers):
    residual = x
    x = F.gelu(layers['norm1'](layers['conv1'](x)))
    x = layers['drop'](x)
    x = layers['norm2'](layers['conv2'](x))
    return F.gelu(x + residual)


def init_unet_up_block(in_ch, skip_ch, out_ch, dropout=0.1):
    return {
        'up': nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
        'skip_proj': nn.Conv2d(skip_ch, out_ch, 1),
        'fuse': nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
        'norm': nn.GroupNorm(8, out_ch),
        'refine': init_residual_conv_block(out_ch, dropout),
    }


def unet_up_block_forward(x, skip, layers):
    x = layers['up'](x)
    skip = layers['skip_proj'](skip)
    if skip.shape[-2:] != x.shape[-2:]:
        skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
    x = F.gelu(layers['norm'](layers['fuse'](torch.cat([x, skip], dim=1))))
    return residual_conv_block_forward(x, layers['refine'])


def init_model_c(cfg):
    channels = cfg['unet_channels']
    n_up = int(math.log2(cfg['patch_size']))
    in_conv = nn.Conv2d(cfg['embed_dim'], channels[0], 3, padding=1)
    up_blocks = []
    ch = channels[0]
    for i in range(n_up):
        next_ch = channels[min(i + 1, len(channels) - 1)]
        up_blocks.append(init_unet_up_block(ch, cfg['img_channels'], next_ch, dropout=cfg.get('decoder_dropout', 0.0)))
        ch = next_ch
    out_conv = nn.Sequential(
        nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
        nn.Conv2d(ch, cfg['img_channels'], 3, padding=1),
    )
    grid_size = cfg['img_size'] // cfg['patch_size']
    return {'in_conv': in_conv, 'up_blocks': up_blocks, 'out_conv': out_conv, 'grid_size': grid_size}


def model_c_forward(noisy_img, encoder_dict, model_c_dict):
    tokens = vit_encoder_forward(noisy_img, encoder_dict, context_mask=None)
    patch_tokens = tokens[:, 1:]
    grid_size = model_c_dict['grid_size']
    x = rearrange(patch_tokens, 'b (h w) d -> b d h w', h=grid_size, w=grid_size)
    h = model_c_dict['in_conv'](x)
    for block in model_c_dict['up_blocks']:
        target_size = (h.shape[-2] * 2, h.shape[-1] * 2)
        skip = F.interpolate(noisy_img, size=target_size, mode='bilinear', align_corners=False)
        h = unet_up_block_forward(h, skip, block)
    residual = model_c_dict['out_conv'](h)
    upsampled_input = F.interpolate(noisy_img, size=residual.shape[-2:], mode='bilinear', align_corners=False)
    return upsampled_input + residual


def build_model(model_cfg: dict = MODEL_CFG) -> dict:
    encoder = init_vit_encoder(
        img_size=model_cfg['img_size'],
        patch_size=model_cfg['patch_size'],
        in_channels=model_cfg['img_channels'],
        embed_dim=model_cfg['embed_dim'],
        depth=model_cfg['encoder_depth'],
        n_heads=model_cfg['encoder_heads'],
    )
    decoder = init_model_c(model_cfg)
    return {'encoder': encoder, 'decoder': decoder}


def load_model(decoder_checkpoint_path: str, pretrained_encoder_path: str = None,
                model_cfg: dict = MODEL_CFG, device: str = DEVICE) -> dict:
    if not os.path.isfile(decoder_checkpoint_path):
        raise FileNotFoundError(f"Decoder checkpoint not found at: {decoder_checkpoint_path}")

    model = build_model(model_cfg)
    model = move_model_dict_to_device(model, device)

    checkpoint = torch.load(decoder_checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError(
            "Expected a checkpoint dict with a 'model' key. "
            f"Got keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
        )

    load_functional_state(checkpoint['model'], model['decoder'])
    print(f"Decoder weights loaded from '{decoder_checkpoint_path}' (epoch {checkpoint.get('epoch', '?')})")

    encoder_state = checkpoint.get('encoder', None)
    if encoder_state is not None:
        load_functional_state(encoder_state, model['encoder'])
        print("Encoder weights loaded from the decoder checkpoint (fine-tuned encoder).")
    else:
        if not pretrained_encoder_path or not os.path.isfile(pretrained_encoder_path):
            raise FileNotFoundError(
                "Decoder checkpoint has no 'encoder' state (freeze_encoder=True), "
                "so PRETRAINED_ENCODER_PATH must point to a valid pretrained US-JEPA encoder checkpoint."
            )
        encoder_saved = torch.load(pretrained_encoder_path, map_location=device)
        load_functional_state(encoder_saved, model['encoder'])
        print(f"Encoder weights loaded from pretrained checkpoint: '{pretrained_encoder_path}'")

    set_module_tree_eval_or_train(model, train=False)
    n_params = sum(p.numel() for p in gather_params(model))
    print(f"Model ready on {device} | total params: {n_params:,}")
    return model


# =============================================================================
# 4. TILED, MASKED INFERENCE + BLEND/UNSHARP POSTPROCESS (verbatim from notebook)
# =============================================================================

def _hann_window_2d(size):
    w1d = np.hanning(size)
    w1d = np.clip(w1d, 1e-3, None)
    return np.outer(w1d, w1d).astype(np.float32)


_TILE_WINDOW = _hann_window_2d(TILE_SIZE)


def _get_tile_origins(length, tile, stride):
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, stride))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


@torch.no_grad()
def denoise_frame_tiled(frame_bgr, model, device, tile_size=TILE_SIZE, overlap=TILE_OVERLAP, use_amp=True):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)
    if pad_h or pad_w:
        gray = cv2.copyMakeBorder(gray, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    Hp, Wp = gray.shape

    stride = max(1, int(tile_size * (1 - overlap)))
    ys = _get_tile_origins(Hp, tile_size, stride)
    xs = _get_tile_origins(Wp, tile_size, stride)

    acc = np.zeros((Hp, Wp), dtype=np.float32)
    weight = np.zeros((Hp, Wp), dtype=np.float32)

    for y in ys:
        for x in xs:
            tile = gray[y:y + tile_size, x:x + tile_size]
            tile01 = tile.astype(np.float32) / 255.0
            tensor = torch.from_numpy(tile01).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.autocast(device_type="cuda", enabled=(use_amp and device == "cuda")):
                out = model_c_forward(tensor, model['encoder'], model['decoder'])
            out = torch.clamp(out, 0.0, 1.0).squeeze(0).squeeze(0).float().cpu().numpy()

            acc[y:y + tile_size, x:x + tile_size] += out * _TILE_WINDOW
            weight[y:y + tile_size, x:x + tile_size] += _TILE_WINDOW

    stitched = acc / np.maximum(weight, 1e-6)
    stitched = stitched[:H, :W]
    return stitched


def build_overlay_mask(frame_bgr):
    """255 = anatomy region to denoise, 0 = scanner overlay/UI to leave untouched."""
    H, W = frame_bgr.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    if not AUTO_DETECT:
        y0 = int(H * MARGIN_TOP_FRAC)
        y1 = int(H * (1 - MARGIN_BOTTOM_FRAC))
        x0 = int(W * MARGIN_LEFT_FRAC)
        x1 = int(W * (1 - MARGIN_RIGHT_FRAC))
        mask[y0:y1, x0:x1] = 255
        return mask

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if n <= 1:
        mask[:] = 255
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask[labels == largest] = 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return mask


@torch.no_grad()
def denoise_frame_tiled_masked(frame_bgr, model, device, tile_size=None, overlap=None, use_amp=True):
    tile_size = tile_size or TILE_SIZE
    overlap = overlap if overlap is not None else TILE_OVERLAP

    denoised_full = denoise_frame_tiled(frame_bgr, model, device, tile_size=tile_size,
                                         overlap=overlap, use_amp=use_amp)
    mask = build_overlay_mask(frame_bgr)
    orig_gray01 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    mask01 = mask.astype(np.float32) / 255.0
    out = denoised_full * mask01 + orig_gray01 * (1 - mask01)
    return out.astype(np.float32)


def unsharp_mask(gray01, amount=0.5, radius=3):
    blurred = cv2.GaussianBlur(gray01, (0, 0), sigmaX=radius)
    sharpened = gray01 + amount * (gray01 - blurred)
    return np.clip(sharpened, 0.0, 1.0)


def postprocess_blend(denoised01, original_bgr, blend_alpha=BLEND_ALPHA, unsharp_amount=UNSHARP_AMOUNT):
    orig_gray01 = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blended = (1 - blend_alpha) * denoised01 + blend_alpha * orig_gray01
    if unsharp_amount > 0:
        blended = unsharp_mask(blended, amount=unsharp_amount)
    img_uint8 = (np.clip(blended, 0, 1) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)


# =============================================================================
# 5. FASTAPI APP
# =============================================================================

app = FastAPI(title="US-JEPA Denoising API")

MODEL = None  # loaded once in the startup event


@app.on_event("startup")
def _load_model_on_startup():
    global MODEL
    print(f"Loading model on {DEVICE} ...")
    MODEL = load_model(
        decoder_checkpoint_path=DECODER_CHECKPOINT_PATH,
        pretrained_encoder_path=PRETRAINED_ENCODER_PATH,
        model_cfg=MODEL_CFG,
        device=DEVICE,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "device": DEVICE}


def _run_denoise_pipeline(contents: bytes):
    """
    Shared logic: decode -> run model -> postprocess -> compute metrics.
    Returns (final_bgr, metrics_dict). Raises HTTPException on bad input.
    Used by both /denoise (full, with image) and /denoise_summary (metrics only).
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    file_bytes = np.frombuffer(contents, dtype=np.uint8)
    frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Send a valid image file.")

    start = time.time()
    denoised01 = denoise_frame_tiled_masked(frame_bgr, MODEL, DEVICE)
    final_bgr = postprocess_blend(denoised01, frame_bgr)
    elapsed = time.time() - start

    final_gray01 = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    noise_proxy = float(cv2.Laplacian(final_gray01, cv2.CV_32F, ksize=3).var())
    gx = cv2.Sobel(final_gray01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(final_gray01, cv2.CV_32F, 0, 1, ksize=3)
    edge_energy = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    metrics = {
        "noise_proxy": noise_proxy,
        "edge_energy": edge_energy,
        "inference_time_sec": round(elapsed, 3),
    }
    return final_bgr, metrics

import requests
from pydantic import BaseModel

class ImageUrlPayload(BaseModel):
    image_url: str


import os
from urllib.parse import urlparse

def _fetch_image_bytes(image_url: str) -> bytes:
    image_url = image_url.strip().lstrip("=")
    parsed = urlparse(image_url)
    # If this URL points to our own /files/ storage, read directly from disk
    if parsed.path.startswith("/files/"):
        fname = os.path.basename(parsed.path)
        local_path = os.path.join("storage", fname)
        if os.path.isfile(local_path):
            with open(local_path, "rb") as f:
                return f.read()
    # Otherwise, fetch remotely
    try:
        r = requests.get(image_url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image_url: {e}")
    return r.content


@app.post("/denoise")
async def denoise(payload: ImageUrlPayload):
    contents = _fetch_image_bytes(payload.image_url)
    final_bgr, metrics = _run_denoise_pipeline(contents)

    ok, buf = cv2.imencode(".png", final_bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode output image.")

    fname = f"{uuid.uuid4()}.png"
    with open(f"storage/{fname}", "wb") as f:
        f.write(buf.tobytes())

    return JSONResponse({
        "status": "success",
        "denoised_image_url": f"http://host.docker.internal:8000/files/{fname}",
        "metrics": metrics,
    })


@app.post("/denoise_summary")
async def denoise_summary(payload: ImageUrlPayload):
    contents = _fetch_image_bytes(payload.image_url)
    _final_bgr, metrics = _run_denoise_pipeline(contents)

    return JSONResponse({
        "status": "success",
        "metrics": metrics,
        "note": "Signal quality improvement metrics only — not a diagnostic assessment. "
                "Image was processed but is not included in this response.",
    })
import uuid, os
from fastapi.staticfiles import StaticFiles

os.makedirs("storage", exist_ok=True)
app.mount("/files", StaticFiles(directory="storage"), name="files")

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1]
    fname = f"{uuid.uuid4()}.{ext}"
    path = f"storage/{fname}"
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"image_url": f"http://host.docker.internal:8000/files/{fname}"}