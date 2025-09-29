import os
import argparse
import pathlib
import torch
import numpy as np
from typing import Optional, List
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor
from torchcodec.decoders import VideoDecoder


_VIDEO_MODEL = None
_VIDEO_PROCESSOR = None


def _ensure_video_model(model_name: str = "facebook/vjepa2-vitl-fpc64-256", device: Optional[torch.device] = None):
    global _VIDEO_MODEL, _VIDEO_PROCESSOR
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _VIDEO_MODEL is None or _VIDEO_PROCESSOR is None:
        _VIDEO_PROCESSOR = AutoVideoProcessor.from_pretrained(model_name)
        # Use float16 on CUDA where possible; fallback to float32 on CPU
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        _VIDEO_MODEL = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(device)
        _VIDEO_MODEL.eval()
    return _VIDEO_MODEL, _VIDEO_PROCESSOR


def _frame_indices(vr: VideoDecoder, num_frames: int = 32) -> np.ndarray:
    total = None
    for attr in ("frame_count", "num_frames", "__len__"):
        if hasattr(vr, attr):
            try:
                total = int(getattr(vr, attr)()) if callable(getattr(vr, attr)) else int(getattr(vr, attr))
            except Exception:
                total = None
        if total is not None and total > 0:
            break
    if not total or total <= 0:
        # Best-effort fallback
        return np.arange(num_frames, dtype=int)
    return np.linspace(0, max(0, total - 1), num=num_frames, dtype=int)


def extract_video_embedding(video_path: str, model_name: str = "facebook/vjepa2-vitl-fpc64-256", device: Optional[torch.device] = None, num_frames: int = 32) -> torch.Tensor:
    """
    Decode frames, run V-JEPA 2, mean-pool tokens to a fixed embedding [D].
    """
    model, processor = _ensure_video_model(model_name=model_name, device=device)

    vr = VideoDecoder(video_path)
    idx = _frame_indices(vr, num_frames=num_frames)
    frames = vr.get_frames_at(indices=idx).data  # (T, C, H, W) tensor/ndarray

    inputs = processor(frames, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model(**inputs).last_hidden_state  # [B, N_tokens, D]
    emb = tokens.mean(dim=1).squeeze(0).detach().cpu()  # [D]
    return emb


def _iter_video_files(root: str) -> List[str]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if pathlib.Path(fn).suffix.lower() in exts:
                files.append(os.path.join(dirpath, fn))
    return files


def main():
    parser = argparse.ArgumentParser(description="Extract mean-pooled V-JEPA2 video embeddings")
    parser.add_argument("input", help="Video file or directory root")
    parser.add_argument("--out_dir", default="video_emb", help="Output directory for .pt embeddings")
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256", help="V-JEPA2 model name")
    parser.add_argument("--num_frames", type=int, default=32, help="Frames sampled per clip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_video_model(args.model, device=device)

    in_path = pathlib.Path(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    if in_path.is_file():
        emb = extract_video_embedding(str(in_path), model_name=args.model, device=device, num_frames=args.num_frames)
        out_path = pathlib.Path(args.out_dir) / (in_path.stem + ".pt")
        torch.save(emb, out_path)
        print(f"Saved video embedding to {out_path} with shape {tuple(emb.shape)}")
    else:
        files = _iter_video_files(str(in_path))
        for fp in tqdm(files, desc="Video embeddings", dynamic_ncols=True):
            rel = pathlib.Path(fp).relative_to(in_path)
            target_dir = pathlib.Path(args.out_dir) / rel.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / (pathlib.Path(fp).stem + ".pt")
            try:
                emb = extract_video_embedding(fp, model_name=args.model, device=device, num_frames=args.num_frames)
                torch.save(emb, out_path)
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Failed {fp}: {e}")


if __name__ == "__main__":
    main()
