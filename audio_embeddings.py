import os
import argparse
import pathlib
import torch
import librosa
from typing import Optional, List
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model


_AUDIO_MODEL = None
_AUDIO_PROCESSOR = None


def _ensure_audio_model(model_name: str = "facebook/wav2vec2-base", device: Optional[torch.device] = None):
    global _AUDIO_MODEL, _AUDIO_PROCESSOR
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _AUDIO_MODEL is None or _AUDIO_PROCESSOR is None:
        _AUDIO_PROCESSOR = Wav2Vec2Processor.from_pretrained(model_name)
        _AUDIO_MODEL = Wav2Vec2Model.from_pretrained(model_name).to(device)
        _AUDIO_MODEL.eval()
    return _AUDIO_MODEL, _AUDIO_PROCESSOR


def extract_audio_embedding(audio_path: str, model_name: str = "facebook/wav2vec2-base", device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Load audio, run Wav2Vec2 model, and return a pooled embedding [D].
    """
    model, processor = _ensure_audio_model(model_name=model_name, device=device)

    speech, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        hidden = model(**inputs).last_hidden_state  # [B, T, D]

    mask = inputs.get("attention_mask")
    if mask is None:
        emb = hidden.mean(dim=1)
    else:
        mask = mask.unsqueeze(-1)  # [B, T, 1]
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    return emb.squeeze(0).detach().cpu()  # [D]


def _iter_audio_files(root: str) -> List[str]:
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if pathlib.Path(fn).suffix.lower() in exts:
                files.append(os.path.join(dirpath, fn))
    return files


def main():
    parser = argparse.ArgumentParser(description="Extract mean-pooled Wav2Vec2 embeddings")
    parser.add_argument("input", help="Audio file or directory root")
    parser.add_argument("--out_dir", default="audio_emb", help="Output directory for .pt embeddings")
    parser.add_argument("--model", default="facebook/wav2vec2-base", help="Wav2Vec2 model name")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_audio_model(args.model, device=device)

    in_path = pathlib.Path(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    if in_path.is_file():
        emb = extract_audio_embedding(str(in_path), model_name=args.model, device=device)
        out_path = pathlib.Path(args.out_dir) / (in_path.stem + ".pt")
        torch.save(emb, out_path)
        print(f"Saved audio embedding to {out_path} with shape {tuple(emb.shape)}")
    else:
        files = _iter_audio_files(str(in_path))
        for fp in tqdm(files, desc="Audio embeddings", dynamic_ncols=True):
            rel = pathlib.Path(fp).relative_to(in_path)
            target_dir = pathlib.Path(args.out_dir) / rel.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / (pathlib.Path(fp).stem + ".pt")
            try:
                emb = extract_audio_embedding(fp, model_name=args.model, device=device)
                torch.save(emb, out_path)
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Failed {fp}: {e}")


if __name__ == "__main__":
    main()
