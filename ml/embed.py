"""Шаг 2: «отпечатки» вырезок замороженной моделью SigLIP2 (модель не обучается).

Читает crops.jsonl из extract_crops.py, считает эмбеддинги изображений и сохраняет:
    embeddings_<модель>.npy  — float16, по строке на вырезку (L2-нормированы)
    embeddings_<модель>.ids  — crop_id в том же порядке
Возобновляется: уже посчитанные crop_id пропускаются.

    python ml/embed.py --output /workspace/ml --model google/siglip2-so400m-patch16-384
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class Crops(Dataset):
    def __init__(self, root: Path, rows: list[dict], processor):
        self.root, self.rows, self.processor = root, rows, processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = Image.open(self.root / self.rows[i]["crop"]).convert("RGB")
        px = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return px, self.rows[i]["crop_id"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="google/siglip2-so400m-patch16-384")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from transformers import AutoModel, AutoProcessor

    out = Path(args.output)
    tag = args.model.split("/")[-1]
    emb_file, ids_file = out / f"embeddings_{tag}.npy", out / f"embeddings_{tag}.ids"
    rows = [json.loads(line) for line in (out / "crops.jsonl").read_text(encoding="utf-8").splitlines() if line]
    old_ids = ids_file.read_text(encoding="utf-8").split("\n") if ids_file.exists() else []
    old_ids = [x for x in old_ids if x]
    old_emb = np.load(emb_file) if emb_file.exists() and old_ids else np.zeros((0, 0), np.float16)
    seen = set(old_ids)
    rows = [r for r in rows if r["crop_id"] not in seen]
    print(f"вырезок к расчёту: {len(rows)} (уже есть: {len(old_ids)})", flush=True)
    if not rows:
        return 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    loader = DataLoader(Crops(out, rows, processor.image_processor), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=device == "cuda")

    chunks, ids, t0 = [], [], time.time()
    with torch.inference_mode():
        for b, (px, batch_ids) in enumerate(loader, 1):
            feats = model.get_image_features(pixel_values=px.to(device, dtype))
            if not torch.is_tensor(feats):  # transformers 5.x возвращает объект, а не тензор
                feats = feats.pooler_output
            feats = torch.nn.functional.normalize(feats.float(), dim=-1)
            chunks.append(feats.cpu().numpy().astype(np.float16))
            ids.extend(batch_ids)
            if b % 20 == 0:
                done = len(ids)
                print(f"{done}/{len(rows)} | {done / (time.time() - t0):.1f} вырезок/с", flush=True)

    new = np.concatenate(chunks)
    emb = new if old_emb.size == 0 else np.concatenate([old_emb, new])
    np.save(emb_file, emb)
    ids_file.write_text("\n".join(old_ids + ids) + "\n", encoding="utf-8")
    print(f"готово: {emb.shape} -> {emb_file} за {(time.time() - t0) / 60:.1f} мин | устройство {device}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
