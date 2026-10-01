from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch


GROUPS = {
    "cold_T": slice(0, 12),
    "hot_T": slice(12, 14),
    "cold_H": slice(14, 26),
    "hot_H": slice(26, 28),
    "LAT": slice(28, 37),
    "EAT": slice(37, 46),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_tensor_split(dataset_dir: Path, split: str, device: torch.device):
    source = np.load(dataset_dir / f"{split}.npz")
    return {key: torch.as_tensor(source[key], device=device) for key in (
        "state", "state_next", "action", "ext"
    )}


def load_norm(dataset_dir: Path, device: torch.device):
    source = np.load(dataset_dir / "normalizer_strict.npz")
    return {key: torch.as_tensor(source[key], device=device) for key in source.files}


def normalized_inputs(data, norm, indices):
    return (
        (data["state"][indices] - norm["s_mean"]) / norm["s_std"],
        (data["action"][indices] - norm["a_mean"]) / norm["a_std"],
        (data["ext"][indices] - norm["e_mean"]) / norm["e_std"],
    )


def validation_epoch(model, data, norm, starts_np, burnin, horizon, batch_size):
    model.eval()
    squared = np.zeros(horizon, dtype=np.float64)
    absolute = np.zeros(horizon, dtype=np.float64)
    counts = np.zeros(horizon, dtype=np.int64)
    group_squared = {name: np.zeros(horizon) for name in GROUPS}
    group_absolute = {name: np.zeros(horizon) for name in GROUPS}
    group_counts = {name: np.zeros(horizon, dtype=np.int64) for name in GROUPS}

    with torch.inference_mode():
        for offset in range(0, len(starts_np), batch_size):
            starts = torch.as_tensor(starts_np[offset:offset + batch_size], device=data["state"].device)
            indices = starts[:, None] + torch.arange(burnin + horizon, device=starts.device)[None, :]
            hidden = model.init_hidden(len(starts), starts.device)
            previous_action = torch.zeros(len(starts), 18, device=starts.device)
            for step in range(burnin):
                state_n, action_n, ext_n = normalized_inputs(data, norm, indices[:, step])
                _, _, hidden = model.step(state_n, previous_action, action_n, ext_n, hidden)
                previous_action = action_n

            current = data["state"][indices[:, burnin]].clone()
            for rollout_step in range(horizon):
                tuple_index = indices[:, burnin + rollout_step]
                state_n = (current - norm["s_mean"]) / norm["s_std"]
                action_n = (data["action"][tuple_index] - norm["a_mean"]) / norm["a_std"]
                ext_n = (data["ext"][tuple_index] - norm["e_mean"]) / norm["e_std"]
                means, _, hidden = model.step(state_n, previous_action, action_n, ext_n, hidden)
                prediction = current[:, :46] + means.mean(dim=0) * norm["s_std"][:46]
                target = data["state_next"][tuple_index, :46]
                error = prediction - target
                normalized_error = error / norm["s_std"][:46]
                squared[rollout_step] += float(normalized_error.square().sum())
                absolute[rollout_step] += float(normalized_error.abs().sum())
                counts[rollout_step] += normalized_error.numel()
                for name, channels in GROUPS.items():
                    values = error[:, channels]
                    group_squared[name][rollout_step] += float(values.square().sum())
                    group_absolute[name][rollout_step] += float(values.abs().sum())
                    group_counts[name][rollout_step] += values.numel()
                current = torch.cat((prediction, data["state_next"][tuple_index, 46:]), dim=1)
                previous_action = action_n

    result = {"n_windows": int(len(starts_np))}
    for evaluation_horizon in sorted(set((1, 5, horizon))):
        pooled = slice(0, evaluation_horizon)
        total = counts[pooled].sum()
        result[f"val_cs_prmse_H{evaluation_horizon}"] = float(np.sqrt(squared[pooled].sum() / total))
        result[f"val_cs_mae_H{evaluation_horizon}"] = float(absolute[pooled].sum() / total)
        for name in GROUPS:
            n = group_counts[name][pooled].sum()
            result[f"val_{name}_rmse_H{evaluation_horizon}"] = float(np.sqrt(group_squared[name][pooled].sum() / n))
            result[f"val_{name}_mae_H{evaluation_horizon}"] = float(group_absolute[name][pooled].sum() / n)
    result["val_cs_prmse"] = result[f"val_cs_prmse_H{horizon}"]
    model.train()
    return result
