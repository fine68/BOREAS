from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pets_dc.model import BOREASEnsemble


GROUPS = {
    "cold_T": list(range(0, 12)),
    "hot_T": list(range(12, 14)),
    "cold_H": list(range(14, 26)),
    "hot_H": list(range(26, 28)),
    "LAT": list(range(28, 37)),
    "EAT": list(range(37, 46)),
}


def load_checkpoint(path: Path, device: torch.device):
    saved = torch.load(path, map_location=device, weights_only=False)
    args = saved["args"]
    model = BOREASEnsemble(
        n_ensemble=int(args.get("n_ensemble", 5)),
        d_model=int(args.get("d_model", 128)),
        d_ext=int(args.get("d_ext", 64)),
        gnn_layers=int(args.get("gnn_layers", 2)),
        n_heads=int(args.get("n_heads", 4)),
        gru_hidden=int(args.get("gru_hidden", 256)),
        head_hidden=int(args.get("head_hidden", 256)),
    ).to(device)
    model.load_state_dict(saved["ensemble_state_dict"])
    model.eval()
    return model, saved


def summarize(prediction: np.ndarray, target: np.ndarray, std: np.ndarray, horizons):
    result = {}
    for horizon in horizons:
        error = prediction[:, :horizon] - target[:, :horizon]
        standardized = error / std[None, None, :]
        result[str(horizon)] = {
            "cs_pRMSE": float(np.sqrt(np.mean(standardized ** 2))),
            "cs_MAE": float(np.mean(np.abs(standardized))),
            "group": {},
        }
        for name, channels in GROUPS.items():
            values = error[:, :, channels]
            result[str(horizon)]["group"][name] = {
                "RMSE": float(np.sqrt(np.mean(values ** 2))),
                "MAE": float(np.mean(np.abs(values))),
                "MSE": float(np.mean(values ** 2)),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 15])
    parser.add_argument("--burnin", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.burnin < 0 or any(h <= 0 for h in args.horizons):
        raise ValueError("burnin must be nonnegative and horizons must be positive")
    max_horizon = max(args.horizons)
    starts = np.load(args.dataset_dir / f"test_valid_starts_L{args.burnin + max_horizon}.npy")
    if args.max_windows:
        starts = starts[:args.max_windows]
    data = np.load(args.dataset_dir / "test.npz")
    norm = np.load(args.dataset_dir / "normalizer_strict.npz")
    state = data["state"]
    state_next = data["state_next"]
    action = data["action"]
    ext = data["ext"]
    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    predictions = np.empty((len(starts), max_horizon, 46), dtype=np.float32)
    targets = np.empty_like(predictions)

    with torch.inference_mode():
        for offset in range(0, len(starts), args.batch_size):
            batch_starts = starts[offset:offset + args.batch_size]
            indices = batch_starts[:, None] + np.arange(args.burnin + max_horizon)[None, :]
            batch_state = torch.as_tensor(state[indices], device=device)
            batch_next = torch.as_tensor(state_next[indices], device=device)
            batch_action = torch.as_tensor(action[indices], device=device)
            batch_ext = torch.as_tensor(ext[indices], device=device)
            s_mean = torch.as_tensor(norm["s_mean"], device=device)
            s_std = torch.as_tensor(norm["s_std"], device=device)
            a_mean = torch.as_tensor(norm["a_mean"], device=device)
            a_std = torch.as_tensor(norm["a_std"], device=device)
            e_mean = torch.as_tensor(norm["e_mean"], device=device)
            e_std = torch.as_tensor(norm["e_std"], device=device)
            hidden = model.init_hidden(len(batch_starts), device)
            previous_action = torch.zeros(len(batch_starts), 18, device=device)
            for step in range(args.burnin):
                state_n = (batch_state[:, step] - s_mean) / s_std
                action_n = (batch_action[:, step] - a_mean) / a_std
                ext_n = (batch_ext[:, step] - e_mean) / e_std
                _, _, hidden = model.step(state_n, previous_action, action_n, ext_n, hidden)
                previous_action = action_n
            current = batch_state[:, args.burnin]
            for rollout_step in range(max_horizon):
                tuple_step = args.burnin + rollout_step
                state_n = (current - s_mean) / s_std
                action_n = (batch_action[:, tuple_step] - a_mean) / a_std
                ext_n = (batch_ext[:, tuple_step] - e_mean) / e_std
                means, _, hidden = model.step(state_n, previous_action, action_n, ext_n, hidden)
                predicted = current[:, :46] + means.mean(dim=0) * s_std[:46]
                predictions[offset:offset + len(batch_starts), rollout_step] = predicted.cpu().numpy()
                targets[offset:offset + len(batch_starts), rollout_step] = batch_next[:, tuple_step, :46].cpu().numpy()
                current = torch.cat((predicted, batch_next[:, tuple_step, 46:]), dim=1)
                previous_action = action_n

    metrics = summarize(predictions, targets, norm["s_std"][:46], sorted(set(args.horizons)))
    output = {
        "protocol": {
            "burnin": args.burnin,
            "horizons": sorted(set(args.horizons)),
            "windows": int(len(starts)),
            "future_actions": "recorded test actions",
            "future_exogenous": "recorded test loads",
            "rollout": "ensemble mean with logged actuator bookkeeping",
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "metrics": metrics,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
