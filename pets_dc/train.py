from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pets_dc.model import BOREASEnsemble
from pets_dc.utils import (
    atomic_torch_save,
    load_norm,
    load_tensor_split,
    normalized_inputs,
    sha256,
    validation_epoch,
)


OBS_DIM = 46
ACT_DIM = 18
U_DIM = 27


class InverseMemberHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.change_head = nn.Linear(hidden_dim, ACT_DIM)
        self.magnitude_head = nn.Linear(hidden_dim, ACT_DIM)

    def forward(self, features: torch.Tensor):
        hidden = self.trunk(features)
        return self.change_head(hidden), self.magnitude_head(hidden)


class InverseDynamicsEnsemble(nn.Module):
    def __init__(
        self,
        n_ensemble: int,
        gru_hidden: int,
        d_model: int,
        d_ext: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        in_dim = gru_hidden + d_model + U_DIM + d_ext
        self.heads = nn.ModuleList(
            [InverseMemberHead(in_dim, hidden_dim) for _ in range(n_ensemble)]
        )

    def forward(
        self,
        model: BOREASEnsemble,
        filtered_hidden: list[torch.Tensor],
        current_state_n: torch.Tensor,
        next_obs_n: torch.Tensor,
        ext_n: torch.Tensor,
    ):
        if next_obs_n.shape[-1] != OBS_DIM:
            raise ValueError("inverse dynamics may receive only the 46-D next observation")

        safe_next_state_n = torch.cat(
            [next_obs_n, current_state_n[:, OBS_DIM:]], dim=-1
        )
        logits, magnitudes = [], []
        for member, head, h_t in zip(model.members, self.heads, filtered_hidden):
            e_next = member.encoder(safe_next_state_n).mean(dim=1)
            e_ext = member.ext_encoder(ext_n)
            features = torch.cat(
                [h_t, e_next, current_state_n[:, OBS_DIM:], e_ext], dim=-1
            )
            member_logits, member_magnitude = head(features)
            logits.append(member_logits)
            magnitudes.append(member_magnitude)
        return torch.stack(logits), torch.stack(magnitudes)


def build_model(args, device: torch.device):
    model_kwargs = dict(
        n_ensemble=args.n_ensemble,
        d_model=args.d_model,
        d_ext=args.d_ext,
        gnn_layers=args.gnn_layers,
        n_heads=args.n_heads,
        gru_hidden=args.gru_hidden,
        head_hidden=args.head_hidden,
    )
    return BOREASEnsemble(**model_kwargs).to(device)


def action_statistics(
    train_data, device: torch.device, pos_weight_exponent: float = 1.0
):
    action = train_data["action"]
    positive = action.ne(0)
    counts = positive.sum(dim=0)
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0).flatten().tolist()
        raise ValueError(f"action channels without positive examples: {missing}")
    negative = action.shape[0] - counts
    pos_weight = (negative / counts).pow(pos_weight_exponent).to(
        device=device, dtype=torch.float32
    )
    prevalence = (counts / action.shape[0]).to(dtype=torch.float32)
    return pos_weight, prevalence


def inverse_losses(
    logits: torch.Tensor,
    predicted_magnitude: torch.Tensor,
    action_raw: torch.Tensor,
    action_std: torch.Tensor,
    pos_weight: torch.Tensor,
):
    changed = action_raw.ne(0).to(dtype=logits.dtype)
    changed_e = changed.unsqueeze(0).expand_as(logits)
    target_magnitude = (action_raw / action_std).unsqueeze(0).expand_as(
        predicted_magnitude
    )
    classification = F.binary_cross_entropy_with_logits(
        logits,
        changed_e,
        pos_weight=pos_weight,
        reduction="mean",
    )
    elementwise_magnitude = F.smooth_l1_loss(
        predicted_magnitude, target_magnitude, reduction="none", beta=1.0
    )
    positive_count = changed_e.sum().clamp_min(1.0)
    magnitude = (elementwise_magnitude * changed_e).sum() / positive_count
    return classification, magnitude


def training_loss(
    model,
    inverse_head,
    data,
    norm,
    starts,
    burnin: int,
    pos_weight,
):
    batch = starts.shape[0]
    indices = starts[:, None] + torch.arange(
        burnin + 1, device=starts.device
    )[None, :]
    hidden = model.init_hidden(batch, starts.device)
    previous_action = torch.zeros(batch, ACT_DIM, device=starts.device)
    for step in range(burnin):
        state_n, action_n, ext_n = normalized_inputs(data, norm, indices[:, step])
        _, _, hidden = model.step(state_n, previous_action, action_n, ext_n, hidden)
        previous_action = action_n

    current_index = indices[:, burnin]
    state_n, action_n, ext_n = normalized_inputs(data, norm, current_index)
    means, logvars, filtered_hidden = model.step(
        state_n, previous_action, action_n, ext_n, hidden
    )
    target_delta = (
        data["state_next"][current_index, :OBS_DIM]
        - data["state"][current_index, :OBS_DIM]
    ) / norm["s_std"][:OBS_DIM]
    forward_nll = model.gaussian_nll(means, logvars, target_delta)

    next_obs_n = (
        data["state_next"][current_index, :OBS_DIM] - norm["s_mean"][:OBS_DIM]
    ) / norm["s_std"][:OBS_DIM]
    logits, magnitude = inverse_head(
        model, filtered_hidden, state_n, next_obs_n, ext_n
    )
    inverse_bce, inverse_huber = inverse_losses(
        logits,
        magnitude,
        data["action"][current_index],
        norm["a_std"],
        pos_weight,
    )
    return forward_nll, inverse_bce, inverse_huber


def validate_inverse(
    model,
    inverse_head,
    data,
    norm,
    starts_np: np.ndarray,
    burnin: int,
    batch_size: int,
):
    model.eval()
    inverse_head.eval()
    tp = np.zeros(ACT_DIM, dtype=np.int64)
    tn = np.zeros(ACT_DIM, dtype=np.int64)
    fp = np.zeros(ACT_DIM, dtype=np.int64)
    fn = np.zeros(ACT_DIM, dtype=np.int64)
    magnitude_abs_sum = 0.0
    magnitude_count = 0

    with torch.inference_mode():
        for offset in range(0, len(starts_np), batch_size):
            starts = torch.as_tensor(
                starts_np[offset:offset + batch_size], device=data["state"].device
            )
            indices = starts[:, None] + torch.arange(
                burnin + 1, device=starts.device
            )[None, :]
            hidden = model.init_hidden(len(starts), starts.device)
            previous_action = torch.zeros(len(starts), ACT_DIM, device=starts.device)
            for step in range(burnin):
                state_n, action_n, ext_n = normalized_inputs(
                    data, norm, indices[:, step]
                )
                _, _, hidden = model.step(
                    state_n, previous_action, action_n, ext_n, hidden
                )
                previous_action = action_n

            current_index = indices[:, burnin]
            state_n, action_n, ext_n = normalized_inputs(data, norm, current_index)
            _, _, filtered_hidden = model.step(
                state_n, previous_action, action_n, ext_n, hidden
            )
            next_obs_n = (
                data["state_next"][current_index, :OBS_DIM]
                - norm["s_mean"][:OBS_DIM]
            ) / norm["s_std"][:OBS_DIM]
            logits, predicted_magnitude = inverse_head(
                model, filtered_hidden, state_n, next_obs_n, ext_n
            )
            predicted_change = logits.sigmoid().mean(dim=0).ge(0.5)
            predicted_raw = predicted_magnitude.mean(dim=0) * norm["a_std"]
            target_raw = data["action"][current_index]
            target_change = target_raw.ne(0)

            tp += (predicted_change & target_change).sum(dim=0).cpu().numpy()
            tn += ((~predicted_change) & (~target_change)).sum(dim=0).cpu().numpy()
            fp += (predicted_change & (~target_change)).sum(dim=0).cpu().numpy()
            fn += ((~predicted_change) & target_change).sum(dim=0).cpu().numpy()
            magnitude_abs_sum += float(
                ((predicted_raw - target_raw).abs() * target_change).sum()
            )
            magnitude_count += int(target_change.sum())

    sensitivity = tp / np.maximum(tp + fn, 1)
    specificity = tn / np.maximum(tn + fp, 1)
    precision = tp.sum() / max(tp.sum() + fp.sum(), 1)
    recall = tp.sum() / max(tp.sum() + fn.sum(), 1)
    result = {
        "inverse_macro_balanced_accuracy": float(
            np.mean(0.5 * (sensitivity + specificity))
        ),
        "inverse_micro_precision": float(precision),
        "inverse_micro_recall": float(recall),
        "inverse_micro_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "inverse_nonzero_magnitude_mae_raw": float(
            magnitude_abs_sum / max(magnitude_count, 1)
        ),
        "inverse_nonzero_targets": int(magnitude_count),
    }
    model.train()
    inverse_head.train()
    return result


def checkpoint_payload(
    model,
    inverse_head,
    optimizer,
    args_dict,
    step,
    best_score,
    stale,
    provenance,
):
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "ensemble_state_dict": model.state_dict(),
        "inverse_head_state_dict": inverse_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_cs_prmse_H15": best_score,
        "stale_validations": stale,
        "args": args_dict,
        "provenance": provenance,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": cuda_rng_state,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--n-ensemble", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ext", type=int, default=64)
    parser.add_argument("--gnn-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--gru-hidden", type=int, default=256)
    parser.add_argument("--head-hidden", type=int, default=256)
    parser.add_argument("--inverse-hidden", type=int, default=256)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--val-batch-size", type=int, default=512)
    parser.add_argument("--burnin", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--early-stop-min-rel", type=float, default=1e-3)
    parser.add_argument("--inverse-loss-weight", type=float, default=0.03)
    parser.add_argument("--inverse-magnitude-weight", type=float, default=1.0)
    parser.add_argument("--inverse-pos-weight-exponent", type=float, default=1.0)
    parser.add_argument("--inverse-warmup-steps", type=int, default=1000)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.burnin != 8 or args.horizon != 15:
        raise ValueError("strict v2 evaluation requires burnin=8 and horizon=15")
    if args.inverse_loss_weight < 0:
        raise ValueError("inverse loss weight must be nonnegative")
    if not 0.0 <= args.inverse_pos_weight_exponent <= 1.0:
        raise ValueError("inverse positive-weight exponent must be in [0, 1]")
    args.logdir.mkdir(parents=True, exist_ok=True)
    args_dict = {
        key.replace("-", "_"): (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    (args.logdir / "args.json").write_text(
        json.dumps(args_dict, indent=2), encoding="utf-8"
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    train_data = load_tensor_split(args.dataset_dir, "train", device)
    val_data = load_tensor_split(args.dataset_dir, "val", device)
    norm = load_norm(args.dataset_dir, device)
    train_starts_file = args.dataset_dir / "train_valid_starts_L9.npy"
    val_starts_file = args.dataset_dir / "val_valid_starts_L23.npy"
    train_starts_np = np.load(train_starts_file)
    val_starts_np = np.load(val_starts_file)
    if args.max_val_windows:
        val_starts_np = val_starts_np[:args.max_val_windows]
    train_starts = torch.as_tensor(train_starts_np, device=device)
    pos_weight, action_prevalence = action_statistics(
        train_data, device, args.inverse_pos_weight_exponent
    )

    model = build_model(args, device)
    inverse_head = InverseDynamicsEnsemble(
        args.n_ensemble,
        args.gru_hidden,
        args.d_model,
        args.d_ext,
        args.inverse_hidden,
    ).to(device)
    saved = None
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved.get("provenance", {}).get("dataset_manifest_sha256") != sha256(
            args.dataset_dir / "manifest.json"
        ):
            raise ValueError("resume checkpoint and strict dataset do not match")
        model.load_state_dict(saved["ensemble_state_dict"])
        if "inverse_head_state_dict" not in saved:
            raise ValueError("checkpoint has no inverse-dynamics head")
        inverse_head.load_state_dict(saved["inverse_head_state_dict"])
        random.setstate(saved["rng_state"]["python"])
        np.random.set_state(saved["rng_state"]["numpy"])
        torch.set_rng_state(saved["rng_state"]["torch_cpu"].cpu())
        if torch.cuda.is_available() and saved["rng_state"]["torch_cuda"]:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in saved["rng_state"]["torch_cuda"]]
            )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8
    )
    optimizer.add_param_group({"params": inverse_head.parameters()})
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer_state_dict"])

    model_source = Path(__import__(model.__module__, fromlist=["x"]).__file__).resolve()
    helper_source = Path(__import__(validation_epoch.__module__, fromlist=["x"]).__file__).resolve()
    provenance = {
        "dataset_version": "v2_strict_tuple_native",
        "dataset_manifest_sha256": sha256(args.dataset_dir / "manifest.json"),
        "normalizer_sha256": sha256(args.dataset_dir / "normalizer_strict.npz"),
        "train_npz_sha256": sha256(args.dataset_dir / "train.npz"),
        "val_npz_sha256": sha256(args.dataset_dir / "val.npz"),
        "train_starts_sha256": sha256(train_starts_file),
        "val_starts_sha256": sha256(val_starts_file),
        "trainer_sha256": sha256(Path(__file__).resolve()),
        "validation_helper_sha256": sha256(helper_source),
        "model_source_sha256": sha256(model_source),
        "training_initialization": "checkpoint_resume" if args.resume else "scratch_seeded",
        "resume_checkpoint": str(args.resume) if args.resume else None,
        "resume_checkpoint_sha256": sha256(args.resume) if args.resume else None,
        "model_initialization_seed": args.seed,
        "training_target": "one-step Gaussian NLL plus leakage-safe inverse dynamics",
        "inverse_inputs": "filtered h_t, true o_{t+1}, current u_t, xi_t; excludes u_{t+1}",
        "inverse_classification": "per-channel prevalence-weighted BCE for exact nonzero action",
        "inverse_pos_weight_exponent": args.inverse_pos_weight_exponent,
        "inverse_magnitude": "conditional normalized smooth-L1 on nonzero actions only",
        "inverse_loss_weight": args.inverse_loss_weight,
        "inverse_magnitude_weight": args.inverse_magnitude_weight,
        "inverse_warmup_steps": args.inverse_warmup_steps,
        "action_nonzero_prevalence": action_prevalence.cpu().tolist(),
        "action_positive_weights": pos_weight.cpu().tolist(),
        "train_window_tuple_count": 9,
        "validation_window_tuple_count": 23,
        "checkpoint_selection": "minimum fixed-window pooled normalized RMSE over steps 1:15 and 46 equal-weight channels; inverse metrics excluded",
        "validation_windows": int(len(val_starts_np)),
    }

    step = int(saved["step"]) if saved is not None else 0
    initial_step = step
    best_score = (
        float(saved.get("best_val_cs_prmse_H15", saved.get("best_val_normalized_rmse", float("inf"))))
        if saved is not None
        else float("inf")
    )
    stale = int(saved.get("stale_validations", 0)) if saved is not None else 0
    history = []
    if saved is not None:
        atomic_torch_save(
            checkpoint_payload(
                model,
                inverse_head,
                optimizer,
                args_dict,
                step,
                best_score,
                stale,
                provenance,
            ),
            args.logdir / "best.pt",
        )

    print(
        f"[data] train tuples={len(train_data['state'])} valid L9={len(train_starts_np)} "
        f"val tuples={len(val_data['state'])} fixed L23={len(val_starts_np)}"
    )
    print(
        f"[model] forward_params={sum(p.numel() for p in model.parameters()) / 1e6:.3f}M "
        f"inverse_params={sum(p.numel() for p in inverse_head.parameters()) / 1e6:.3f}M"
    )
    print(f"[action prevalence] {action_prevalence.cpu().tolist()}")
    started = time.time()
    while step < args.steps:
        step += 1
        sampled_positions = torch.randint(
            0, len(train_starts), (args.batch_size,), device=device
        )
        sampled_starts = train_starts[sampled_positions]
        nll, inverse_bce, inverse_huber = training_loss(
            model,
            inverse_head,
            train_data,
            norm,
            sampled_starts,
            args.burnin,
            pos_weight,
        )
        update_step = step - initial_step
        warmup_fraction = (
            min(1.0, update_step / args.inverse_warmup_steps)
            if args.inverse_warmup_steps > 0
            else 1.0
        )
        inverse_weight = args.inverse_loss_weight * warmup_fraction
        inverse_total = inverse_bce + args.inverse_magnitude_weight * inverse_huber
        bound = model.logvar_bound_penalty(weight=0.01)
        loss = nll + inverse_weight * inverse_total + bound

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        all_parameters = list(model.parameters()) + list(inverse_head.parameters())
        grad_norm = torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=100.0)
        optimizer.step()

        if step == initial_step + 1 or step % args.log_every == 0:
            item = {
                "step": step,
                "nll": float(nll.detach()),
                "inverse_bce": float(inverse_bce.detach()),
                "inverse_huber": float(inverse_huber.detach()),
                "inverse_weight": inverse_weight,
                "bound": float(bound.detach()),
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "steps_per_second": update_step / max(time.time() - started, 1e-9),
            }
            history.append(item)
            print(
                f"step {step:6d} nll={item['nll']:+.5f} "
                f"inv_bce={item['inverse_bce']:.5f} inv_huber={item['inverse_huber']:.5f} "
                f"iw={item['inverse_weight']:.4f} bound={item['bound']:+.5f} "
                f"loss={item['loss']:+.5f} grad={item['grad_norm']:.3f} "
                f"sps={item['steps_per_second']:.2f}"
            )

        if step % args.val_every == 0 or step == args.steps:
            val_started = time.time()
            metrics = validation_epoch(
                model,
                val_data,
                norm,
                val_starts_np,
                args.burnin,
                args.horizon,
                args.val_batch_size,
            )
            metrics.update(
                validate_inverse(
                    model,
                    inverse_head,
                    val_data,
                    norm,
                    val_starts_np,
                    args.burnin,
                    args.val_batch_size,
                )
            )
            metrics.update({"step": step, "validation_seconds": time.time() - val_started})
            history.append(metrics)
            score = metrics["val_cs_prmse_H15"]
            print(
                f"  [val@{step}] pooled_nrmse="
                f"{metrics['val_cs_prmse_H1']:.6f}/"
                f"{metrics['val_cs_prmse_H5']:.6f}/"
                f"{metrics['val_cs_prmse_H15']:.6f} "
                f"LAT15={metrics['val_LAT_rmse_H15']:.4f} "
                f"EAT15={metrics['val_EAT_rmse_H15']:.4f} "
                f"inv_bacc={metrics['inverse_macro_balanced_accuracy']:.4f} "
                f"inv_f1={metrics['inverse_micro_f1']:.4f} "
                f"seconds={metrics['validation_seconds']:.1f}"
            )
            previous_best = best_score
            if score < best_score:
                best_score = score
                atomic_torch_save(
                    checkpoint_payload(
                        model,
                        inverse_head,
                        optimizer,
                        args_dict,
                        step,
                        best_score,
                        stale,
                        provenance,
                    ),
                    args.logdir / "best.pt",
                )
                print(f"  [checkpoint] best normalized RMSE={best_score:.6f}")

            if score < previous_best * (1.0 - args.early_stop_min_rel):
                stale = 0
            else:
                stale += 1
            (args.logdir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8"
            )
            if stale >= args.patience:
                print(f"  [early-stop] {stale} validations without improvement")
                break

        if step % args.checkpoint_every == 0:
            atomic_torch_save(
                checkpoint_payload(
                    model,
                    inverse_head,
                    optimizer,
                    args_dict,
                    step,
                    best_score,
                    stale,
                    provenance,
                ),
                args.logdir / f"step_{step:06d}.pt",
            )

    atomic_torch_save(
        checkpoint_payload(
            model,
            inverse_head,
            optimizer,
            args_dict,
            step,
            best_score,
            stale,
            provenance,
        ),
        args.logdir / "last.pt",
    )
    (args.logdir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    summary = {
        "step": step,
        "best_val_cs_prmse_H15": best_score,
        "elapsed_seconds": time.time() - started,
        "best_checkpoint": str(args.logdir / "best.pt"),
        "inverse_loss_weight": args.inverse_loss_weight,
        "provenance": provenance,
    }
    (args.logdir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
