from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


OBS_DIM = 46
STATE_DIM = 73
ACTION_DIM = 18
RAW_CONTROL = np.r_[28:37, 46:55, 64:73]
RAW_TO_MODEL = np.r_[0:28, 37:46, 55:64, 28:37, 46:55, 64:73]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_parquet(path: Path):
    table = pq.ParquetFile(path).read()
    metadata = table.schema.metadata or {}
    if b"pandas" not in metadata:
        raise ValueError(f"{path} has no pandas index metadata")
    index_name = json.loads(metadata[b"pandas"])["index_columns"][0]
    timestamps = table[index_name].to_numpy().astype("datetime64[s]").astype(np.int64)
    columns = [name for name in table.column_names if name != index_name]
    values = np.column_stack([table[name].to_numpy() for name in columns])
    return timestamps, values, columns


def valid_starts(segment: np.ndarray, length: int) -> np.ndarray:
    if len(segment) < length:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(len(segment) - length + 1, dtype=np.int64)
    return starts[segment[starts] == segment[starts + length - 1]]


def segment_ids(timestamp: np.ndarray, state: np.ndarray, state_next: np.ndarray) -> np.ndarray:
    segment = np.zeros(len(timestamp), dtype=np.int32)
    if len(timestamp) > 1:
        linked = (timestamp[1:] == timestamp[:-1] + 300) & np.all(
            state[1:] == state_next[:-1], axis=1
        )
        segment[1:] = np.cumsum(~linked)
    return segment


def action_from_states(raw_state: np.ndarray, raw_next: np.ndarray) -> np.ndarray:
    fan = raw_next[:, 46:55] - raw_state[:, 46:55]
    valve = raw_next[:, 64:73] - raw_state[:, 64:73]
    return np.concatenate((fan, valve), axis=1).astype(np.float32)


def clip_threshold(action: np.ndarray, percentile: float) -> np.ndarray:
    thresholds = []
    for block in (action[:, :9], action[:, 9:]):
        nonzero = np.abs(block[block != 0])
        thresholds.append(float(np.percentile(nonzero, percentile)) if nonzero.size else 1.0)
    return np.asarray(thresholds, dtype=np.float32)


def clip_action(action: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    clipped = action.copy()
    clipped[:, :9] = np.clip(clipped[:, :9], -thresholds[0], thresholds[0])
    clipped[:, 9:] = np.clip(clipped[:, 9:], -thresholds[1], thresholds[1])
    return clipped


def fit_normalizer(state: np.ndarray, action: np.ndarray, ext: np.ndarray):
    action_std = np.empty(ACTION_DIM, dtype=np.float32)
    for channel in range(ACTION_DIM):
        nonzero = action[:, channel][action[:, channel] != 0]
        action_std[channel] = np.abs(nonzero).std() if nonzero.size else 1.0
    return {
        "s_mean": state.mean(axis=0).astype(np.float32),
        "s_std": np.maximum(state.std(axis=0), 1e-4).astype(np.float32),
        "a_mean": np.zeros(ACTION_DIM, dtype=np.float32),
        "a_std": np.maximum(action_std, 1e-4).astype(np.float32),
        "e_mean": ext.mean(axis=0).astype(np.float32),
        "e_std": np.maximum(ext.std(axis=0), 1e-4).astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the BOREAS strict tuple dataset.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--observed-mask", type=Path)
    parser.add_argument("--winsorize-p", type=float, default=99.0)
    args = parser.parse_args()
    if not 0.0 < args.winsorize_p <= 100.0:
        raise ValueError("winsorize percentile must be in (0, 100]")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    names = ("mdp_s_t.parquet", "mdp_s_next.parquet", "mdp_a_t.parquet", "mdp_ext_t.parquet")
    paths = {name: args.source_dir / name for name in names}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing source files: " + ", ".join(missing))

    timestamp, raw_state, state_columns = read_parquet(paths[names[0]])
    next_timestamp, raw_next, next_columns = read_parquet(paths[names[1]])
    action_timestamp, raw_action, action_columns = read_parquet(paths[names[2]])
    ext_timestamp, ext, ext_columns = read_parquet(paths[names[3]])
    if raw_state.shape != (len(timestamp), STATE_DIM) or raw_next.shape != raw_state.shape:
        raise ValueError("state files must contain 73 data columns")
    if raw_action.shape != (len(timestamp), 27) or ext.shape != (len(timestamp), 8):
        raise ValueError("action and exogenous files have unexpected dimensions")
    if not (np.array_equal(timestamp, next_timestamp)
            and np.array_equal(timestamp, action_timestamp)
            and np.array_equal(timestamp, ext_timestamp)):
        raise ValueError("Parquet indexes are not aligned")
    if state_columns != next_columns or len(action_columns) != 27 or len(ext_columns) != 8:
        raise ValueError("unexpected source schema")
    if not np.array_equal(raw_action, raw_state[:, RAW_CONTROL]):
        raise ValueError("mdp_a_t does not match the actuator channels in mdp_s_t")
    if not all(np.isfinite(array).all() for array in (raw_state, raw_next, raw_action, ext)):
        raise ValueError("source data contains non-finite values")
    if len(timestamp) < 2 or np.any(np.diff(timestamp) <= 0):
        raise ValueError("timestamps must be strictly increasing")

    position = {int(value): index for index, value in enumerate(timestamp.tolist())}
    successor = np.asarray([position.get(int(value) + 300, -1) for value in timestamp], dtype=np.int64)
    has_successor = successor >= 0
    next_matches = np.zeros(len(timestamp), dtype=bool)
    next_matches[has_successor] = np.all(
        raw_next[has_successor] == raw_state[successor[has_successor]], axis=1
    )
    onoff_switch = np.any(raw_state[:, 28:37] != raw_next[:, 28:37], axis=1)
    observed_pair = np.ones(len(timestamp), dtype=bool)
    if args.observed_mask:
        mask_file = np.load(args.observed_mask, allow_pickle=False)
        mask_timestamp = mask_file["timestamp"].astype(np.int64)
        mask_observed = mask_file["observed"]
        if mask_observed.ndim != 2 or mask_observed.shape[1] != 45:
            raise ValueError("observed mask must have shape (grid_rows, 45)")
        mask_positions = {int(value): index for index, value in enumerate(mask_timestamp.tolist())}
        grid_pos = np.asarray([mask_positions.get(int(value), -1) for value in timestamp], dtype=np.int64)
        if np.any(grid_pos < 0) or np.any(grid_pos + 1 >= len(mask_timestamp)):
            raise ValueError("observed mask does not cover all transition timestamps")
        if np.any(mask_timestamp[grid_pos] != timestamp):
            raise ValueError("observed mask timestamps are not aligned")
        observed_pair = mask_observed[grid_pos].all(axis=1)
        observed_pair &= mask_observed[grid_pos + 1].all(axis=1)
    strict = has_successor & next_matches & observed_pair & ~onoff_switch

    pair_count = len(timestamp) - 1
    train_boundary = int(timestamp[int(pair_count * 0.70)])
    test_boundary = int(timestamp[int(pair_count * 0.80)])
    split_intervals = {
        "train": (int(timestamp[0]), train_boundary),
        "val": (train_boundary, test_boundary),
        "test": (test_boundary, int(timestamp[-1] + 600)),
    }

    projected = raw_state[:, RAW_TO_MODEL].astype(np.float32)
    projected_next = raw_next[:, RAW_TO_MODEL].astype(np.float32)
    action_raw = action_from_states(raw_state, raw_next)
    split_masks = {
        name: strict & (timestamp >= start) & (timestamp + 300 < stop)
        for name, (start, stop) in split_intervals.items()
    }
    threshold = clip_threshold(action_raw[split_masks["train"]], args.winsorize_p)
    action = clip_action(action_raw, threshold)
    np.savez(args.out_dir / "normalizer_strict.npz", **fit_normalizer(
        projected[split_masks["train"]], action[split_masks["train"]], ext[split_masks["train"]]
    ))

    manifest = {
        "dataset_version": "v2_strict_tuple_native",
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "rules": {
            "target": "same-row mdp_s_next verified against timestamp t+300",
            "action": "next-minus-current fan and valve readbacks",
            "excluded": "missing successor, state-chain mismatch, on/off switch, or non-finite value",
            "split": "chronological 70/10/20 boundaries with targets before the next boundary",
            "windows": "adjacent 300-second tuples with exact state chaining",
            "observed_mask": "45-channel current/next validity mask when supplied",
        },
        "source_rows": int(len(timestamp)),
        "strict_rows": int(strict.sum()),
        "onoff_switch_rows": int(onoff_switch.sum()),
        "observed_mask": str(args.observed_mask) if args.observed_mask else None,
        "boundaries": {name: {"start": start, "stop": stop} for name, (start, stop) in split_intervals.items()},
        "winsorize_p": args.winsorize_p,
        "fan_clip": float(threshold[0]),
        "valve_clip": float(threshold[1]),
        "splits": {},
    }

    for split, keep in split_masks.items():
        indices = np.flatnonzero(keep).astype(np.int64)
        split_state = projected[keep]
        split_next = projected_next[keep]
        split_action_raw = action_raw[keep]
        split_action = action[keep]
        split_ext = ext[keep].astype(np.float32)
        split_timestamp = timestamp[keep].astype(np.int64)
        split_segment = segment_ids(split_timestamp, split_state, split_next)
        reconstructed = split_state[:, 46:].copy()
        reconstructed[:, 9:18] += split_action_raw[:, :9]
        reconstructed[:, 18:27] += split_action_raw[:, 9:]
        if not np.allclose(reconstructed, split_next[:, 46:], rtol=0.0, atol=1e-5):
            raise ValueError(f"{split} action does not reconstruct the next controller state")
        np.savez_compressed(
            args.out_dir / f"{split}.npz",
            state=split_state,
            state_next=split_next,
            action=split_action,
            action_raw=split_action_raw,
            ext=split_ext,
            timestamp=split_timestamp,
            timestamp_next=split_timestamp + 300,
            segment_id=split_segment,
            source_row_id=indices,
        )
        windows = {}
        for length in (9, 23):
            starts = valid_starts(split_segment, length)
            np.save(args.out_dir / f"{split}_valid_starts_L{length}.npy", starts)
            windows[str(length)] = int(len(starts))
        manifest["splits"][split] = {
            "rows": int(len(split_state)),
            "windows": windows,
            "first_timestamp": str(np.datetime64(int(split_timestamp[0]), "s")),
            "last_timestamp": str(np.datetime64(int(split_timestamp[-1]), "s")),
            "npz_sha256": sha256(args.out_dir / f"{split}.npz"),
        }

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
