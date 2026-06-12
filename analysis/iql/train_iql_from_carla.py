#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train IQL from recorded CARLA driving and pupil traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from iql_core import IQLConfig, IQLTrainer, Transition


PUPIL_FILES = {
    "S01": "pupil_S01.csv",
    "S02": "pupil_S02.csv",
    "S03": "pupil_S03.csv",
    "S04": "pupil_S04.csv",
}

STATE_COLUMNS = [
    "Speed",
    "Steer",
    "Throttle",
    "Brake",
    "pupil_mm",
    "pupil_delta",
    "speed_delta",
    "recent_feedback",
]


def parse_subject_id(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected CARLA filename format: {path.name}")
    return parts[2]


def load_pupil_data(pupil_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for subject_id, filename in PUPIL_FILES.items():
        path = pupil_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["pupil_mm"] = pd.to_numeric(df["PupilDiameter"], errors="coerce") * 1000
        out[subject_id] = df[["timestamp", "pupil_mm"]].dropna().sort_values("timestamp")
    return out


def load_carla_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in list(df.columns):
        if col.startswith("Unnamed"):
            values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
            if len(values) > 0 and set(values).issubset({0, 1}):
                df = df.rename(columns={col: "Feedback_Marker"})
            else:
                df = df.drop(columns=[col])
    if "Feedback_Marker" not in df.columns:
        df["Feedback_Marker"] = np.nan
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def align_pupil(carla_df: pd.DataFrame, pupil_df: pd.DataFrame | None) -> pd.DataFrame:
    if pupil_df is None or pupil_df.empty:
        carla_df = carla_df.copy()
        carla_df["pupil_mm"] = np.nan
        return carla_df
    return pd.merge_asof(
        carla_df.sort_values("timestamp"),
        pupil_df.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=0.25,
    )


def derive_actions(df: pd.DataFrame) -> pd.Series:
    marker = pd.to_numeric(df["Feedback_Marker"], errors="coerce")
    starts = marker.eq(0)
    ends = marker.eq(1)
    actions = pd.Series(0, index=df.index, dtype=int)
    actions.loc[starts] = 1
    actions.loc[ends] = 2
    return actions


def add_state_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pupil_mm"] = out["pupil_mm"].interpolate(limit_direction="both")
    out["pupil_delta"] = out["pupil_mm"].diff().fillna(0.0)
    out["speed_delta"] = pd.to_numeric(out["Speed"], errors="coerce").diff().fillna(0.0)
    out["recent_feedback"] = derive_actions(out).replace({2: 1}).rolling(50, min_periods=1).mean()
    for col in STATE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=STATE_COLUMNS)


def reward_from_row(row: pd.Series) -> float:
    speed = float(row["Speed"])
    brake = float(row["Brake"])
    steer = abs(float(row["Steer"]))
    pupil = float(row["pupil_mm"])
    pupil_target = 4.2
    pupil_penalty = abs(pupil - pupil_target) / pupil_target
    return float((speed / 100.0) - 0.2 * steer - 0.3 * brake - pupil_penalty)


def fit_standardizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return mean, std


def build_transitions(data_dir: Path) -> tuple[list[Transition], int]:
    carla_dir = data_dir / "carla_data"
    pupil_data = load_pupil_data(data_dir / "pupil_data")
    raw_states = []
    rows = []

    for path in sorted(carla_dir.glob("*RL*.csv")):
        subject_id = parse_subject_id(path)
        carla_df = load_carla_file(path)
        aligned = align_pupil(carla_df, pupil_data.get(subject_id))
        featured = add_state_features(aligned)
        if len(featured) < 2:
            continue
        actions = derive_actions(featured).to_numpy()
        states = featured[STATE_COLUMNS].to_numpy(dtype=np.float32)
        rewards = featured.apply(reward_from_row, axis=1).to_numpy(dtype=np.float32)
        raw_states.append(states)
        rows.append((states, actions, rewards))

    if not rows:
        raise RuntimeError("No usable RL transitions were found.")

    mean, std = fit_standardizer(np.vstack(raw_states))

    transitions: list[Transition] = []
    for states, actions, rewards in rows:
        scaled = ((states - mean) / std).astype(np.float32)
        for idx in range(len(scaled) - 1):
            transitions.append(Transition(
                state=scaled[idx],
                action=int(actions[idx]),
                reward=float(rewards[idx]),
                next_state=scaled[idx + 1],
                done=idx == len(scaled) - 2,
            ))

    return transitions, len(STATE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IQL from recorded CARLA and pupil data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/iql_training"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    transitions, state_dim = build_transitions(args.data_dir)
    config = IQLConfig(state_dim=state_dim, batch_size=args.batch_size, epochs=args.epochs)
    trainer = IQLTrainer(config)
    history = trainer.fit(transitions)

    trainer.save(str(args.output_dir / "iql_model.pt"))
    with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with (args.output_dir / "training_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({
            "num_transitions": len(transitions),
            "state_columns": STATE_COLUMNS,
            "action_mapping": {"0": "hold", "1": "feedback_start", "2": "feedback_end"},
        }, f, indent=2)

    print(f"IQL training outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
