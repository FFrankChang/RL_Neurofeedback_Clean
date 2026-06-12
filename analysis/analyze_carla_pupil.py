#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA driving and pupil response analysis.

This script reads recorded CARLA driving CSV files and processed pupil traces,
extracts feedback periods from RL trials, and generates summary tables and
figures for downstream reporting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = ("RL", "feedback", "silence")
PUPIL_FILES = {
    "S01": "pupil_S01.csv",
    "S02": "pupil_S02.csv",
    "S03": "pupil_S03.csv",
    "S04": "pupil_S04.csv",
}


@dataclass(frozen=True)
class TrialMeta:
    path: Path
    subject_id: str
    condition: str
    timestamp_label: str


def parse_trial_meta(path: Path) -> TrialMeta:
    parts = path.stem.split("_")
    if len(parts) < 6:
        raise ValueError(f"Unexpected CARLA filename format: {path.name}")

    subject_id = parts[2]
    condition = parts[4]
    timestamp_label = parts[5]
    if condition not in CONDITIONS:
        raise ValueError(f"Unexpected condition in {path.name}: {condition}")

    return TrialMeta(path=path, subject_id=subject_id, condition=condition, timestamp_label=timestamp_label)


def load_carla_trials(carla_dir: Path) -> list[tuple[TrialMeta, pd.DataFrame]]:
    trials: list[tuple[TrialMeta, pd.DataFrame]] = []
    for path in sorted(carla_dir.glob("*.csv")):
        meta = parse_trial_meta(path)
        df = pd.read_csv(path)
        marker_cols = [col for col in df.columns if col.startswith("Unnamed")]
        for col in marker_cols:
            values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
            if len(values) > 0 and set(values).issubset({0, 1}):
                df = df.rename(columns={col: "Feedback_Marker"})
            else:
                df = df.drop(columns=[col])
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
        trials.append((meta, df))
    return trials


def load_pupil_data(pupil_dir: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for subject_id, filename in PUPIL_FILES.items():
        path = pupil_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["pupil_mm"] = pd.to_numeric(df["PupilDiameter"], errors="coerce") * 1000
        data[subject_id] = df.dropna(subset=["timestamp", "pupil_mm"]).copy()
    return data


def detect_feedback_column(df: pd.DataFrame) -> str | None:
    candidate_cols = [col for col in df.columns if col not in {
        "timestamp", "Speed", "Location_x", "Location_y", "Location_z",
        "Steer", "Throttle", "Brake", "Collision_Time",
    }]
    for col in candidate_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
        if len(values) > 0 and set(values).issubset({0, 1}):
            return col
    return None


def extract_feedback_periods(df: pd.DataFrame) -> list[tuple[float, float]]:
    feedback_col = detect_feedback_column(df)
    if feedback_col is None:
        return []

    feedback_data = df[["timestamp", feedback_col]].dropna()
    periods: list[tuple[float, float]] = []
    start_time: float | None = None

    for _, row in feedback_data.iterrows():
        marker = int(row[feedback_col])
        timestamp = float(row["timestamp"])
        if marker == 0:
            start_time = timestamp
        elif marker == 1 and start_time is not None:
            periods.append((start_time, timestamp))
            start_time = None

    return periods


def summarize_trial(meta: TrialMeta, carla_df: pd.DataFrame, pupil_df: pd.DataFrame | None) -> dict[str, float | str | int]:
    duration = float(carla_df["timestamp"].max() - carla_df["timestamp"].min())
    feedback_periods = extract_feedback_periods(carla_df) if meta.condition == "RL" else []
    feedback_duration = float(sum(max(0.0, end - start) for start, end in feedback_periods))

    row: dict[str, float | str | int] = {
        "file": meta.path.name,
        "subject_id": meta.subject_id,
        "condition": meta.condition,
        "trial_start": float(carla_df["timestamp"].min()),
        "trial_end": float(carla_df["timestamp"].max()),
        "duration_sec": duration,
        "mean_speed": float(carla_df["Speed"].mean()),
        "max_speed": float(carla_df["Speed"].max()),
        "mean_abs_steer": float(carla_df["Steer"].abs().mean()),
        "mean_throttle": float(carla_df["Throttle"].mean()),
        "mean_brake": float(carla_df["Brake"].mean()),
        "collision_events": int(carla_df["Collision_Time"].notna().sum()) if "Collision_Time" in carla_df else 0,
        "feedback_events": len(feedback_periods),
        "feedback_duration_sec": feedback_duration,
        "feedback_ratio": feedback_duration / duration if duration > 0 else 0.0,
    }

    if pupil_df is not None:
        mask = (pupil_df["timestamp"] >= row["trial_start"]) & (pupil_df["timestamp"] <= row["trial_end"])
        pupil_segment = pupil_df.loc[mask]
        row["pupil_mean_mm"] = float(pupil_segment["pupil_mm"].mean()) if len(pupil_segment) else np.nan
        row["pupil_std_mm"] = float(pupil_segment["pupil_mm"].std()) if len(pupil_segment) else np.nan
        row["pupil_samples"] = int(len(pupil_segment))
    else:
        row["pupil_mean_mm"] = np.nan
        row["pupil_std_mm"] = np.nan
        row["pupil_samples"] = 0

    return row


def build_summary(trials: Iterable[tuple[TrialMeta, pd.DataFrame]], pupil_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for meta, carla_df in trials:
        rows.append(summarize_trial(meta, carla_df, pupil_data.get(meta.subject_id)))
    return pd.DataFrame(rows)


def save_condition_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    summary_df.boxplot(column="mean_speed", by="condition", grid=False)
    plt.title("Mean Speed by Condition")
    plt.suptitle("")
    plt.xlabel("Condition")
    plt.ylabel("Mean Speed")
    plt.tight_layout()
    plt.savefig(output_dir / "mean_speed_by_condition.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    summary_df.boxplot(column="pupil_mean_mm", by="condition", grid=False)
    plt.title("Mean Pupil Diameter by Condition")
    plt.suptitle("")
    plt.xlabel("Condition")
    plt.ylabel("Pupil Diameter (mm)")
    plt.tight_layout()
    plt.savefig(output_dir / "pupil_by_condition.png", dpi=200)
    plt.close()

    rl_df = summary_df[summary_df["condition"] == "RL"]
    if len(rl_df):
        plt.figure(figsize=(9, 5))
        plt.bar(rl_df["file"], rl_df["feedback_events"], color="#4C78A8")
        plt.xticks(rotation=75, ha="right")
        plt.ylabel("Feedback Events")
        plt.title("Feedback Events in RL Trials")
        plt.tight_layout()
        plt.savefig(output_dir / "rl_feedback_events.png", dpi=200)
        plt.close()


def write_report(summary_df: pd.DataFrame, output_dir: Path) -> None:
    condition_summary = summary_df.groupby("condition").agg(
        trials=("file", "count"),
        mean_duration_sec=("duration_sec", "mean"),
        mean_speed=("mean_speed", "mean"),
        mean_pupil_mm=("pupil_mean_mm", "mean"),
        total_feedback_events=("feedback_events", "sum"),
        mean_feedback_ratio=("feedback_ratio", "mean"),
    ).reset_index()

    markdown_table = condition_summary.to_csv(index=False, float_format="%.3f").strip()

    report_path = output_dir / "final_analysis_summary.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# CARLA Driving And Pupil Analysis Summary\n\n")
        f.write(f"- Total trials: {len(summary_df)}\n")
        f.write(f"- Subjects: {', '.join(sorted(summary_df['subject_id'].unique()))}\n")
        f.write(f"- Conditions: {', '.join(sorted(summary_df['condition'].unique()))}\n\n")
        f.write("## Condition Summary\n\n")
        f.write("```csv\n")
        f.write(markdown_table)
        f.write("\n```")
        f.write("\n\n## Output Files\n\n")
        f.write("- `trial_summary.csv`: trial-level metrics\n")
        f.write("- `condition_summary.csv`: condition-level metrics\n")
        f.write("- `plots/`: exported figures\n")

    condition_summary.to_csv(output_dir / "condition_summary.csv", index=False)


def run_analysis(data_dir: Path, output_dir: Path) -> None:
    carla_dir = data_dir / "carla_data"
    pupil_dir = data_dir / "pupil_data"

    trials = load_carla_trials(carla_dir)
    pupil_data = load_pupil_data(pupil_dir)

    if not trials:
        raise RuntimeError(f"No CARLA CSV files found in {carla_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = build_summary(trials, pupil_data)
    summary_df.to_csv(output_dir / "trial_summary.csv", index=False)

    save_condition_plots(summary_df, output_dir / "plots")
    write_report(summary_df, output_dir)

    print(f"Analysis complete: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CARLA driving and pupil response data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    run_analysis(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
