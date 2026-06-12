#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cluster subject-level CARLA and pupil features.

The script only consumes existing feature tables. It does not create fallback
records when inputs are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_FEATURES = [
    "mean_speed",
    "max_speed",
    "mean_abs_steer",
    "mean_throttle",
    "mean_brake",
    "feedback_events",
    "feedback_duration_sec",
    "feedback_ratio",
    "pupil_mean_mm",
    "pupil_std_mm",
]


def load_feature_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature table not found: {path}")
    df = pd.read_csv(path)
    required = {"subject_id", "condition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")
    return df


def aggregate_subject_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    available = [col for col in feature_cols if col in df.columns]
    if not available:
        raise ValueError("No requested feature columns are available in the input table.")

    subject_features = df.groupby("subject_id")[available].mean(numeric_only=True)
    subject_features = subject_features.dropna(axis=1, how="all")
    subject_features = subject_features.fillna(subject_features.mean(numeric_only=True))
    return subject_features


def standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (values - mean) / std


def kmeans(values: np.ndarray, n_clusters: int, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray, float]:
    centers = values[:n_clusters].copy()
    labels = np.zeros(values.shape[0], dtype=int)

    for _ in range(max_iter):
        distances = ((values[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if mask.any():
                centers[cluster_id] = values[mask].mean(axis=0)

    inertia = float(((values - centers[labels]) ** 2).sum())
    return labels, centers, inertia


def run_clustering(features: pd.DataFrame, n_clusters: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(features) < n_clusters:
        raise ValueError(f"n_clusters={n_clusters} is larger than subject count={len(features)}")

    scaled = standardize(features.to_numpy(dtype=float))
    labels, _, inertia = kmeans(scaled, n_clusters)

    result = features.copy()
    result.insert(0, "cluster", labels)

    metrics = pd.DataFrame([{
        "n_clusters": n_clusters,
        "inertia": inertia,
        "subjects": len(features),
        "features": features.shape[1],
    }])
    return result, metrics


def save_pca_plot(features: pd.DataFrame, labels: pd.Series, output_path: Path) -> None:
    scaled = standardize(features.to_numpy(dtype=float))
    n_components = min(2, scaled.shape[1], scaled.shape[0])
    if n_components < 2:
        return

    centered = scaled - scaled.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T
    points = centered @ components
    explained = (points.var(axis=0) / max(centered.var(axis=0).sum(), 1e-12))

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=90)
    for idx, subject_id in enumerate(features.index):
        plt.annotate(subject_id, (points[idx, 0], points[idx, 1]), xytext=(5, 5), textcoords="offset points")
    plt.xlabel(f"PC1 ({explained[0]:.1%})")
    plt.ylabel(f"PC2 ({explained[1]:.1%})")
    plt.title("Subject Clusters")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster subject-level CARLA and pupil features.")
    parser.add_argument("--input", type=Path, default=Path("results/trial_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/clustering"))
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trial_df = load_feature_table(args.input)
    subject_features = aggregate_subject_features(trial_df, args.features)

    clustered, metrics = run_clustering(subject_features, args.n_clusters)
    clustered.to_csv(args.output_dir / "subject_clusters.csv")
    subject_features.to_csv(args.output_dir / "subject_features.csv")
    metrics.to_csv(args.output_dir / "cluster_metrics.csv", index=False)
    save_pca_plot(subject_features, clustered["cluster"], args.output_dir / "subject_clusters_pca.png")

    print(f"Clustering outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
