# -*- coding: utf-8 -*-
"""
EEG connectivity → Riemann (SPD) pipeline: 4-cluster unsupervised + supervised classifier
Author: you
"""

import os
import warnings
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (f1_score, balanced_accuracy_score, confusion_matrix,
                             adjusted_rand_score, classification_report, silhouette_score)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import check_random_state
from joblib import dump, load

from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann

warnings.filterwarnings("ignore", category=UserWarning)


# -------------------------
# Utilities
# -------------------------

def _symmetrize_spd(X: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """
    Ensure symmetry and positive-definiteness by eigenvalue floor.
    X: (..., C, C)
    """
    X = 0.5 * (X + np.swapaxes(X, -1, -2))
    *head, C, _ = X.shape
    Xr = X.reshape((-1, C, C))
    Y = np.empty_like(Xr)
    for i, M in enumerate(Xr):
        w, v = np.linalg.eigh(M)
        w = np.maximum(w, eps)
        Y[i] = (v * w) @ v.T
    return Y.reshape((*head, C, C))


def _ensure_4d_spd(X: np.ndarray) -> np.ndarray:
    """
    Accept shapes:
      - (N, C, C)           -> add view dim as 1: (N, 1, C, C)
      - (N, V, C, C)        -> return as-is
    """
    if X.ndim == 3:
        N, C, C2 = X.shape
        assert C == C2, "Last two dims must be square SPD matrices"
        return X[:, None, :, :]
    elif X.ndim == 4:
        return X
    else:
        raise ValueError("X must be (N, C, C) or (N, V, C, C)")


# -------------------------
# Riemann Multi-View Tangent transformer
# -------------------------

class MultiViewRiemannTangent(BaseEstimator, TransformerMixin):
    """
    Fit TangentSpace per view on training data (Riemannian mean per view),
    then transform to tangent features and concatenate across views.

    Input X: (N, V, C, C) or (N, C, C)
    Output Z: (N, sum_d_view), where d_view = C*(C+1)/2 per view (for SPD)
    """
    def __init__(self, metric: str = "riemann"):
        self.metric = metric
        self.ts_list_: List[TangentSpace] = []
        self.views_ = 0
        self.channels_ = None
        self.feature_dims_: List[int] = []

    def fit(self, X: np.ndarray, y=None):
        X = _ensure_4d_spd(_symmetrize_spd(X))
        N, V, C, _ = X.shape
        self.views_ = V
        self.channels_ = C
        self.ts_list_ = []
        self.feature_dims_ = []
        for v in range(V):
            ts = TangentSpace(metric=self.metric)  # ts.fit() will compute Riemann mean of training X[:, v]
            ts.fit(X[:, v])
            self.ts_list_.append(ts)
            self.feature_dims_.append(ts.n_ts)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _ensure_4d_spd(_symmetrize_spd(X))
        assert X.shape[1] == self.views_, f"Expected {self.views_} views, got {X.shape[1]}"
        feats = []
        for v in range(self.views_):
            fv = self.ts_list_[v].transform(X[:, v])
            feats.append(fv)
        return np.hstack(feats)

    def get_feature_slices(self) -> List[slice]:
        """Return index slices for each view in concatenated feature space."""
        idx = 0
        slices = []
        for d in self.feature_dims_:
            slices.append(slice(idx, idx + d))
            idx += d
        return slices


# -------------------------
# Unsupervised: clustering in tangent space
# -------------------------

@dataclass
class ClusteringResult:
    labels: np.ndarray           # (N,)
    responsibilities: np.ndarray # (N, K)
    gmm: GaussianMixture
    features: np.ndarray         # (N, D)
    silhouette: Optional[float]


def cluster_in_tangent_space(
    X_spd: np.ndarray,
    n_clusters: int = 4,
    n_init: int = 20,
    random_state: int = 0
) -> ClusteringResult:
    """
    1) Fit Riemann Tangent per view on ALL subjects (无监督阶段)
    2) Concatenate tangent features
    3) GMM(K=4) clustering
    """
    mv = MultiViewRiemannTangent(metric="riemann")
    mv.fit(X_spd)
    Z = mv.transform(X_spd)

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        n_init=n_init,
        reg_covar=1e-6,
        random_state=random_state
    )
    gmm.fit(Z)
    labels = gmm.predict(Z)
    resp = gmm.predict_proba(Z)

    sil = None
    try:
        sil = silhouette_score(Z, labels)
    except Exception:
        pass

    # pack model with fitted transformer for possible use (store inside gmm? keep separately as needed)
    gmm.mv_tangent_ = mv  # attach for convenience

    return ClusteringResult(labels=labels, responsibilities=resp, gmm=gmm, features=Z, silhouette=sil)


def clustering_stability(
    Z: np.ndarray,
    n_clusters: int = 4,
    n_boot: int = 100,
    sample_frac: float = 0.8,
    random_state: int = 0
) -> Dict[str, Any]:
    """
    Stability by bootstrap subsampling in feature space (tangent already done):
    - Each run: sample ~80% subjects, fit GMM, then predict ALL subjects
    - Compute ARIs between all pairs of runs
    """
    rng = check_random_state(random_state)
    N = Z.shape[0]
    label_runs = []
    for b in range(n_boot):
        idx = rng.choice(N, size=max(3, int(sample_frac * N)), replace=False)
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            n_init=10,
            reg_covar=1e-6,
            random_state=rng.randint(0, 1_000_000)
        ).fit(Z[idx])
        pred_all = gmm.predict(Z)
        label_runs.append(pred_all)

    # pairwise ARI
    K = len(label_runs)
    if K < 2:
        return {"mean_ari": np.nan, "std_ari": np.nan, "all_ari": []}
    all_ari = []
    for i in range(K):
        for j in range(i + 1, K):
            all_ari.append(adjusted_rand_score(label_runs[i], label_runs[j]))
    all_ari = np.array(all_ari)
    return {"mean_ari": float(all_ari.mean()), "std_ari": float(all_ari.std()), "all_ari": all_ari}


# -------------------------
# Supervised: nested-CV classifier to predict cluster label
# -------------------------

@dataclass
class SupervisedCVResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    report: str
    ba: float
    f1m: float
    cm: np.ndarray
    best_models: List[Pipeline]


def train_classifier_loocv(
    X_spd: np.ndarray,
    y: np.ndarray,
    pca_options: Tuple[Optional[float], ...] = (None, 0.95, 0.9, 0.8),
    Cs: Tuple[float, ...] = (0.01, 0.1, 1, 10),
    random_state: int = 0
) -> SupervisedCVResult:
    """
    Outer loop: Leave-One-Out (N folds).
    Inner loop: 5-fold GridSearch on training set to choose PCA rate + C of LR.
    Pipeline: [Riemann Tangent (fit on train)] -> Standardize -> PCA -> LR (multinomial, balanced)
    """
    X_spd = _ensure_4d_spd(_symmetrize_spd(X_spd))
    N = X_spd.shape[0]
    loo = LeaveOneOut()

    y_true, y_pred, y_prob = [], [], []
    best_models = []

    for tr, te in loo.split(np.arange(N)):
        # Build a fresh pipeline per outer fold (to prevent leakage)
        riem = MultiViewRiemannTangent(metric="riemann")
        scaler = StandardScaler(with_mean=True, with_std=True)
        pca = PCA(svd_solver="full")
        lr = LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state
        )

        pipe = Pipeline([
            ("riem", riem),        # fits on train SPD, transforms to tangent features
            ("scaler", scaler),
            ("pca", pca),
            ("clf", lr),
        ])

        # inner CV (on train set only)
        inner = StratifiedKFold(n_splits=min(5, len(np.unique(y[tr])) if len(tr) >= 5 else 3), shuffle=True, random_state=random_state)

        param_grid = {
            "pca__n_components": pca_options,
            "clf__C": Cs,
        }

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=inner,
            scoring="f1_macro",
            n_jobs=-1,
            refit=True
        )
        gs.fit(X_spd[tr], y[tr])

        # Probability calibration on training set (isotonic)
        calibrated = CalibratedClassifierCV(gs.best_estimator_, method="isotonic", cv=inner)
        calibrated.fit(X_spd[tr], y[tr])

        proba = calibrated.predict_proba(X_spd[te])[0]
        pred = int(np.argmax(proba))

        y_true.append(int(y[te][0]))
        y_pred.append(pred)
        y_prob.append(proba)
        best_models.append(calibrated)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    ba = balanced_accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=3)

    return SupervisedCVResult(
        y_true=y_true, y_pred=y_pred, y_prob=y_prob,
        report=report, ba=float(ba), f1m=float(f1m), cm=cm, best_models=best_models
    )


# -------------------------
# Train final production model on ALL data (after CV)
# -------------------------

def train_final_model(
    X_spd: np.ndarray,
    y: np.ndarray,
    pca_n_components: Optional[float] = 0.9,
    C: float = 1.0,
    random_state: int = 0
) -> Pipeline:
    """
    Fit a final pipeline on ALL labelled data with chosen hyperparams.
    Returns a sklearn Pipeline that can be saved and used for new subjects.
    """
    riem = MultiViewRiemannTangent(metric="riemann")
    scaler = StandardScaler(with_mean=True, with_std=True)
    pca = PCA(svd_solver="full", n_components=pca_n_components)
    lr = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced",
        max_iter=5000,
        C=C,
        random_state=random_state
    )
    pipe = Pipeline([("riem", riem), ("scaler", scaler), ("pca", pca), ("clf", lr)])
    # Wrap with calibration for well-calibrated probabilities
    final = CalibratedClassifierCV(pipe, method="isotonic", cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state))
    final.fit(_ensure_4d_spd(_symmetrize_spd(X_spd)), y)
    return final


def save_model(model: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(model, path)


def load_model(path: str) -> Pipeline:
    return load(path)


# -------------------------
# Explainability helpers
# -------------------------

def cluster_centroids_in_tangent(gmm: GaussianMixture, Z: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Return per-cluster centroid in tangent space (simple Euclidean mean in Z-space).
    """
    centroids = {}
    for k in np.unique(labels):
        centroids[int(k)] = Z[labels == k].mean(axis=0)
    return centroids


def top_features_lr(model: Pipeline, top_k: int = 10) -> Dict[int, List[int]]:
    """
    Return top |weights| feature indices per class for multinomial LR inside a calibrated pipeline.
    """
    # CalibratedClassifierCV wraps the pipeline; its estimator is 'base_estimator'
    if isinstance(model, CalibratedClassifierCV):
        pipe = model.base_estimator
    else:
        pipe = model
    lr = pipe.named_steps["clf"]
    coef = lr.coef_  # (K, D_red)
    top = {}
    for k, w in enumerate(coef):
        idx = np.argsort(np.abs(w))[::-1][:top_k]
        top[int(k)] = idx.tolist()
    return top


# -------------------------
# SPD connectivity tensor loader
# -------------------------

def load_spd_tensors(path: str) -> np.ndarray:
    """加载SPD功能连接性张量。

    输入文件（.npy 或 .npz）应包含形状为
        (N_subjects, V_views, C_channels, C_channels)  或
        (N_subjects, C_channels, C_channels)
    的EEG功能连接矩阵（多视图V可对应不同频段/任务条件）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到SPD连接性张量: {path}\n"
            f"请提供符合论文3.7.1节定义的功能连接矩阵，"
            f"形状为 (N, V, C, C) 或 (N, C, C) 的 .npy/.npz 文件。"
        )
    if path.endswith(".npz"):
        npz = np.load(path)
        key = "X_spd" if "X_spd" in npz else npz.files[0]
        X = npz[key]
    else:
        X = np.load(path)
    X = np.asarray(X, dtype=float)
    if X.ndim not in (3, 4):
        raise ValueError(f"SPD张量维度应为3或4，实际为 {X.ndim}（shape={X.shape}）。")
    return X


# -------------------------
# Pipeline entry point
# -------------------------

def run_pipeline(spd_path: str, labels_path: Optional[str] = None,
                 model_out: str = "./models/riemann_fourclass_calibrated.joblib") -> None:
    """对连接性数据执行：黎曼切空间聚类 → LOOCV监督评估 → 生产模型保存。

    Args:
        spd_path: SPD连接性张量路径，(N, V, C, C) 或 (N, C, C)
        labels_path: 可选，分型标签 .npy（长度N）。若未提供，则使用
            无监督切空间GMM(K=4)的标签作为监督信号（与论文一致：以聚类标签
            训练分类器）。
        model_out: 生产模型保存路径
    """
    X_spd = load_spd_tensors(spd_path)
    print(f"[Data] 加载SPD连接性张量: shape={X_spd.shape}")

    # 1) 无监督：黎曼切空间 GMM(K=4)
    clus = cluster_in_tangent_space(X_spd, n_clusters=4, n_init=30, random_state=42)
    print(f"[Cluster] silhouette (tangent space): {clus.silhouette:.3f}"
          if clus.silhouette is not None else "[Cluster] silhouette: N/A")

    stab = clustering_stability(clus.features, n_clusters=4, n_boot=100,
                                sample_frac=0.8, random_state=42)
    print(f"[Cluster] stability ARI mean={stab['mean_ari']:.3f} ± {stab['std_ari']:.3f}")

    # 监督信号：优先使用外部标签，否则用聚类标签
    if labels_path is not None:
        y = np.load(labels_path)
        print(f"[Labels] 使用分型标签: {labels_path}")
    else:
        y = clus.labels
        print("[Labels] 未提供分型标签，使用无监督聚类标签作为监督信号")

    # 2) 监督分类：LOOCV 评估
    cvres = train_classifier_loocv(
        X_spd, y, pca_options=(None, 0.95, 0.9, 0.8),
        Cs=(0.01, 0.1, 1, 10), random_state=42
    )
    print("\n[Supervised LOOCV] classification report:\n", cvres.report)
    print(f"[Supervised LOOCV] Balanced-Acc={cvres.ba:.3f}  Macro-F1={cvres.f1m:.3f}")
    print("[Supervised LOOCV] Confusion matrix:\n", cvres.cm)

    # 导出LOOCV结果（混淆矩阵/指标）供论文绘图复用（图3-12）
    import json as _json
    acc = float(np.trace(cvres.cm) / np.sum(cvres.cm)) if np.sum(cvres.cm) > 0 else 0.0
    weighted_f1 = float(f1_score(cvres.y_true, cvres.y_pred, average="weighted"))
    loocv_out = {
        "confusion_matrix": cvres.cm.tolist(),
        "accuracy": acc,
        "balanced_accuracy": cvres.ba,
        "macro_f1": cvres.f1m,
        "weighted_f1": weighted_f1,
    }
    results_dir = os.path.dirname(os.path.abspath(model_out)) or "."
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "loocv_results.json"), "w", encoding="utf-8") as f:
        _json.dump(loocv_out, f, indent=2, ensure_ascii=False)
    print(f"[LOOCV] 结果已保存: {os.path.join(results_dir, 'loocv_results.json')}")

    # 3) 生产模型（全量数据 + 概率校准）
    final_model = train_final_model(X_spd, y, pca_n_components=0.9, C=1.0, random_state=42)
    save_model(final_model, model_out)
    print(f"[Model] Saved to {model_out}")

    centroids = cluster_centroids_in_tangent(clus.gmm, clus.features, clus.labels)
    for k in sorted(centroids.keys()):
        print(f"[Cluster] centroid vector L2-norm for class {k}: {np.linalg.norm(centroids[k]):.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="EEG黎曼切空间四类分型：无监督聚类 + LOOCV监督分类（3.7.1/3.7.2）"
    )
    parser.add_argument(
        "--spd", type=str, default="data/eeg_connectivity_spd.npy",
        help="SPD连接性张量 .npy/.npz，形状 (N, V, C, C) 或 (N, C, C)"
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="可选：分型标签 .npy（长度N）。缺省时用无监督聚类标签"
    )
    parser.add_argument(
        "--model-out", type=str,
        default="./models/riemann_fourclass_calibrated.joblib"
    )
    args = parser.parse_args()

    run_pipeline(args.spd, labels_path=args.labels, model_out=args.model_out)