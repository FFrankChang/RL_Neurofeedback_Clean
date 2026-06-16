"""
认知负荷二分类建模（论文3.5.2节 二、基于遥测式眼动指标的认知负荷预测模型）
=================================================================

按论文方法，从遥测式眼动特征预测由EEG脑力负荷指数(MWI)定义的高/低认知负荷：

- 输入特征 (d_e = 12)：瞳孔8维 + 注视2维 + 扫视1维 + 眨眼1维（论文表3-5及3.5.2节）
- 标签生成：以EEG端的MWI（式3-10）按中位数二值化（式3-14）
- 标准化：StandardScaler
- 主分类器：XGBoost（n_estimators=200, max_depth=6, learning_rate=0.1）
- 基线模型：SVM(RBF)、Random Forest(100棵树)
- 数据划分：分层 75% 训练 / 25% 测试
- 交叉验证：5折分层交叉验证（用于报告稳健性能）
- 评估：Accuracy / Precision / Recall / F1 / AUC，混淆矩阵、ROC、特征重要性

输入为样本级CSV，输出分类指标、ROC数据、混淆矩阵与特征重要性结果。
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from xgboost import XGBClassifier
import matplotlib.pyplot as plt


# 论文3.5.2节定义的12维眼动特征（默认列名，可用 --feature-cols 覆盖）
DEFAULT_FEATURE_COLS: List[str] = [
    # 瞳孔相关特征（8维，论文表3-5）
    "pupil_mean",          # 瞳孔直径均值
    "pupil_baseline",      # 基线瞳孔直径
    "pupil_rel_change",    # 瞳孔相对变化率
    "pupil_velocity",      # 瞳孔变化速率
    "pupil_std",           # 瞳孔标准差
    "pupil_mean_1s",       # 1秒窗均值
    "pupil_mean_5s",       # 5秒窗均值
    "pupil_trend",         # 瞳孔变化趋势
    # 注视相关特征（2维）
    "fixation_duration_mean",
    "fixation_rate",
    # 扫视相关特征（1维）
    "saccade_amplitude_mean",
    # 眨眼相关特征（1维）
    "blink_rate",
]


def load_dataset(
    data_path: str,
    feature_cols: List[str],
    label_col: str,
    label_is_mwi: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """加载数据集。

    Args:
        data_path: CSV路径。每行一个时间窗样本，包含眼动特征列与标签列。
        feature_cols: 眼动特征列名（默认12维，见 DEFAULT_FEATURE_COLS）。
        label_col: 标签列名。可为连续的EEG MWI值，或已二值化的0/1标签。
        label_is_mwi: 若为True，label_col为连续MWI，将按中位数二值化（式3-14）。

    Returns:
        X, y, feature_names
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"未找到眼动-认知负荷数据: {data_path}\n"
            f"请提供CSV：每行一个时间窗样本，列包含眼动特征"
            f"（默认12维: {DEFAULT_FEATURE_COLS}）与标签列 '{label_col}'。\n"
            f"标签可为连续的EEG MWI值（--label-is-mwi，将按中位数二值化），"
            f"或已二值化的0/1高低负荷标签。"
        )

    df = pd.read_csv(data_path)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"数据缺少特征列: {missing}\n现有列: {list(df.columns)}"
        )
    if label_col not in df.columns:
        raise ValueError(f"数据缺少标签列 '{label_col}'。现有列: {list(df.columns)}")

    # 丢弃任何包含缺失的样本（明确告知而非静默填充）
    n_before = len(df)
    df = df.dropna(subset=feature_cols + [label_col]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"[数据] 丢弃含缺失值样本 {n_before - len(df)} 条，剩余 {len(df)} 条")

    X = df[feature_cols].values.astype(float)
    raw_label = df[label_col].values

    if label_is_mwi:
        # 论文式3-14：以MWI中位数二值化
        median_mwi = float(np.median(raw_label))
        y = (raw_label >= median_mwi).astype(int)
        print(f"[标签] 按MWI中位数 {median_mwi:.4f} 二值化（式3-14）")
    else:
        y = raw_label.astype(int)

    uniq = np.unique(y)
    if len(uniq) != 2:
        raise ValueError(f"标签需为二分类，实际类别: {uniq}")

    print(f"[数据] 样本数={len(y)}, 特征数={X.shape[1]}, "
          f"正类比例={float(np.mean(y)):.3f}")
    return X, y, list(feature_cols)


def build_models(seed: int) -> Dict[str, object]:
    """构建论文3.5.2节(2)指定的模型与配置。"""
    return {
        # 主分类器：XGBoost（论文指定超参）
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="binary:logistic",
            random_state=seed,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
            tree_method="hist",
        ),
        # 基线：SVM(RBF核)
        "SVM": SVC(kernel="rbf", gamma="scale", probability=True, random_state=seed),
        # 基线：随机森林（100棵树）
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
    }


def evaluate_model(
    name: str,
    model,
    X_train_s, X_test_s, X_train_raw, X_test_raw,
    y_train, y_test,
    needs_scaling: bool,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """在测试集上评估单个模型，返回指标、预测标签与正类概率。"""
    Xtr = X_train_s if needs_scaling else X_train_raw
    Xte = X_test_s if needs_scaling else X_test_raw

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_score = model.predict_proba(Xte)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, y_score)),
    }
    return metrics, y_pred, y_score


def cross_validate_model(model, X, y, needs_scaling: bool, seed: int) -> Dict[str, float]:
    """5折分层交叉验证（论文3.5.2节(2)）。标准化在每折内拟合以避免泄漏。"""
    from sklearn.pipeline import make_pipeline

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    est = make_pipeline(StandardScaler(), model) if needs_scaling else model
    acc = cross_val_score(est, X, y, cv=skf, scoring="accuracy")
    f1 = cross_val_score(est, X, y, cv=skf, scoring="f1")
    return {
        "cv_accuracy_mean": float(acc.mean()),
        "cv_accuracy_std": float(acc.std()),
        "cv_f1_mean": float(f1.mean()),
        "cv_f1_std": float(f1.std()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="眼动→认知负荷二分类建模（论文3.5.2节）"
    )
    parser.add_argument(
        "--data", type=str, default="data/eye_cognitive_load.csv",
        help="数据CSV：每行一个时间窗样本，含眼动特征列与标签列"
    )
    parser.add_argument(
        "--label-col", type=str, default="mwi",
        help="标签列名（连续MWI或0/1标签）"
    )
    parser.add_argument(
        "--label-is-mwi", action="store_true",
        help="标签列为连续MWI时设置此项，按中位数二值化（式3-14）"
    )
    parser.add_argument(
        "--feature-cols", type=str, nargs="*", default=None,
        help="自定义特征列名（默认使用论文12维眼动特征）"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    feature_cols = args.feature_cols if args.feature_cols else DEFAULT_FEATURE_COLS

    X, y, feature_names = load_dataset(
        args.data, feature_cols, args.label_col, args.label_is_mwi
    )

    # 分层 75/25 划分
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    # 标准化（仅用训练集拟合）
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_test_s = scaler.transform(X_test_raw)

    models = build_models(args.seed)
    # XGBoost / RF 基于树，无需标准化；SVM 需要标准化
    needs_scaling = {"XGBoost": False, "SVM": True, "RandomForest": False}

    results: Dict[str, Dict[str, float]] = {}
    outputs: Dict[str, Dict[str, np.ndarray]] = {}

    for name, model in models.items():
        metrics, y_pred, y_score = evaluate_model(
            name, model, X_train_s, X_test_s, X_train_raw, X_test_raw,
            y_train, y_test, needs_scaling[name]
        )
        cv = cross_validate_model(
            build_models(args.seed)[name], X, y, needs_scaling[name], args.seed
        )
        metrics.update(cv)
        results[name] = metrics
        outputs[name] = {"y_pred": y_pred, "y_score": y_score, "model": model}

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = Path(args.output_dir) if args.output_dir else Path(".")
    out_dir = base_root / f"arousal_classification_results_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存指标
    summary = {
        "seed": args.seed,
        "timestamp": timestamp,
        "data": args.data,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "metrics": results,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 混淆矩阵
    for name in models:
        cm = confusion_matrix(y_test, outputs[name]["y_pred"])
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_matrix_{name}.png", dpi=160)
        plt.close(fig)

    # ROC 曲线（同时导出原始 FPR/TPR/AUC 供论文绘图复用）
    roc_data = {}
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name in ["SVM", "RandomForest", "XGBoost"]:
        fpr, tpr, _ = roc_curve(y_test, outputs[name]["y_score"])
        roc_auc = float(auc(fpr, tpr))
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves.png", dpi=160)
    plt.close(fig)
    with open(out_dir / "roc_data.json", "w", encoding="utf-8") as f:
        json.dump(roc_data, f, indent=2, ensure_ascii=False)

    # XGBoost 特征重要性（增益）
    xgb_model = outputs["XGBoost"]["model"]
    importances = xgb_model.feature_importances_
    order = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([feature_names[i] for i in order][::-1],
            [importances[i] for i in order][::-1])
    ax.set_xlabel("Importance (gain)")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=160)
    plt.close(fig)

    importance_dict = {feature_names[i]: float(importances[i]) for i in order}
    with open(out_dir / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump(importance_dict, f, indent=2, ensure_ascii=False)

    # 打印
    print("\n测试集性能 (accuracy / precision / recall / f1 / auc):")
    for name, m in results.items():
        print(f"- {name}: acc={m['accuracy']:.3f}, prec={m['precision']:.3f}, "
              f"rec={m['recall']:.3f}, f1={m['f1']:.3f}, auc={m['auc']:.3f} "
              f"| CV acc={m['cv_accuracy_mean']:.3f}±{m['cv_accuracy_std']:.3f}")
    print(f"\n结果与图表已保存至: {out_dir}")


if __name__ == "__main__":
    main()
