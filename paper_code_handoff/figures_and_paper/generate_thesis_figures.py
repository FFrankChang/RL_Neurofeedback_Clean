# -*- coding: utf-8 -*-
"""
生成论文图表
============

本脚本从各分析管线输出的结果文件读取数据并绘制论文图表；
若缺少对应结果文件，会明确报错提示先运行对应管线。
图10为方法示意图（非数据图），保留为绘制。

输入来源：
- 图3-5 特征重要性  ← arousal_classification.py 的 feature_importance.json
- 图3-6 ROC曲线     ← arousal_classification.py 的 roc_data.json
- 图3-12 混淆矩阵   ← SVM_cluster_v2.py 的 loocv_results.json
- 图4-5 训练面板    ← train_neural_feedback.py 的 training_stats.pkl
- 图4-6 延迟分布    ← 延迟测量CSV（列: t_acquire,t_feature,t_infer,t_actuate, 单位ms）
- 图4-7 安全层级    ← 方法示意图（无数据）
"""
import argparse
import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'thesis_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _require(path, what):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"未找到{what}: {path}\n请先运行对应管线生成结果文件后再绘图。"
        )
    return path


def fig1_feature_importance(importance_json):
    """图3-5: 眼动特征重要性（来自XGBoost feature_importance.json）"""
    _require(importance_json, "特征重要性结果(feature_importance.json)")
    with open(importance_json, 'r', encoding='utf-8') as f:
        imp = json.load(f)
    items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
    features = [k for k, _ in items]
    importance = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#c0392b' if v > 0.1 else '#2980b9' if v > 0.05 else '#7f8c8d' for v in importance]
    bars = ax.barh(range(len(features)), importance, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
    for bar, val in zip(bars, importance):
        ax.text(bar.get_width() + max(importance) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel('归一化增益重要性 (Normalized Gain Importance)', fontsize=12)
    ax.set_title('XGBoost模型眼动特征重要性排序', fontsize=14, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    legend_elements = [
        mpatches.Patch(facecolor='#c0392b', label='高重要性 (>0.10)'),
        mpatches.Patch(facecolor='#2980b9', label='中重要性 (0.05-0.10)'),
        mpatches.Patch(facecolor='#7f8c8d', label='低重要性 (<0.05)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig1_feature_importance.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] {path}')


def fig2_roc_curve(roc_json):
    """图3-6: ROC曲线（来自分类结果 roc_data.json）"""
    _require(roc_json, "ROC原始数据(roc_data.json)")
    with open(roc_json, 'r', encoding='utf-8') as f:
        roc = json.load(f)
    style = {
        'SVM': ('#3498db', 2, 'SVM (RBF)'),
        'RandomForest': ('#2ecc71', 2, 'Random Forest'),
        'XGBoost': ('#e74c3c', 2.5, 'XGBoost'),
    }
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, (color, lw, label) in style.items():
        if name not in roc:
            continue
        fpr = np.array(roc[name]['fpr'])
        tpr = np.array(roc[name]['tpr'])
        ax.plot(fpr, tpr, color=color, linewidth=lw,
                label=f"{label}  AUC = {roc[name]['auc']:.3f}")
    if 'XGBoost' in roc:
        ax.fill_between(np.array(roc['XGBoost']['fpr']), np.array(roc['XGBoost']['tpr']),
                        alpha=0.1, color='#e74c3c')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='随机基线 (AUC = 0.500)')
    ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=12)
    ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=12)
    ax.set_title('认知负荷预测模型ROC曲线对比', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])
    ax.grid(alpha=0.3, linestyle='--'); ax.set_aspect('equal')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig2_roc_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] {path}')


def fig7_confusion_matrix(loocv_json):
    """图3-12: 四类分类器混淆矩阵（来自 loocv_results.json）"""
    _require(loocv_json, "LOOCV结果(loocv_results.json)")
    with open(loocv_json, 'r', encoding='utf-8') as f:
        res = json.load(f)
    cm = np.array(res['confusion_matrix'])
    classes = ['类型I\n低唤醒-稳定型', '类型II\n中唤醒-适应型',
               '类型III\n高唤醒-波动型', '类型IV\n高唤醒-高效型'][:cm.shape[0]]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=cm.max())
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    fontsize=18, fontweight='bold', color=color)
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=10); ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlabel('预测标签', fontsize=13, labelpad=10)
    ax.set_ylabel('参考标签', fontsize=13, labelpad=10)
    ax.set_title('LOOCV四类认知亚型分类混淆矩阵', fontsize=14, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('样本数', fontsize=11)

    acc = res.get('accuracy', np.trace(cm) / cm.sum())
    ba = res.get('balanced_accuracy', float('nan'))
    mf1 = res.get('macro_f1', float('nan'))
    ax.text(0.5, -0.18,
            f'总体准确率: {np.trace(cm)}/{cm.sum()} = {acc:.1%}    '
            f'平衡准确率: {ba:.1%}    宏平均F1: {mf1:.3f}',
            transform=ax.transAxes, ha='center', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray'))
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig7_confusion_matrix.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] {path}')


def _smooth(a, w):
    a = np.asarray(a, dtype=float)
    if len(a) < w or w <= 1:
        return a
    return np.convolve(a, np.ones(w) / w, mode='same')


def fig8_iql_training_panel(stats_pkl, eval_frequency=500):
    """图4-5: IQL训练六子图面板（来自 training_stats.pkl）"""
    _require(stats_pkl, "训练统计(training_stats.pkl)")
    with open(stats_pkl, 'rb') as f:
        s = pickle.load(f)

    rewards = np.asarray(s.get('episode_rewards', []), dtype=float)
    arousal = np.asarray(s.get('arousal_tracking_error', []), dtype=float)
    fb = np.asarray(s.get('feedback_usage_rate', []), dtype=float)
    lh = s.get('loss_history', {})
    q_loss = np.asarray(lh.get('q_loss', []), dtype=float)
    v_loss = np.asarray(lh.get('v_loss', []), dtype=float)
    pi_loss = np.asarray(lh.get('policy_loss', []), dtype=float)
    eval_reward = np.asarray(s.get('evaluation_rewards', []), dtype=float)
    eval_arousal = np.asarray(s.get('evaluation_arousal_errors', []), dtype=float)

    x = np.arange(len(rewards))
    eval_x = np.arange(1, len(eval_reward) + 1) * eval_frequency

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('IQL神经反馈策略模型训练过程', fontsize=16, fontweight='bold', y=0.98)

    ax = axes[0, 0]
    if len(rewards):
        ax.fill_between(x, rewards, alpha=0.15, color='blue')
        ax.plot(x, _smooth(rewards, 50), color='#2c3e50', linewidth=2, label='滑动平均(50)')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='目标奖励')
        ax.legend(fontsize=9)
    ax.set_title('(a) Episode累积奖励', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode'); ax.set_ylabel('累积奖励'); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if len(arousal):
        ax.fill_between(x, arousal, alpha=0.15, color='green')
        ax.plot(x, _smooth(arousal, 50), color='#27ae60', linewidth=2, label='滑动平均')
        ax.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='目标误差(0.10)')
        ax.legend(fontsize=9)
    ax.set_title('(b) Arousal跟踪误差', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode'); ax.set_ylabel('平均绝对误差'); ax.grid(alpha=0.3)

    ax = axes[0, 2]
    if len(fb):
        ax.fill_between(x, fb, alpha=0.15, color='orange')
        ax.plot(x, _smooth(fb, 50), color='#e67e22', linewidth=2, label='滑动平均')
        ax.legend(fontsize=9)
    ax.set_title('(c) 反馈使用率', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode'); ax.set_ylabel('反馈比例'); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if len(q_loss):
        ax.plot(_smooth(q_loss, 30), color='#e74c3c', linewidth=1.5, label='Q-Loss')
    if len(v_loss):
        ax.plot(_smooth(v_loss, 30), color='#3498db', linewidth=1.5, label='V-Loss')
    if len(pi_loss):
        ax.plot(_smooth(pi_loss, 30), color='#9b59b6', linewidth=1.5, label='Policy-Loss')
    ax.set_title('(d) 训练损失函数', fontsize=12, fontweight='bold')
    ax.set_xlabel('更新步数'); ax.set_ylabel('损失值'); ax.set_yscale('log')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if len(eval_reward):
        ax.plot(eval_x, eval_reward, 'o-', color='#8e44ad', linewidth=2, markersize=5)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('(e) 评估平均奖励', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode'); ax.set_ylabel('平均奖励'); ax.grid(alpha=0.3)

    ax = axes[1, 2]
    if len(eval_arousal):
        ax.plot(eval_x, eval_arousal, 's-', color='#c0392b', linewidth=2, markersize=5)
        ax.axhline(y=0.10, color='orange', linestyle='--', alpha=0.7, label='目标误差')
        ax.legend(fontsize=9)
    ax.set_title('(f) 评估Arousal误差', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode'); ax.set_ylabel('平均绝对误差'); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUTPUT_DIR, 'fig8_iql_training_panel.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] {path}')


def fig9_latency_histogram(latency_csv):
    """图4-6: 端到端延迟分布（来自延迟测量CSV）

    CSV需含列: t_acquire, t_feature, t_infer, t_actuate（单位ms，每行一次测量）。
    """
    import pandas as pd
    _require(latency_csv, "延迟测量CSV")
    df = pd.read_csv(latency_csv)
    need = ['t_acquire', 't_feature', 't_infer', 't_actuate']
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"延迟CSV缺少列: {miss}（需含 {need}，单位ms）")
    t_acquire = df['t_acquire'].values
    t_feature = df['t_feature'].values
    t_infer = df['t_infer'].values
    t_actuate = df['t_actuate'].values
    t_total = t_acquire + t_feature + t_infer + t_actuate

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    bp = ax.boxplot([t_acquire, t_feature, t_infer, t_actuate],
                    labels=['数据采集', '特征提取', '模型推理', '反馈执行'],
                    patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel('延迟 (ms)', fontsize=12)
    ax.set_title('(a) 各环节延迟分布', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    ax = axes[1]
    ax.hist(t_total, bins=50, color='#2c3e50', alpha=0.7, edgecolor='white', linewidth=0.5, density=True)
    mean_t = float(np.mean(t_total)); p99 = float(np.percentile(t_total, 99))
    ax.axvline(mean_t, color='#e74c3c', linewidth=2, label=f'均值 = {mean_t:.1f} ms')
    ax.axvline(p99, color='#f39c12', linewidth=2, linestyle='--', label=f'P99 = {p99:.1f} ms')
    ax.axvline(500, color='green', linewidth=2, linestyle=':', alpha=0.6, label='交互阈值 = 500 ms')
    ax.set_xlabel('端到端延迟 (ms)', fontsize=12)
    ax.set_ylabel('概率密度', fontsize=12)
    ax.set_title(f'(b) 端到端总延迟分布 (N={len(t_total)})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right'); ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig9_latency_distribution.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] {path}')


def fig10_safety_hierarchy():
    """图4-7: 安全约束层级示意图（方法示意，非数据图）"""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9); ax.axis('off')
    layer_configs = [
        {'y': 7.0, 'color': '#e74c3c', 'title': '系统层安全保护',
         'items': ['置信度过滤: 策略置信度 < 0.3 时回退默认策略',
                   '频率限制: 两次反馈最小间隔 ≥ 5秒',
                   'Arousal极端保护: 连续3步 < 0.15 时强制多模态反馈',
                   '24小时连续运行无故障验证']},
        {'y': 4.5, 'color': '#f39c12', 'title': '网络层稳定性保障',
         'items': ['梯度L2范数裁剪 (阈值 = 1.0)',
                   '目标网络软更新 (τ = 0.005)',
                   '优势函数裁剪 (±100)',
                   'PER重要性采样偏差修正 (β: 0.4→1.0)']},
        {'y': 2.0, 'color': '#3498db', 'title': '奖励层安全约束',
         'items': ['硬安全边界: arousal < 0.2 时二次惩罚',
                   '频繁反馈爆增惩罚: 最近3步≥2次反馈时惩罚×2',
                   '紧急场景适配: 要求arousal > 0.7才给正奖励',
                   '信号质量惩罚: -0.5 × artifact_level']},
    ]
    for cfg in layer_configs:
        y = cfg['y']; color = cfg['color']
        rect = FancyBboxPatch((0.5, y - 0.85), 13, 1.7, boxstyle="round,pad=0.15",
                              facecolor=color, alpha=0.12, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(7, y + 0.65, cfg['title'], fontsize=14, fontweight='bold',
                ha='center', va='center', color=color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, linewidth=1.5))
        for i, item in enumerate(cfg['items']):
            ax.text(1.3, y - 0.05 - i * 0.35, f'● {item}', fontsize=10, va='center', color='#2c3e50')
    for y_start, y_end in [(7.0 - 0.95, 4.5 + 0.9), (4.5 - 0.95, 2.0 + 0.9)]:
        ax.annotate('', xy=(7, y_end), xytext=(7, y_start),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=2.5, mutation_scale=20))
    ax.text(7, 0.3, '三层递进防御：奖励函数内嵌安全约束 → 网络训练稳定性机制 → 系统级硬保护',
            fontsize=11, ha='center', va='center', style='italic', color='#555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ecf0f1', edgecolor='#bdc3c7'))
    ax.text(7, 8.8, '神经反馈系统三级安全约束机制', fontsize=16, fontweight='bold',
            ha='center', va='center', color='#2c3e50')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig10_safety_hierarchy.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'[OK] {path}')


def main():
    parser = argparse.ArgumentParser(description="生成论文图表")
    parser.add_argument('--importance-json', type=str, default=None,
                        help='图3-5: arousal_classification 输出的 feature_importance.json')
    parser.add_argument('--roc-json', type=str, default=None,
                        help='图3-6: arousal_classification 输出的 roc_data.json')
    parser.add_argument('--loocv-json', type=str, default=None,
                        help='图3-12: SVM_cluster_v2 输出的 loocv_results.json')
    parser.add_argument('--stats-pkl', type=str, default=None,
                        help='图4-5: train_neural_feedback 输出的 training_stats.pkl')
    parser.add_argument('--latency-csv', type=str, default=None,
                        help='图4-6: 延迟测量CSV')
    parser.add_argument('--eval-frequency', type=int, default=500)
    parser.add_argument('--only', type=str, nargs='*', default=None,
                        help='仅生成指定图，如 --only fig10')
    args = parser.parse_args()

    print('=== 生成论文图表 ===')
    print(f'输出目录: {OUTPUT_DIR}\n')

    jobs = {
        'fig1': lambda: fig1_feature_importance(args.importance_json),
        'fig2': lambda: fig2_roc_curve(args.roc_json),
        'fig7': lambda: fig7_confusion_matrix(args.loocv_json),
        'fig8': lambda: fig8_iql_training_panel(args.stats_pkl, args.eval_frequency),
        'fig9': lambda: fig9_latency_histogram(args.latency_csv),
        'fig10': fig10_safety_hierarchy,
    }
    selected = args.only if args.only else list(jobs.keys())
    for name in selected:
        try:
            jobs[name]()
        except (FileNotFoundError, ValueError) as e:
            print(f'[跳过 {name}] {e}\n')

    print('\n=== 完成 ===')


if __name__ == '__main__':
    main()
