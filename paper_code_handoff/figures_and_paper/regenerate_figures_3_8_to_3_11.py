#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成图3-8到图3-11，修复中文字体渲染问题
读取 eeg_6_features_clustering.py 产生的真实结果目录，并将图写回该目录。
用法: python regenerate_figures_3_8_to_3_11.py --results-dir <eeg_6features_clustering_xxx>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = None  # 由命令行 --results-dir 指定（指向真实聚类结果目录）


def setup_chinese_font():
    """设置中文字体，确保中文字符正确显示"""
    font_candidates = [
        ('C:/Windows/Fonts/msyh.ttc', 'Microsoft YaHei'),
        ('C:/Windows/Fonts/msyh.ttf', 'Microsoft YaHei'),
        ('C:/Windows/Fonts/simhei.ttf', 'SimHei'),
        ('C:/Windows/Fonts/simsun.ttc', 'SimSun'),
    ]
    for font_path, font_name in font_candidates:
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"使用字体: {font_path}")
            return font_prop.get_name()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("使用回退字体设置")
    return None


def apply_font():
    """每次新建图表前重新应用字体（防止style.use重置）"""
    plt.rcParams['font.sans-serif'] = plt.rcParams.get('font.sans-serif', ['Microsoft YaHei'])
    plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """从已有CSV文件加载数据"""
    scaled_path = os.path.join(OUTPUT_DIR, 'eeg_6features_scaled_data.csv')
    results_path = os.path.join(OUTPUT_DIR, 'clustering_results.csv')

    selected_features = [
        'psd_alpha_frontal',
        'rel_power_beta_parietal',
        'psd_theta_temporal',
        'conn_frontal_parietal',
        'clustering',
        'rel_power_gamma_frontal'
    ]

    # 中文特征标签（用于图表显示）
    feature_labels_cn = [
        '前额α功率',
        '顶叶β相对功率',
        '颞叶θ功率',
        '前额-顶叶连接性',
        '聚类系数',
        '前额γ相对功率'
    ]

    df_scaled = pd.read_csv(scaled_path, index_col=0)
    df_results = pd.read_csv(results_path, index_col=0)

    cluster_labels = df_results['cluster'].values
    df_features = df_results[selected_features]

    return df_scaled, df_features, cluster_labels, selected_features, feature_labels_cn


COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

# 亚型名称映射
CLUSTER_NAMES = {
    0: '类型I\n低唤醒-疲劳型',
    1: '类型II\n低唤醒-稳定型',
    2: '类型III\n高唤醒-紧张型',
    3: '类型IV\n高唤醒-高效型',
}


def create_pca_visualization(df_scaled, cluster_labels):
    """图3-8: PCA降维可视化"""
    apply_font()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))

    for cluster_id in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1],
                   c=COLORS[cluster_id], label=CLUSTER_NAMES.get(cluster_id, f'簇 {cluster_id}'),
                   alpha=0.8, s=80, edgecolors='white', linewidths=0.5)

    for i, subject in enumerate(df_scaled.index):
        ax.annotate(subject, (pca_result[i, 0], pca_result[i, 1]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.7)

    ax.set_xlabel(f'PC1（方差解释率 {pca.explained_variance_ratio_[0]:.1%}）', fontsize=12)
    ax.set_ylabel(f'PC2（方差解释率 {pca.explained_variance_ratio_[1]:.1%}）', fontsize=12)
    ax.set_title(f'聚类结果PCA二维投影可视化\n（总方差解释率：{pca.explained_variance_ratio_.sum():.1%}）', fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(OUTPUT_DIR, 'pca_visualization.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 图3-8 已保存: {out_path}")


def create_radar_charts(df_scaled, cluster_labels, feature_labels_cn):
    """图3-9: 四类认知亚型EEG雷达图"""
    apply_font()
    n_features = len(feature_labels_cn)
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
    angles += angles[:1]

    cluster_means = []
    for cluster_id in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        cluster_means.append(df_scaled[mask].mean().values)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    for i, (cluster_id, cluster_mean) in enumerate(zip(sorted(np.unique(cluster_labels)), cluster_means)):
        ax = axes[i]
        values = cluster_mean.tolist() + [cluster_mean[0]]

        ax.plot(angles, values, 'o-', linewidth=2, color=COLORS[i])
        ax.fill(angles, values, alpha=0.3, color=COLORS[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels_cn, fontsize=9)
        name = CLUSTER_NAMES.get(cluster_id, f'簇 {cluster_id}').replace('\n', ' ')
        ax.set_title(name, fontsize=12, pad=20, color=COLORS[i])
        ax.grid(True)
        # y轴范围
        ax.set_ylim(-2.5, 2.5)
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels(['-2', '-1', '0', '1', '2'], fontsize=7)

    plt.suptitle('四类认知亚型EEG特征雷达图', fontsize=16, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'cluster_radar_charts.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 图3-9 已保存: {out_path}")


def create_feature_boxplots(df_features, cluster_labels, selected_features, feature_labels_cn):
    """图3-10: 四类亚型特征箱线图"""
    apply_font()
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (feature, label_cn) in enumerate(zip(selected_features, feature_labels_cn)):
        ax = axes[i]
        data_by_cluster = []
        x_labels = []
        for cluster_id in sorted(np.unique(cluster_labels)):
            mask = cluster_labels == cluster_id
            data_by_cluster.append(df_features[mask][feature].values)
            x_labels.append(CLUSTER_NAMES.get(cluster_id, f'簇 {cluster_id}').replace('\n', '\n'))

        bp = ax.boxplot(data_by_cluster, labels=x_labels, patch_artist=True,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(label_cn, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=8)

    plt.suptitle('四类认知亚型EEG特征箱线图对比', fontsize=16)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'feature_boxplots.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 图3-10 已保存: {out_path}")


def create_cluster_centers_heatmap(df_scaled, cluster_labels, selected_features, feature_labels_cn):
    """图3-11: 聚类中心热力图"""
    apply_font()
    cluster_centers = []
    row_labels = []

    for cluster_id in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        center = df_scaled[mask].mean().values
        cluster_centers.append(center)
        row_labels.append(CLUSTER_NAMES.get(cluster_id, f'簇 {cluster_id}').replace('\n', ' '))

    centers_df = pd.DataFrame(
        cluster_centers,
        columns=feature_labels_cn,
        index=row_labels
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(centers_df, annot=True, cmap='RdBu_r', center=0,
                fmt='.3f', ax=ax,
                cbar_kws={'label': '标准化特征值', 'shrink': 0.8},
                linewidths=0.5)

    ax.set_title('四类认知亚型聚类中心热力图', fontsize=14, pad=15)
    ax.set_xlabel('EEG特征', fontsize=12)
    ax.set_ylabel('认知亚型', fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'cluster_centers_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 图3-11 已保存: {out_path}")


def main():
    import argparse
    global OUTPUT_DIR
    parser = argparse.ArgumentParser(description="重生成图3-8~3-11（读取真实聚类结果目录）")
    parser.add_argument("--results-dir", required=True,
                        help="eeg_6_features_clustering.py 输出的结果目录"
                             "（含 eeg_6features_scaled_data.csv 与 clustering_results.csv）")
    args = parser.parse_args()
    OUTPUT_DIR = args.results_dir
    if not os.path.isdir(OUTPUT_DIR):
        raise FileNotFoundError(f"结果目录不存在: {OUTPUT_DIR}")

    print("=" * 60)
    print("重新生成图3-8 ~ 图3-11（修复中文字体渲染）")
    print("=" * 60)

    # 1. 设置字体
    setup_chinese_font()

    # 2. 加载数据
    df_scaled, df_features, cluster_labels, selected_features, feature_labels_cn = load_data()
    print(f"加载数据: {len(df_scaled)} 个受试者, {len(selected_features)} 个特征")
    print(f"聚类标签分布: {dict(zip(*np.unique(cluster_labels, return_counts=True)))}")
    print()

    # 3. 生成图3-8: PCA可视化
    create_pca_visualization(df_scaled, cluster_labels)

    # 4. 生成图3-9: 雷达图
    create_radar_charts(df_scaled, cluster_labels, feature_labels_cn)

    # 5. 生成图3-10: 箱线图
    create_feature_boxplots(df_features, cluster_labels, selected_features, feature_labels_cn)

    # 6. 生成图3-11: 热力图
    create_cluster_centers_heatmap(df_scaled, cluster_labels, selected_features, feature_labels_cn)

    print()
    print("=" * 60)
    print(f"全部图表已保存到: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
