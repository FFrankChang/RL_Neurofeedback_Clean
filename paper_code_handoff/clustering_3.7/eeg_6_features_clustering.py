#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EEG 6特征聚类分析
使用6个关键脑电特征对40个受试者进行K=4聚类分析（默认GMM）

作者: AI Assistant
创建时间: 2025-09-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from datetime import datetime
import os

# 设置中文字体和忽略警告
warnings.filterwarnings('ignore')

def _setup_chinese_font():
    """设置中文字体，确保中文字符正确显示"""
    import matplotlib.font_manager as fm
    import os
    # Windows系统常见中文字体路径
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
            plt.rcParams['font.sans-serif'] = [font_prop.get_name(), 'Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return
    # 回退方案
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

_setup_chinese_font()

class EEG6FeaturesClustering:
    """EEG 6特征聚类分析类"""
    
    def __init__(self, data_file=None, output_dir=None, n_init=20, max_iter=200, random_state=42):
        """
        初始化聚类分析
        
        Parameters:
        -----------
        data_file : str, optional
            输入数据文件路径，如果为None则使用现有的EEG特征数据
        output_dir : str, optional
            输出目录，如果为None则自动创建
        """
        self.data_file = data_file
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        
        if output_dir is None:
            self.output_dir = f"eeg_6features_clustering_{self.timestamp}"
        else:
            self.output_dir = output_dir
            
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 选择的6个关键脑电特征
        self.selected_features = [
            'psd_alpha_frontal',      # α波前额叶功率谱密度
            'rel_power_beta_parietal', # β波顶叶相对功率
            'psd_theta_temporal',      # θ波颞叶功率谱密度
            'conn_frontal_parietal',   # 前额-顶叶连接性
            'clustering',              # 聚类系数（图论指标）
            'rel_power_gamma_frontal'  # γ波前额叶相对功率
        ]
        
        print(f"=== EEG 6特征聚类分析 ===")
        print(f"输出目录: {self.output_dir}")
        print(f"选择的6个关键特征:")
        for i, feature in enumerate(self.selected_features, 1):
            print(f"  {i}. {feature}")
        print()
    
    def load_data(self):
        """加载EEG特征数据

        输入：self.data_file 指向的65维EEG特征CSV（首列为被试ID），
        从中提取论文3.7.2节使用的6个关键特征。
        """
        data_file = self.data_file
        if data_file is None:
            raise ValueError(
                "未指定数据文件。请通过 data_file 参数传入65维EEG特征CSV路径"
                "（首列为被试ID）。"
            )
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"未找到EEG特征数据: {data_file}\n"
                f"请提供符合论文3.7.1节定义的特征CSV。"
            )

        print(f"加载数据: {data_file}")
        self.df_raw = pd.read_csv(data_file, index_col=0)

        # 检查是否包含论文3.7.2节定义的6个关键特征
        missing_features = [f for f in self.selected_features if f not in self.df_raw.columns]
        if missing_features:
            raise ValueError(
                f"输入数据缺少论文定义的关键特征: {missing_features}\n"
                f"请确认特征提取流程与论文一致（需包含: {self.selected_features}）。"
            )

        # 提取选择的6个特征
        self.df_features = self.df_raw[self.selected_features].copy()

        if self.df_features.isnull().any().any():
            raise ValueError(
                "6个关键特征中存在缺失值，请在特征提取阶段处理后再进行聚类。"
            )
        
        print(f"数据加载完成:")
        print(f"  - 受试者数量: {len(self.df_features)}")
        print(f"  - 特征数量: {len(self.selected_features)}")
        print(f"  - 实际使用的特征: {self.selected_features}")
        print()
        
        return self.df_features
    
    def preprocess_data(self):
        """数据预处理"""
        print("=== 数据预处理 ===")
        
        # 检查缺失值
        missing_count = self.df_features.isnull().sum().sum()
        if missing_count > 0:
            raise ValueError(
                f"发现 {missing_count} 个缺失值，请在特征提取阶段处理后再聚类。"
            )
        
        # 检查异常值 (使用IQR方法)
        outliers_info = []
        for col in self.df_features.columns:
            Q1 = self.df_features[col].quantile(0.25)
            Q3 = self.df_features[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.df_features[col] < lower_bound) | 
                       (self.df_features[col] > upper_bound)).sum()
            if outliers > 0:
                outliers_info.append(f"  {col}: {outliers}个异常值")
        
        if outliers_info:
            print("检测到异常值:")
            for info in outliers_info:
                print(info)
        else:
            print("未检测到异常值")
        
        # 标准化
        self.scaler = StandardScaler()
        self.df_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df_features),
            columns=self.df_features.columns,
            index=self.df_features.index
        )
        
        print(f"数据标准化完成")
        print(f"标准化后数据形状: {self.df_scaled.shape}")
        
        # 保存预处理后的数据
        scaled_data_path = os.path.join(self.output_dir, 'eeg_6features_scaled_data.csv')
        self.df_scaled.to_csv(scaled_data_path, encoding='utf-8-sig')
        print(f"预处理数据保存至: {scaled_data_path}")
        print()
        
        return self.df_scaled
    
    def perform_clustering(self, k=4):
        """执行GMM聚类（full covariance）"""
        print(f"=== GMM聚类分析 (K={k}) ===")

        # 执行聚类
        self.gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            reg_covar=1e-6,
        )
        self.gmm.fit(self.df_scaled)
        self.cluster_labels = self.gmm.predict(self.df_scaled)
        self.cluster_proba = self.gmm.predict_proba(self.df_scaled)
        
        # 计算聚类评价指标
        silhouette_avg = silhouette_score(self.df_scaled, self.cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.df_scaled, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(self.df_scaled, self.cluster_labels)
        
        print(f"聚类评价指标:")
        print(f"  - Silhouette Score: {silhouette_avg:.3f}")
        print(f"  - Calinski-Harabasz Index: {calinski_harabasz:.3f}")
        print(f"  - Davies-Bouldin Index: {davies_bouldin:.3f}")
        print(f"  - AIC: {self.gmm.aic(self.df_scaled):.2f}")
        print(f"  - BIC: {self.gmm.bic(self.df_scaled):.2f}")
        
        # 统计各簇的样本数量
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        print(f"\n各簇样本分布:")
        for cluster_id, count in zip(unique, counts):
            percentage = count / len(self.cluster_labels) * 100
            print(f"  - 簇 {cluster_id}: {count}个样本 ({percentage:.1f}%)")
        
        # 创建结果DataFrame
        self.df_results = self.df_features.copy()
        self.df_results['cluster'] = self.cluster_labels
        self.df_results['cluster_confidence'] = self.cluster_proba.max(axis=1)
        
        # 保存聚类结果
        results_path = os.path.join(self.output_dir, 'clustering_results.csv')
        self.df_results.to_csv(results_path, encoding='utf-8-sig')
        print(f"\n聚类结果保存至: {results_path}")
        print()
        
        return self.cluster_labels
    
    def analyze_clusters(self):
        """分析各簇的特征"""
        print("=== 簇特征分析 ===")
        
        # 计算各簇的统计信息
        cluster_stats = []
        for cluster_id in sorted(np.unique(self.cluster_labels)):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_data = self.df_features[cluster_mask]
            
            stats = {
                'cluster': cluster_id,
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(self.df_features) * 100
            }
            
            # 计算各特征的均值
            for feature in self.selected_features:
                stats[f'{feature}_mean'] = cluster_data[feature].mean()
                stats[f'{feature}_std'] = cluster_data[feature].std()
            
            cluster_stats.append(stats)
        
        # 转换为DataFrame
        self.df_cluster_stats = pd.DataFrame(cluster_stats)
        
        # 保存簇统计信息
        stats_path = os.path.join(self.output_dir, 'cluster_statistics.csv')
        self.df_cluster_stats.to_csv(stats_path, index=False, encoding='utf-8-sig')
        
        # 打印簇特征摘要
        print("各簇特征摘要:")
        for _, row in self.df_cluster_stats.iterrows():
            cluster_id = int(row['cluster'])
            count = int(row['count'])
            percentage = row['percentage']
            
            print(f"\n簇 {cluster_id} ({count}个样本, {percentage:.1f}%):")
            
            # 找出该簇的特征特点
            feature_highlights = []
            for feature in self.selected_features:
                mean_val = row[f'{feature}_mean']
                # 与整体均值比较
                overall_mean = self.df_features[feature].mean()
                if mean_val > overall_mean * 1.2:
                    feature_highlights.append(f"{feature}: 高 ({mean_val:.3f})")
                elif mean_val < overall_mean * 0.8:
                    feature_highlights.append(f"{feature}: 低 ({mean_val:.3f})")
            
            if feature_highlights:
                print("  特征特点:")
                for highlight in feature_highlights[:3]:  # 只显示前3个最突出的特征
                    print(f"    - {highlight}")
            else:
                print("  特征特点: 接近整体平均水平")
        
        print(f"\n详细统计信息保存至: {stats_path}")
        print()
        
        return self.df_cluster_stats
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("=== 生成可视化图表 ===")
        
        # 设置图表样式（style.use会重置rcParams，需在之后重新设置中文字体）
        plt.style.use('default')
        _setup_chinese_font()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        # 1. 特征分布雷达图
        self._create_radar_chart(colors)
        
        # 2. 特征相关性热图
        self._create_correlation_heatmap()
        
        # 3. PCA降维可视化
        self._create_pca_visualization(colors)
        
        # 4. t-SNE降维可视化
        self._create_tsne_visualization(colors)
        
        # 5. 特征箱型图
        self._create_feature_boxplots(colors)
        
        # 6. 聚类中心热图
        self._create_cluster_centers_heatmap(colors)
        
        print("所有可视化图表生成完成!")
        print()
    
    def _create_radar_chart(self, colors):
        """创建雷达图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        # 计算各簇的特征均值（标准化后）
        cluster_means = []
        for cluster_id in sorted(np.unique(self.cluster_labels)):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_mean = self.df_scaled[cluster_mask].mean()
            cluster_means.append(cluster_mean.values)
        
        # 角度设置
        angles = np.linspace(0, 2 * np.pi, len(self.selected_features), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        for i, (cluster_id, cluster_mean) in enumerate(zip(sorted(np.unique(self.cluster_labels)), cluster_means)):
            ax = axes[i]
            
            # 数据闭合
            values = cluster_mean.tolist()
            values += values[:1]
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2, label=f'簇 {cluster_id}', color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([f.replace('_', '\n') for f in self.selected_features], fontsize=8)
            ax.set_title(f'簇 {cluster_id} 特征分布', fontsize=12, pad=20)
            ax.grid(True)
        
        plt.tight_layout()
        radar_path = os.path.join(self.output_dir, 'cluster_radar_charts.png')
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"雷达图保存至: {radar_path}")
    
    def _create_correlation_heatmap(self):
        """创建特征相关性热图"""
        plt.figure(figsize=(10, 8))
        
        # 计算相关性矩阵
        correlation_matrix = self.df_features.corr()
        
        # 创建热图
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.3f', cbar_kws={'label': '相关系数'})
        
        plt.title('6个EEG特征相关性热图', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        heatmap_path = os.path.join(self.output_dir, 'features_correlation_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"相关性热图保存至: {heatmap_path}")
    
    def _create_pca_visualization(self, colors):
        """创建PCA降维可视化"""
        # 执行PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(self.df_scaled)
        
        plt.figure(figsize=(12, 8))
        
        # 绘制散点图
        for cluster_id in sorted(np.unique(self.cluster_labels)):
            cluster_mask = self.cluster_labels == cluster_id
            plt.scatter(pca_result[cluster_mask, 0], pca_result[cluster_mask, 1],
                       c=colors[cluster_id], label=f'簇 {cluster_id}', 
                       alpha=0.7, s=60)
        
        # 添加受试者标签
        for i, subject in enumerate(self.df_features.index):
            plt.annotate(subject, (pca_result[i, 0], pca_result[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差解释)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差解释)')
        plt.title(f'PCA降维可视化 (总方差解释: {pca.explained_variance_ratio_.sum():.1%})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        pca_path = os.path.join(self.output_dir, 'pca_visualization.png')
        plt.savefig(pca_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PCA可视化保存至: {pca_path}")
    
    def _create_tsne_visualization(self, colors):
        """创建t-SNE降维可视化"""
        # 执行t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.df_scaled)-1))
        tsne_result = tsne.fit_transform(self.df_scaled)
        
        plt.figure(figsize=(12, 8))
        
        # 绘制散点图
        for cluster_id in sorted(np.unique(self.cluster_labels)):
            cluster_mask = self.cluster_labels == cluster_id
            plt.scatter(tsne_result[cluster_mask, 0], tsne_result[cluster_mask, 1],
                       c=colors[cluster_id], label=f'簇 {cluster_id}', 
                       alpha=0.7, s=60)
        
        # 添加受试者标签
        for i, subject in enumerate(self.df_features.index):
            plt.annotate(subject, (tsne_result[i, 0], tsne_result[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.7)
        
        plt.xlabel('t-SNE 维度 1')
        plt.ylabel('t-SNE 维度 2')
        plt.title('t-SNE降维可视化')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        tsne_path = os.path.join(self.output_dir, 'tsne_visualization.png')
        plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE可视化保存至: {tsne_path}")
    
    def _create_feature_boxplots(self, colors):
        """创建特征箱型图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(self.selected_features):
            ax = axes[i]
            
            # 准备数据
            data_by_cluster = []
            labels = []
            for cluster_id in sorted(np.unique(self.cluster_labels)):
                cluster_mask = self.cluster_labels == cluster_id
                cluster_data = self.df_features[cluster_mask][feature].values
                data_by_cluster.append(cluster_data)
                labels.append(f'簇 {cluster_id}')
            
            # 创建箱型图
            box_plot = ax.boxplot(data_by_cluster, labels=labels, patch_artist=True)
            
            # 设置颜色
            for patch, color in zip(box_plot['boxes'], colors[:len(data_by_cluster)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{feature}', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 旋转x轴标签
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('各簇特征分布箱型图', fontsize=16)
        plt.tight_layout()
        
        boxplot_path = os.path.join(self.output_dir, 'feature_boxplots.png')
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"箱型图保存至: {boxplot_path}")
    
    def _create_cluster_centers_heatmap(self, colors):
        """创建聚类中心热图"""
        # 计算聚类中心
        cluster_centers = []
        cluster_labels_list = []
        
        for cluster_id in sorted(np.unique(self.cluster_labels)):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_center = self.df_scaled[cluster_mask].mean()
            cluster_centers.append(cluster_center.values)
            cluster_labels_list.append(f'簇 {cluster_id}')
        
        cluster_centers_df = pd.DataFrame(
            cluster_centers,
            columns=self.selected_features,
            index=cluster_labels_list
        )
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(cluster_centers_df, annot=True, cmap='RdBu_r', center=0,
                   fmt='.3f', cbar_kws={'label': '标准化特征值'})
        
        plt.title('聚类中心特征热图', fontsize=14, pad=20)
        plt.xlabel('EEG特征')
        plt.ylabel('聚类簇')
        plt.xticks(rotation=45, ha='right')
        
        centers_path = os.path.join(self.output_dir, 'cluster_centers_heatmap.png')
        plt.savefig(centers_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"聚类中心热图保存至: {centers_path}")
    
    def generate_report(self):
        """生成分析报告"""
        print("=== 生成分析报告 ===")
        
        report_content = f"""# EEG 6特征聚类分析报告

**分析时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

**数据概况**: {len(self.df_features)}个受试者, 6个EEG特征

## 1. 选择的EEG特征

本次分析选择了以下6个关键脑电特征：

"""
        
        # 添加特征说明
        feature_descriptions = {
            'psd_alpha_frontal': 'α波前额叶功率谱密度 - 反映注意力和觉醒状态',
            'rel_power_beta_parietal': 'β波顶叶相对功率 - 反映认知处理和执行功能',
            'psd_theta_temporal': 'θ波颞叶功率谱密度 - 反映记忆处理和情绪状态',
            'conn_frontal_parietal': '前额-顶叶连接性 - 反映执行控制网络连接强度',
            'clustering': '聚类系数 - 反映脑网络的局部连接密度',
            'rel_power_gamma_frontal': 'γ波前额叶相对功率 - 反映高级认知功能和注意力绑定'
        }
        
        for i, feature in enumerate(self.selected_features, 1):
            description = feature_descriptions.get(feature, '脑电特征')
            report_content += f"{i}. **{feature}**: {description}\n"
        
        # 添加聚类结果
        silhouette_avg = silhouette_score(self.df_scaled, self.cluster_labels)
        calinski_harabasz = calinski_harabasz_score(self.df_scaled, self.cluster_labels)
        davies_bouldin = davies_bouldin_score(self.df_scaled, self.cluster_labels)
        
        report_content += f"""

## 2. 聚类分析结果

- **聚类算法**: Gaussian Mixture Model (full covariance)
- **簇数**: 4
- **Silhouette Score**: {silhouette_avg:.3f}
- **Calinski-Harabasz Index**: {calinski_harabasz:.3f}
- **Davies-Bouldin Index**: {davies_bouldin:.3f}
- **AIC**: {self.gmm.aic(self.df_scaled):.2f}
- **BIC**: {self.gmm.bic(self.df_scaled):.2f}

### 簇分布

"""
        
        # 添加各簇分布信息
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            percentage = count / len(self.cluster_labels) * 100
            report_content += f"- **簇 {cluster_id}**: {count}个样本 ({percentage:.1f}%)\n"
        
        # 添加各簇特征分析
        report_content += "\n## 3. 各簇特征分析\n\n"
        
        for _, row in self.df_cluster_stats.iterrows():
            cluster_id = int(row['cluster'])
            count = int(row['count'])
            percentage = row['percentage']
            
            report_content += f"### 簇 {cluster_id} ({count}个样本, {percentage:.1f}%)\n\n"
            
            # 找出该簇的突出特征
            feature_highlights = []
            for feature in self.selected_features:
                mean_val = row[f'{feature}_mean']
                overall_mean = self.df_features[feature].mean()
                if mean_val > overall_mean * 1.2:
                    feature_highlights.append(f"- **{feature}**: 高于平均 ({mean_val:.3f} vs {overall_mean:.3f})")
                elif mean_val < overall_mean * 0.8:
                    feature_highlights.append(f"- **{feature}**: 低于平均 ({mean_val:.3f} vs {overall_mean:.3f})")
            
            if feature_highlights:
                report_content += "**突出特征**:\n"
                for highlight in feature_highlights:
                    report_content += f"{highlight}\n"
            else:
                report_content += "**特征特点**: 各项指标接近整体平均水平\n"
            
            report_content += "\n"
        
        # 添加方法说明
        report_content += """## 4. 分析方法

1. **数据预处理**: 标准化处理，确保各特征具有相同的量纲
2. **聚类算法**: 高斯混合模型（GMM，full covariance），K=4
3. **评价指标**: 
   - Silhouette Score: 衡量簇内相似性和簇间差异性
   - Calinski-Harabasz Index: 衡量簇间分离度和簇内紧密度
   - Davies-Bouldin Index: 衡量簇的紧密性和分离性
4. **可视化**: PCA降维、t-SNE降维、雷达图、箱型图等

## 5. 结论

本次分析成功将40个受试者基于6个关键EEG特征分为4个不同的群体，每个群体在脑电特征上表现出不同的模式，这些差异可能反映了不同的认知状态、注意力水平或神经活动模式。

## 6. 文件说明

- `eeg_6features_raw_data.csv`: 原始6特征数据
- `eeg_6features_scaled_data.csv`: 标准化后的数据
- `clustering_results.csv`: 聚类结果
- `cluster_statistics.csv`: 各簇统计信息
- `cluster_radar_charts.png`: 雷达图
- `features_correlation_heatmap.png`: 特征相关性热图
- `pca_visualization.png`: PCA降维可视化
- `tsne_visualization.png`: t-SNE降维可视化
- `feature_boxplots.png`: 特征分布箱型图
- `cluster_centers_heatmap.png`: 聚类中心热图
"""
        
        # 保存报告
        report_path = os.path.join(self.output_dir, 'analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"分析报告保存至: {report_path}")
        print()
    
    def run_complete_analysis(self):
        """运行完整的聚类分析"""
        print("开始完整的EEG 6特征聚类分析...")
        print("=" * 60)
        
        try:
            # 1. 加载数据
            self.load_data()
            
            # 2. 数据预处理
            self.preprocess_data()
            
            # 3. 执行聚类
            self.perform_clustering(k=4)
            
            # 4. 分析簇特征
            self.analyze_clusters()
            
            # 5. 创建可视化
            self.create_visualizations()
            
            # 6. 生成报告
            self.generate_report()
            
            print("=" * 60)
            print("🎉 EEG 6特征聚类分析完成！")
            print(f"📁 所有结果保存在: {self.output_dir}")
            print("=" * 60)
            
            return True
            
        except Exception as e:
            print(f"❌ 分析过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(
        description="EEG 6特征聚类分析（3.7.2，GMM K=4）"
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="65维EEG特征CSV路径（首列为被试ID，需包含6个关键特征）"
    )
    parser.add_argument("--n-init", type=int, default=20, help="GMM n_init")
    parser.add_argument("--max-iter", type=int, default=200, help="GMM max_iter")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    print("EEG 6特征聚类分析")
    print("=" * 50)

    # 创建分析实例
    analyzer = EEG6FeaturesClustering(
        data_file=args.data,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.seed,
    )

    # 运行完整分析
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n✅ 分析成功完成！")
        print(f"📊 查看结果目录: {analyzer.output_dir}")
    else:
        print("\n❌ 分析失败，请检查错误信息")

if __name__ == "__main__":
    main()

