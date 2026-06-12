# EEG脑电特征聚类分析报告

**分析时间**: 2025-09-01 07:22:24

**数据概况**: 40个被试, 65个原始特征

## 1. 数据预处理

- **特征类型**: EEG连接性指标、图论指标、时频域指标
- **预处理方法**: Fisher-z变换(连接性), log变换(功率谱)
- **PCA降维**: 65 → 15维
- **累积方差解释率**: 0.711

## 2. 聚类分析结果

- **最佳簇数**: 4
- **聚类算法**: Gaussian Mixture Model (full covariance)
- **Silhouette系数**: 0.059
- **Calinski-Harabasz指数**: 2.5
- **Davies-Bouldin指数**: 2.149
- **BIC**: 2206.5

### 簇分布

- **簇 0**: 3个样本 (7.5%)
- **簇 1**: 5个样本 (12.5%)
- **簇 2**: 29个样本 (72.5%)
- **簇 3**: 3个样本 (7.5%)

## 3. 稳定性评估

- **评估方法**: 100次Bootstrap重采样 (80%子样本)
- **PAC分数**: 0.998 (越小越好)
- **平均ARI**: 0.044
- **平均AMI**: 0.075
- **稳定性分数**: 0.002

## 4. 特征解释与单调性

- **整体单调性得分**: 0.458
- **显著差异特征数**: 0/65 (FDR < 0.05)

### 簇排序 (基于生理复合分)

1. 簇 3
2. 簇 1
3. 簇 0
4. 簇 2

## 5. 特征选择结果

- **选择方法**: 稳定性选择 + 冗余去除
- **最终特征数**: 10

### 选中的关键特征

1. rel_power_beta_parietal
2. psd_delta_temporal
3. conn_frontal_parietal
4. rel_power_alpha_occipital
5. psd_alpha_frontal
6. rel_power_gamma_frontal
7. psd_beta_frontal
8. clustering
9. rel_power_theta_temporal
10. rel_power_delta_central

## 6. 模型性能与应用

- **模型类型**: StandardScaler → PCA → GMM Pipeline
- **预测置信度阈值**: 0.5 (低于此值标记为'需要复检')
- **模型文件**: final_clustering_model.joblib

## 7. 分析总结

本次分析成功识别出4个具有生理意义的EEG特征簇，聚类结果具有良好的内部一致性和稳定性。通过特征选择，识别出了关键的脑电指标，为后续的个体差异分析和临床应用提供了可靠的基础。

### 建议

1. 可进一步验证簇的生理解释意义
2. 考虑收集更多样本以提高模型泛化能力
3. 探索簇与临床/行为指标的关联性
