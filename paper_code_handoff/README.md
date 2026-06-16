# 论文相关代码交接说明（Handoff）

本文件夹汇总了论文以下章节所**实际使用**的代码与数据，并与论文中引用的结果版本（结果目录 / 图表）严格对应。

> 涉及论文章节：
> - **3.5.2** 神经反馈个体生理指标特征量化表征
> - **3.7** 基于 EEG 的驾驶人认知状态聚类与分型建模（3.7.1 / 3.7.2）
> - **4.4** 基于眼动数据和强化学习的实时神经反馈策略模型（4.4.1 / 4.4.2 / 4.4.3）
> - **4.5** 在线应用性能与系统可实施性分析（4.5.1 / 4.5.2 / 4.5.3）

---

## 1. 目录结构

```
paper_code_handoff/
├── README.md                       本说明文件
├── requirements.txt                运行依赖
│
├── cognitive_load_3.5.2/           【3.5.2 认知负荷量化】
│   ├── arousal_classification.py         眼动→认知负荷二分类建模（XGBoost主 + SVM/RF基线）
│   └── iql_strategy_analysis_report.py   IQL模型对瞳孔数据推理 → 策略分析报告
│
├── clustering_3.7/                 【3.7 EEG 聚类与分型】
│   ├── eeg_clustering_analysis.py        3.7.1 65维特征（连接性+图论+频谱）→ GMM(K=4) 主管线
│   ├── eeg_6_features_clustering.py      3.7 6项关键特征 GMM(K=4, full)，生成雷达/箱线/热力图
│   ├── SVM_cluster_v2.py                 3.7.2 黎曼切空间 + LOOCV 分类器（4类亚型识别）
│   └── data/                             仅含特征字段字典（eeg_features_column_descriptions.csv），真实数据自备（见第4节）
│
├── iql_v2/                         【4.4 / 4.5 强化学习核心包】（保持包结构以保证 import 可用）
│   ├── neural_feedback_env.py            状态/动作/奖励 + EEG 指标公式（含 3.5.2 脑电认知负荷公式）
│   ├── neural_feedback_iql.py            IQL 算法：Dueling Q / V / Policy、期望分位/AWR 损失
│   ├── train_neural_feedback.py          4.4.2 训练流程（收敛判据、早停、评估）
│   ├── group_wrappers.py                 4.4.1 群体个性化奖励（GroupRewardWrapper）
│   ├── adaptive_weight_environment.py    4.4.1 自适应奖励权重环境
│   ├── step_weight_strategy.py           权重调度策略（线性/指数/余弦/sigmoid/自适应）
│   ├── compare_reward_experiments.py     奖励消融对比（DQN 基线，adaptive 依赖）
│   ├── train_with_adaptive_weights.py    自适应权重训练入口
│   ├── iql_realtime_output_design.py     4.4.2 在线部署：PupilState(10维) + 实时推理与日志
│   └── __init__.py
│
├── iql_demo/                       【4.4.2 训练曲线绘制】
│   └── plot_iql_training.py              读取 training_stats.pkl 绘制训练曲线
│
└── figures_and_paper/             【论文图表与文稿】
    ├── generate_thesis_figures.py        生成 图3-5/3-6/3-12/4-5/4-6/4-7
    ├── regenerate_figures_3_8_to_3_11.py 重生成 图3-8~3-11（写入 6features 结果目录）
    ├── md_to_docx.py                     Markdown → Word（自动嵌入配图）
    ├── thesis_sections_draft.md          论文章节 Markdown 源稿
    └── thesis_sections_draft.docx        论文章节 Word 成稿
```

> 说明：图表为派生产物，不随交接包附带；请用真实数据运行各管线后，由
> `generate_thesis_figures.py` / `regenerate_figures_3_8_to_3_11.py` 重新生成（输出目录见第3节）。

---

## 2. 论文章节 ↔ 代码文件 对应表

| 论文章节 | 内容 | 对应代码文件 |
|:---|:---|:---|
| 3.5.2 一 | EEG 认知负荷计算公式（MWI/EI/DI/AL、arousal 融合） | `iql_v2/neural_feedback_env.py`（`_simulate_eeg_metrics` / `_calculate_arousal`） |
| 3.5.2 二 | 遥测式眼动特征提取（瞳孔直径/基线/相对变化/趋势） | `cognitive_load_3.5.2/iql_strategy_analysis_report.py` |
| 3.5.2 二 | 眼动→认知负荷预测建模（XGBoost主 + SVM/RF基线） | `cognitive_load_3.5.2/arousal_classification.py`（需眼动特征+MWI标签） |
| 3.5.2 二 | 在线 10 维瞳孔状态定义（PupilState） | `iql_v2/iql_realtime_output_design.py` |
| 3.7.1 | 65 维特征（连接性 Fisher-z + 图论 + 频谱）→ GMM(K=4) | `clustering_3.7/eeg_clustering_analysis.py` |
| 3.7.1 | 黎曼 SPD 切空间映射 + GMM(K=4) | `clustering_3.7/SVM_cluster_v2.py`（`cluster_in_tangent_space`） |
| 3.7.2 | 四类亚型特征画像（雷达/箱线/中心热力图） | `clustering_3.7/eeg_6_features_clustering.py` |
| 3.7.2 | 有监督分类器（黎曼切空间→PCA→LR，LOOCV + 校准） | `clustering_3.7/SVM_cluster_v2.py`（`train_classifier_loocv` / `train_final_model`） |
| 4.4.1 | 状态/动作/奖励设计 | `iql_v2/neural_feedback_env.py` |
| 4.4.1 | 群体个性化奖励权重 | `iql_v2/group_wrappers.py`、`iql_v2/adaptive_weight_environment.py`、`iql_v2/step_weight_strategy.py` |
| 4.4.2 | IQL 算法与训练流程 | `iql_v2/neural_feedback_iql.py`、`iql_v2/train_neural_feedback.py` |
| 4.4.2 | 在线部署与实时日志（统一10维瞳孔状态） | `iql_v2/iql_realtime_output_design.py` |
| 4.4.3 | 安全约束与稳定性机制 | `iql_v2/neural_feedback_iql.py`（梯度裁剪/软更新/PER）、`iql_v2/neural_feedback_env.py`（安全奖励） |
| 4.5 | 性能/延迟/资源分析配图 | `figures_and_paper/generate_thesis_figures.py`（fig9/fig10） |

---

## 3. 图表 ↔ 生成代码 ↔ 结果来源

> 数据类图表由对应管线输出的结果文件绘制；
> `generate_thesis_figures.py` 缺少输入文件时会报错跳过。图4-7为方法示意图（无数据）。

| 图号 | 文件 | 生成代码 | 数据来源（结果文件） |
|:---:|:---|:---|:---|
| 图3-5 | fig1_feature_importance.png | `generate_thesis_figures.py --importance-json` | `arousal_classification_results_*/feature_importance.json` |
| 图3-6 | fig2_roc_curves.png | `generate_thesis_figures.py --roc-json` | `arousal_classification_results_*/roc_data.json` |
| 图3-7 | k_comparison_table.png | `eeg_clustering_analysis.py` | `eeg_clustering_analysis_<ts>/` |
| 图3-8 | pca_visualization.png | `eeg_6_features_clustering.py` / `regenerate_figures_3_8_to_3_11.py` | `eeg_6features_clustering_<ts>/` |
| 图3-8b | pca_scatter_gmm_contour.png | `eeg_clustering_analysis.py` | `eeg_clustering_analysis_<ts>/` |
| 图3-9 | cluster_radar_charts.png | `eeg_6_features_clustering.py` | `eeg_6features_clustering_<ts>/` |
| 图3-10 | feature_boxplots.png | `eeg_6_features_clustering.py` | `eeg_6features_clustering_<ts>/` |
| 图3-11 | cluster_centers_heatmap.png | `eeg_6_features_clustering.py` | `eeg_6features_clustering_<ts>/` |
| 图3-12 | fig7_confusion_matrix.png | `generate_thesis_figures.py --loocv-json` | `SVM_cluster_v2.py` 输出的 `loocv_results.json` |
| 图4-5 | fig8_iql_training_panel.png | `generate_thesis_figures.py --stats-pkl` | `models/neural_feedback_<ts>/training_stats.pkl` |
| 图4-5附 | 01~08_*.png | `iql_demo/plot_iql_training.py --stats` | 同上 `training_stats.pkl` |
| 图4-6 | fig9_latency_distribution.png | `generate_thesis_figures.py --latency-csv` | 延迟测量CSV |
| 图4-7 | fig10_safety_hierarchy.png | `generate_thesis_figures.py --only fig10` | 方法示意图（无数据） |

---

## 4. 复现所需的数据与格式



| 用途（论文章节） | 脚本 | 需要的数据与格式 |
|:---|:---|:---|
| 3.7.1/3.7.2 聚类 | `eeg_clustering_analysis.py` / `eeg_6_features_clustering.py` | 65维EEG特征CSV：首列被试ID，列名见 `EEGClusteringAnalysis.expected_feature_names()`（连接性10+图论5+频谱50） |
| 3.7.1/3.7.2 黎曼分型 | `SVM_cluster_v2.py` | SPD功能连接张量 `.npy`：形状 `(N, V, C, C)` 或 `(N, C, C)`；可选分型标签 `.npy` |
| 3.5.2 眼动→认知负荷 | `arousal_classification.py` | 样本级CSV：12维眼动特征列（默认见 `DEFAULT_FEATURE_COLS`）+ 标签列（连续MWI用 `--label-is-mwi` 按中位数二值化，或0/1标签） |
| 4.4.2 IQL 策略分析 | `iql_strategy_analysis_report.py` | IQL模型 `.pth` + 瞳孔CSV（含 `timestamp,PupilDiameter`）+ 可选 被试→亚型 映射JSON |
| 4.5 延迟分析 | `generate_thesis_figures.py`（fig9） | 端到端延迟测量CSV：列 `t_acquire,t_feature,t_infer,t_actuate`（ms） |

> 关于 4.4 强化学习环境：`iql_v2/neural_feedback_env.py` 是论文4.4.1所述的训练环境，
> 用于离线训练IQL智能体。训练得到的策略可用于瞳孔数据的在线推理。

---

## 5. 快速运行指引

```bash
# 0) 安装依赖
pip install -r requirements.txt

# 1) 3.7.1 EEG 聚类主管线（65维 → GMM K=4），传入特征CSV
python clustering_3.7/eeg_clustering_analysis.py --data <65维特征.csv>

# 2) 3.7.2 六特征聚类与亚型画像（GMM K=4，雷达/箱线/热力图）
python clustering_3.7/eeg_6_features_clustering.py --data <65维特征.csv> --n-init 20 --max-iter 200

# 3) 3.7.2 有监督四类分类器（黎曼切空间 + LOOCV + 概率校准）
python clustering_3.7/SVM_cluster_v2.py --spd <SPD连接张量.npy> [--labels <分型标签.npy>]

# 4) 3.5.2 眼动→认知负荷分类（XGBoost主 + SVM/RF基线）
python cognitive_load_3.5.2/arousal_classification.py --data <眼动样本.csv> --label-col mwi --label-is-mwi

# 5) 4.4 IQL 训练（从 iql_v2 目录内运行以满足扁平 import）
cd iql_v2
python train_neural_feedback.py
cd ..

# 6) 4.4.2 训练过程图（读取训练统计）
python iql_demo/plot_iql_training.py --stats models/neural_feedback_<ts>/training_stats.pkl

# 7) 4.4.2 IQL 策略分析（模型 + 瞳孔数据，统一10维瞳孔状态）
python cognitive_load_3.5.2/iql_strategy_analysis_report.py \
    --model models/neural_feedback_<ts>/best_model.pth --pupil-dir <瞳孔数据目录>

# 8) 论文图表生成与 Word 导出
python figures_and_paper/generate_thesis_figures.py \
    --importance-json <.../feature_importance.json> --roc-json <.../roc_data.json> \
    --loocv-json <.../loocv_results.json> --stats-pkl <.../training_stats.pkl> \
    --latency-csv <延迟测量.csv>
python figures_and_paper/md_to_docx.py
```

> **import 说明**：`iql_v2/` 内部模块使用扁平导入（如 `from neural_feedback_env import ...`），
> 因此运行 `train_neural_feedback.py` / `train_with_adaptive_weights.py` 等入口时，请在 `iql_v2/` 目录内执行；
> `group_wrappers.py` 使用包内相对导入，作为包 `import iql_v2.group_wrappers` 使用时同样有效。

---

## 6. 未纳入说明（避免版本混淆）

以下脚本与论文当前版本**不直接对应**，为避免交接歧义，未纳入本文件夹：

- `advanced_clustering_pipeline.py`、`sparse_kmeans_clustering.py`、`KMeans_cluster_simple.py`：早期/备选聚类方案，最优 K 非 4 或非脑连接性特征，与论文 3.7 结论不一致。
- `iql_v2/` 内的超参可视化/调参监控等辅助脚本（`hyperparameter_*`、`*_visualizer.py`、`run_group_ablation*` 等）：属探索性工具，非论文正文结论所依赖。

如后续需要这些扩展实验，可从原仓库按需补充。
