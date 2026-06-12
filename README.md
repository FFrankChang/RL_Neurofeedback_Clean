# CARLA 驾驶实验与瞳孔反应分析

本项目整理了 CARLA 驾驶实验数据、瞳孔直径数据，以及用于复现实验汇总、被试聚类和 IQL 训练的分析代码。仓库面向研究复现与结果交流，重点回答三个问题：

1. 不同实验条件下，驾驶行为和瞳孔反应有什么差异？
2. 基于驾驶行为与瞳孔特征，4 名被试是否呈现不同群体模式？
3. 如何从真实记录的 RL 试次中构造 IQL 训练样本？

## 数据说明

`data/carla_data/` 保存 CARLA 驾驶记录。文件名包含被试编号、实验条件和采集时间，例如：

```text
carla_c06_S01_D01_RL_20260119114024.csv
```

主要字段包括：

- `timestamp`: Unix 时间戳
- `Speed`: 车辆速度
- `Location_x`, `Location_y`, `Location_z`: 车辆位置
- `Steer`, `Throttle`, `Brake`: 控制输入
- `Collision_Time`: 碰撞记录
- `Feedback_Marker`: RL 条件下的反馈开始/结束标记

`data/pupil_data/` 保存被试 S01-S04 的瞳孔直径时间序列：

- `timestamp`: Unix 时间戳
- `PupilDiameter`: 瞳孔直径，分析脚本中会转换为毫米

本项目包含三类实验条件：

- `RL`: 个性化 RL 反馈条件
- `feedback`: 固定反馈条件
- `silence`: 无反馈条件

## 目录结构

```text
clean_final_repository/
├── data/
│   ├── carla_data/          # CARLA 驾驶实验 CSV
│   └── pupil_data/          # 瞳孔直径 CSV
├── analysis/
│   ├── analyze_carla_pupil.py
│   ├── clustering/
│   │   └── cluster_subjects.py
│   └── iql/
│       ├── iql_core.py
│       └── train_iql_from_carla.py
├── results/
│   ├── final_analysis_summary.md
│   ├── trial_summary.csv
│   ├── condition_summary.csv
│   ├── plots/
│   └── clustering/
└── requirements.txt
```

## 环境安装

建议使用 Python 3.10 或更高版本。

```bash
pip install -r requirements.txt
```

如果只运行数据汇总和聚类分析，需要 `pandas`、`numpy`、`matplotlib`。如果运行 IQL 训练，还需要 `torch`。

## 分析流程

### 1. 生成驾驶与瞳孔汇总

该步骤会读取 CARLA 与 pupil 数据，按试次计算驾驶行为、反馈事件和瞳孔统计指标。

```bash
python analysis/analyze_carla_pupil.py --data-dir data --output-dir results
```

主要输出：

- `results/trial_summary.csv`: 每个试次的速度、控制输入、反馈事件、瞳孔均值等指标
- `results/condition_summary.csv`: 按 `RL`、`feedback`、`silence` 汇总的条件级指标
- `results/final_analysis_summary.md`: 简短结果报告
- `results/plots/`: 条件对比图

当前汇总结果显示：

- 共 50 个试次
- 覆盖 S01-S04 共 4 名被试
- RL 条件下检测到 34 次反馈事件
- RL 条件平均反馈时长占试次时长约 15.0%

### 2. 运行被试聚类

该步骤使用 `trial_summary.csv` 中已有的试次特征，聚合到被试级别后进行 K-means 聚类，并输出 PCA 可视化。

```bash
python analysis/clustering/cluster_subjects.py \
  --input results/trial_summary.csv \
  --output-dir results/clustering \
  --n-clusters 4
```

主要输出：

- `results/clustering/subject_features.csv`: 被试级特征表
- `results/clustering/subject_clusters.csv`: 被试聚类标签
- `results/clustering/cluster_metrics.csv`: 聚类指标
- `results/clustering/subject_clusters_pca.png`: PCA 可视化图

默认使用的聚类特征包括平均速度、最大速度、方向盘输入、油门、刹车、反馈次数、反馈时长、瞳孔均值和瞳孔标准差。

### 3. 运行 IQL 训练

IQL 训练入口会读取真实记录的 RL 试次，将 CARLA 状态与瞳孔数据按时间戳对齐，并构造离线 transition：

```bash
python analysis/iql/train_iql_from_carla.py \
  --data-dir data \
  --output-dir results/iql_training
```

状态向量包含：

- 车辆速度、方向盘、油门、刹车
- 瞳孔直径
- 瞳孔变化量
- 速度变化量
- 最近反馈比例

动作定义：

- `0`: 保持当前状态
- `1`: 反馈开始
- `2`: 反馈结束

训练完成后会输出：

- `results/iql_training/iql_model.pt`: IQL 模型参数
- `results/iql_training/training_history.json`: 训练损失记录
- `results/iql_training/training_metadata.json`: 状态字段和动作映射说明

## 关键结果文件

建议优先阅读以下文件：

- `results/final_analysis_summary.md`: 总体结果摘要
- `results/condition_summary.csv`: 不同实验条件的汇总指标
- `results/trial_summary.csv`: 试次级完整指标
- `results/clustering/subject_clusters.csv`: 被试聚类结果
- `results/plots/mean_speed_by_condition.png`: 不同条件下平均速度对比
- `results/plots/pupil_by_condition.png`: 不同条件下瞳孔直径对比
- `results/plots/rl_feedback_events.png`: RL 试次中的反馈事件数量

## 方法说明

驾驶与瞳孔分析以时间戳为核心，将 CARLA 数据和 pupil 数据对齐。RL 条件下，脚本会识别 `Feedback_Marker` 中的反馈开始和结束标记，并计算反馈事件数量、反馈时长和反馈比例。

聚类分析不依赖额外标签，而是直接从试次汇总表中提取被试级行为与生理特征。当前样本量为 4 名被试，因此聚类结果主要用于展示个体差异和后续分组分析流程。

IQL 训练代码采用离线数据构造方式，不需要额外交互环境。奖励函数综合考虑速度、方向盘幅度、刹车和瞳孔偏离目标值的程度，可根据后续研究目标进一步调整。

## 复现顺序

从空的 `results/` 目录开始，推荐按以下顺序运行：

```bash
python analysis/analyze_carla_pupil.py --data-dir data --output-dir results
python analysis/clustering/cluster_subjects.py --input results/trial_summary.csv --output-dir results/clustering --n-clusters 4
python analysis/iql/train_iql_from_carla.py --data-dir data --output-dir results/iql_training
```

前两个步骤较快，适合检查数据和生成报告。IQL 训练耗时取决于本机硬件和训练轮数，可通过 `--epochs` 和 `--batch-size` 调整。

## 注意事项

- 当前数据覆盖 4 名被试，适合展示分析流程和个体差异，但统计推断应结合更多被试数据进一步验证。
- 聚类数量默认设为 4，是为了对应被试级分组分析流程；如新增被试，可重新设置 `--n-clusters`。
- IQL 训练入口默认只使用 `*RL*.csv` 文件构造训练样本。
