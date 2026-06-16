"""
IQL策略分析报告生成器
=====================

本脚本加载IQL模型与瞳孔数据，按固定决策间隔执行在线推理，
记录并分析每个决策时间点的模型输出（Q值/策略概率/状态值/优势函数），
据此生成策略分析报告与可视化。

依赖输入：
1) IQL模型 checkpoint（由 iql_v2/train_neural_feedback.py 产生），
   状态为10维瞳孔状态。
2) 瞳孔数据CSV（每名被试一个文件），至少包含列：
       timestamp, PupilDiameter
3) 可选：被试→认知亚型 的映射JSON（来自3.7.2分类器输出），用于分组对比。

输出：
- 各被试的IQL决策详细日志（JSON/CSV）
- 策略分析可视化与综合文字报告
"""

import argparse
import glob
import json
import os
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 允许从 iql_v2 扁平导入（PupilState / IQL算法）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_IQL_V2_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "iql_v2"))
if _IQL_V2_DIR not in sys.path:
    sys.path.insert(0, _IQL_V2_DIR)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from iql_realtime_output_design import (  # noqa: E402
    PupilState,
    FeedbackAction,
)
from neural_feedback_iql import ImplicitQLearning, IQLConfig  # noqa: E402

# 论文表4-3的5维动作空间
ACTION_NAMES = [a.name for a in FeedbackAction]  # NONE/VISUAL/AUDITORY/HAPTIC/MULTIMODAL
ACTION_COLORS = ['#95a5a6', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']


@dataclass
class IQLDecision:
    """单次IQL决策记录"""
    timestamp: float
    step_index: int
    pupil_diameter: float
    pupil_relative_change: float
    pupil_trend: float
    action: int
    action_name: str
    q_values: List[float]
    q_value_selected: float
    q_value_max: float
    q_value_std: float
    policy_probs: List[float]
    policy_entropy: float
    policy_confidence: float
    v_value: float
    advantage: float


# ===================== 模型封装 =====================

class IQLPolicyModel:
    """加载IQL模型，提供 Q/V/策略概率 的前向推理。"""

    def __init__(self, model_path: str, device: str = 'cpu'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"未找到IQL模型: {model_path}\n"
                f"请提供 iql_v2/train_neural_feedback.py 生成的 .pth 模型。"
            )
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)

        # 优先使用checkpoint内保存的配置以匹配网络结构
        cfg = checkpoint.get('config', None)
        if isinstance(cfg, IQLConfig):
            self.config = cfg
            self.config.device = str(self.device)
        else:
            raise ValueError(
                "模型checkpoint缺少IQLConfig配置，无法还原网络结构。"
            )

        self.agent = ImplicitQLearning(self.config)
        self.agent.load_model(model_path)
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        if self.state_dim != 10:
            raise ValueError(
                f"当前模型为 {self.state_dim} 维，请使用10维模型。"
            )

    def _build_model_state(self, pupil_state: PupilState) -> np.ndarray:
        return pupil_state.to_vector()

    def infer_from_pupil(self, pupil_state: PupilState) -> Dict[str, np.ndarray]:
        """对10维瞳孔状态做前向推理（论文式4-9）。"""
        state_vec = self._build_model_state(pupil_state)
        q_values = self.agent.get_q_values(state_vec)
        policy_probs = self.agent.get_policy_probs(state_vec)
        st = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v_value = float(self.agent.v_network(st).cpu().numpy().flatten()[0])
        return {"q_values": q_values, "policy_probs": policy_probs, "v_value": v_value}


# ===================== 瞳孔数据处理 =====================

def load_pupil_data(pupil_file: str) -> Optional[pd.DataFrame]:
    """加载瞳孔数据CSV（需含 timestamp, PupilDiameter）。"""
    if not os.path.exists(pupil_file):
        print(f"警告: 瞳孔文件不存在 - {pupil_file}")
        return None
    df = pd.read_csv(pupil_file)
    if 'PupilDiameter' not in df.columns or 'timestamp' not in df.columns:
        raise ValueError(
            f"{pupil_file} 缺少必要列 timestamp / PupilDiameter。现有列: {list(df.columns)}"
        )
    # 统一换算为毫米（若原始单位为米）
    if df['PupilDiameter'].abs().median() < 0.1:
        df['pupil_mm'] = df['PupilDiameter'] * 1000.0
    else:
        df['pupil_mm'] = df['PupilDiameter']
    return df


def build_pupil_state(
    pupil_df: pd.DataFrame,
    t: float,
    baseline: float,
    time_since_last_feedback: float,
    recent_feedback_count: int,
) -> Optional[PupilState]:
    """从瞳孔时序在时间点 t 构造10维瞳孔状态向量（论文式4-9）。"""
    idx = (pupil_df['timestamp'] - t).abs().idxmin()
    current = float(pupil_df.loc[idx, 'pupil_mm'])

    win1 = pupil_df[(pupil_df['timestamp'] >= t - 1.0) & (pupil_df['timestamp'] <= t)]['pupil_mm']
    win5 = pupil_df[(pupil_df['timestamp'] >= t - 5.0) & (pupil_df['timestamp'] <= t)]['pupil_mm']
    if len(win5) == 0:
        return None

    mean_1s = float(win1.mean()) if len(win1) > 0 else current
    mean_5s = float(win5.mean())
    std_win = float(win5.std()) if len(win5) > 1 else 0.0

    # 变化速率：相邻帧差分均值 (mm/s)
    if len(win1) > 2:
        dt = np.diff(win1.index.to_series().map(pupil_df['timestamp']))
        diffs = np.diff(win1.values)
        change_rate = float(np.mean(diffs / np.where(dt == 0, 1e-3, dt)))
    else:
        change_rate = 0.0

    rel_change = (current - baseline) / baseline * 100.0 if baseline > 0 else 0.0

    # 趋势：窗口内线性拟合斜率，归一化到[-1,1]
    if len(win5) > 2:
        x = np.arange(len(win5))
        slope = np.polyfit(x, win5.values, 1)[0]
        trend = float(np.clip(slope * 10, -1, 1))
    else:
        trend = 0.0

    return PupilState(
        pupil_diameter=current,
        pupil_diameter_baseline=baseline,
        pupil_change_rate=change_rate,
        pupil_relative_change=rel_change,
        pupil_std_window=std_win,
        pupil_mean_1s=mean_1s,
        pupil_mean_5s=mean_5s,
        pupil_trend=trend,
        time_since_last_feedback=time_since_last_feedback,
        recent_feedback_count=recent_feedback_count,
    )


def run_inference_on_subject(
    pupil_df: pd.DataFrame,
    model: IQLPolicyModel,
    subject_id: str,
    group_type: str,
    decision_interval_s: float = 1.0,
    min_feedback_interval_s: float = 5.0,
) -> Optional[Dict]:
    """对单名被试的瞳孔时序按固定决策间隔执行模型推理。"""
    t_start = float(pupil_df['timestamp'].min())
    t_end = float(pupil_df['timestamp'].max())

    # 基线：任务前静息5秒均值（论文 d_base）
    base_mask = pupil_df['timestamp'] < (t_start + 5.0)
    baseline = float(pupil_df[base_mask]['pupil_mm'].mean())
    if not np.isfinite(baseline) or baseline <= 0:
        baseline = float(pupil_df['pupil_mm'].median())

    decisions: List[IQLDecision] = []
    last_feedback_t = -np.inf
    recent_feedback_times: List[float] = []

    step = 0
    t = t_start + 5.0  # 跳过基线期
    while t <= t_end:
        recent_feedback_times = [ft for ft in recent_feedback_times if t - ft <= 10.0]
        time_since = t - last_feedback_t if np.isfinite(last_feedback_t) else 999.0

        state = build_pupil_state(
            pupil_df, t, baseline, time_since, len(recent_feedback_times)
        )
        if state is None:
            t += decision_interval_s
            continue

        out = model.infer_from_pupil(state)
        q_values = np.asarray(out["q_values"], dtype=float)
        policy_probs = np.asarray(out["policy_probs"], dtype=float)
        v_value = float(out["v_value"])

        # 在线部署采用贪心策略（论文式4-25）
        action = int(np.argmax(q_values))
        entropy = float(-np.sum(policy_probs * np.log(policy_probs + 1e-8)))

        # 安全频率限制（论文4.4.3）：两次反馈最小间隔
        executed = action != 0 and time_since >= min_feedback_interval_s
        if executed:
            last_feedback_t = t
            recent_feedback_times.append(t)

        decisions.append(IQLDecision(
            timestamp=t,
            step_index=step,
            pupil_diameter=state.pupil_diameter,
            pupil_relative_change=state.pupil_relative_change,
            pupil_trend=state.pupil_trend,
            action=action,
            action_name=ACTION_NAMES[action],
            q_values=q_values.tolist(),
            q_value_selected=float(q_values[action]),
            q_value_max=float(q_values.max()),
            q_value_std=float(q_values.std()),
            policy_probs=policy_probs.tolist(),
            policy_entropy=entropy,
            policy_confidence=float(policy_probs.max()),
            v_value=v_value,
            advantage=float(q_values[action] - v_value),
        ))
        step += 1
        t += decision_interval_s

    if not decisions:
        return None

    action_distribution = {a: sum(1 for d in decisions if d.action_name == a) for a in ACTION_NAMES}
    return {
        'subject_id': subject_id,
        'group_type': group_type,
        'start_time': t_start,
        'end_time': t_end,
        'decisions': decisions,
        'num_decisions': len(decisions),
        'action_distribution': action_distribution,
        'mean_q_value': float(np.mean([d.q_value_selected for d in decisions])),
        'mean_policy_entropy': float(np.mean([d.policy_entropy for d in decisions])),
        'mean_advantage': float(np.mean([d.advantage for d in decisions])),
        'mean_confidence': float(np.mean([d.policy_confidence for d in decisions])),
    }


# ===================== 可视化与报告 =====================

def plot_group_action_distribution(all_results: List[Dict], output_path: Path):
    groups = sorted(set(r['group_type'] for r in all_results))
    x = np.arange(len(ACTION_NAMES))
    width = 0.8 / max(len(groups), 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for gi, g in enumerate(groups):
        grp = [r for r in all_results if r['group_type'] == g]
        counts = [sum(r['action_distribution'].get(a, 0) for r in grp) for a in ACTION_NAMES]
        total = sum(counts) or 1
        pct = [c / total * 100 for c in counts]
        ax.bar(x + gi * width, pct, width, label=g, alpha=0.85, edgecolor='black')
    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels(ACTION_NAMES, fontsize=9)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Action Distribution by Group')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout(); fig.savefig(output_path, dpi=150, facecolor='white'); plt.close(fig)


def plot_decision_dynamics(all_results: List[Dict], output_path: Path):
    all_decisions = [d for r in all_results for d in r['decisions']]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    q_sel = [d.q_value_selected for d in all_decisions]
    axes[0, 0].hist(q_sel, bins=30, color='#3498db', alpha=0.8)
    axes[0, 0].set_title('Selected Q-Value Distribution'); axes[0, 0].set_xlabel('Q(s,a)')

    ent = [d.policy_entropy for d in all_decisions]
    axes[0, 1].hist(ent, bins=30, color='#9b59b6', alpha=0.8)
    axes[0, 1].set_title('Policy Entropy Distribution'); axes[0, 1].set_xlabel('Entropy')

    pupil = [d.pupil_diameter for d in all_decisions]
    colors = [ACTION_COLORS[d.action] for d in all_decisions]
    axes[1, 0].scatter(pupil, q_sel, c=colors, s=20, alpha=0.5)
    axes[1, 0].set_xlabel('Pupil Diameter (mm)'); axes[1, 0].set_ylabel('Q(s,a)')
    axes[1, 0].set_title('Pupil vs Q-Value')

    adv = [d.advantage for d in all_decisions]
    axes[1, 1].hist(adv, bins=30, color='#27ae60', alpha=0.8)
    axes[1, 1].axvline(0, color='red', linestyle='--')
    axes[1, 1].set_title('Advantage A(s,a) Distribution'); axes[1, 1].set_xlabel('Advantage')

    fig.suptitle('IQL Decision Dynamics', fontsize=14, fontweight='bold')
    fig.tight_layout(); fig.savefig(output_path, dpi=150, facecolor='white'); plt.close(fig)


def generate_text_report(all_results: List[Dict], output_path: Path) -> str:
    report = []
    report.append("=" * 80)
    report.append("        IQL策略分析报告")
    report.append("=" * 80)
    report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"被试数: {len(set(r['subject_id'] for r in all_results))}, "
                  f"试次数: {len(all_results)}")

    all_decisions = [d for r in all_results for d in r['decisions']]
    total = len(all_decisions)
    report.append(f"\n总决策次数: {total}")

    report.append("\n动作分布:")
    for a in ACTION_NAMES:
        cnt = sum(1 for d in all_decisions if d.action_name == a)
        report.append(f"  {a:12s}: {cnt:5d} ({cnt / total * 100:5.1f}%)")

    report.append(f"\n平均Q值: {np.mean([d.q_value_selected for d in all_decisions]):.4f}")
    report.append(f"平均策略熵: {np.mean([d.policy_entropy for d in all_decisions]):.4f}")
    report.append(f"平均置信度: {np.mean([d.policy_confidence for d in all_decisions]):.2%}")
    report.append(f"正优势决策比例: "
                  f"{np.mean([1 for d in all_decisions if d.advantage > 0]) * 100:.1f}%")

    groups = sorted(set(r['group_type'] for r in all_results))
    if len(groups) > 1:
        report.append("\n各亚型策略差异:")
        for g in groups:
            gd = [d for r in all_results if r['group_type'] == g for d in r['decisions']]
            if gd:
                report.append(f"  [{g}] 决策{len(gd)}次, "
                              f"平均Q={np.mean([d.q_value_selected for d in gd]):.3f}, "
                              f"平均熵={np.mean([d.policy_entropy for d in gd]):.3f}")

    report.append("\n" + "=" * 80)
    text = '\n'.join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return text


def parse_subject_id(filename: str) -> Optional[str]:
    """从文件名解析被试ID（形如 ..._S01_... 或 pupil_S01.csv）。"""
    import re
    m = re.search(r'(S\d{2,})', filename)
    return m.group(1) if m else None


def main():
    parser = argparse.ArgumentParser(
        description="IQL策略分析报告"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="IQL模型 .pth 路径")
    parser.add_argument("--pupil-dir", type=str, required=True,
                        help="瞳孔数据目录（每名被试一个CSV，含 timestamp,PupilDiameter）")
    parser.add_argument("--pupil-glob", type=str, default="*.csv",
                        help="瞳孔文件匹配模式")
    parser.add_argument("--group-map", type=str, default=None,
                        help="可选：被试→亚型 映射JSON（来自3.7.2分类器）")
    parser.add_argument("--output-dir", type=str, default="iql_analysis_report")
    parser.add_argument("--decision-interval", type=float, default=1.0,
                        help="决策间隔（秒）")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("加载IQL模型...")
    model = IQLPolicyModel(args.model)
    print(f"  模型状态维度={model.state_dim}, 动作数={model.action_dim}")

    group_map = {}
    if args.group_map and os.path.exists(args.group_map):
        with open(args.group_map, 'r', encoding='utf-8') as f:
            group_map = json.load(f)

    pupil_files = sorted(glob.glob(str(Path(args.pupil_dir) / args.pupil_glob)))
    if not pupil_files:
        raise FileNotFoundError(f"在 {args.pupil_dir} 未找到瞳孔数据文件")

    all_results = []
    for pf in pupil_files:
        sid = parse_subject_id(os.path.basename(pf)) or os.path.basename(pf)
        pupil_df = load_pupil_data(pf)
        if pupil_df is None:
            continue
        group_type = group_map.get(sid, 'unknown')
        res = run_inference_on_subject(
            pupil_df, model, sid, group_type,
            decision_interval_s=args.decision_interval
        )
        if res:
            all_results.append(res)
            print(f"  ✓ {sid}: {res['num_decisions']} 次决策 (亚型: {group_type})")

    if not all_results:
        print("错误: 没有成功处理任何被试数据。")
        return

    # 保存决策数据
    json_path = out_dir / "iql_decisions_data.json"
    json_data = []
    for r in all_results:
        rc = {k: v for k, v in r.items() if k != 'decisions'}
        rc['decisions'] = [asdict(d) for d in r['decisions']]
        json_data.append(rc)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    plot_group_action_distribution(all_results, out_dir / "group_action_distribution.png")
    plot_decision_dynamics(all_results, out_dir / "decision_dynamics.png")
    report_text = generate_text_report(all_results, out_dir / "iql_strategy_analysis_report.txt")

    print("\n" + report_text[:1200])
    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
