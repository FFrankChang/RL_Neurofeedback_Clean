"""
IQL实时实验输出设计
==================
用于实时实验时记录IQL模型的决策过程和状态信息

State: 仅使用 pupil size（瞳孔大小）+ 必要的上下文信息
"""
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import csv
import os
from enum import Enum


class FeedbackAction(Enum):
    """反馈动作类型（论文5.4.1 表5.7）"""
    NONE = 0           # 不干预
    SHORT_5S = 1       # 5秒短时反馈
    LONG_20S = 2       # 20秒长时反馈


@dataclass
class PupilState:
    """
    瞳孔状态 - 简化版State设计
    
    仅使用瞳孔数据作为输入，无脑电
    """
    # 核心瞳孔特征
    pupil_diameter: float           # 当前瞳孔直径 (mm)
    pupil_diameter_baseline: float  # 基线瞳孔直径 (mm)
    pupil_change_rate: float        # 瞳孔变化速率 (mm/s)
    
    # 瞳孔衍生指标
    pupil_relative_change: float    # 相对基线变化比例 (%)
    pupil_std_window: float         # 窗口内瞳孔标准差 (反映波动性)
    
    # 历史特征（用于捕捉时序信息）
    pupil_mean_1s: float            # 过去1秒平均瞳孔大小
    pupil_mean_5s: float            # 过去5秒平均瞳孔大小
    pupil_trend: float              # 瞳孔变化趋势 (-1到1, 负=收缩, 正=扩张)
    
    # 上下文信息（可选但推荐）
    time_since_last_feedback: float  # 距上次反馈的时间 (s)
    recent_feedback_count: int       # 最近N秒内反馈次数
    
    def to_vector(self) -> np.ndarray:
        """转换为模型输入向量"""
        return np.array([
            self.pupil_diameter,
            self.pupil_diameter_baseline,
            self.pupil_change_rate,
            self.pupil_relative_change,
            self.pupil_std_window,
            self.pupil_mean_1s,
            self.pupil_mean_5s,
            self.pupil_trend,
            self.time_since_last_feedback,
            self.recent_feedback_count / 10.0  # 归一化
        ], dtype=np.float32)
    
    @staticmethod
    def get_state_dim() -> int:
        """获取状态维度"""
        return 10


@dataclass
class IQLDecisionOutput:
    """
    IQL模型单次决策的完整输出
    
    这是实时实验时每个决策时间点应该记录的全部信息
    """
    # ========== 时间戳信息 ==========
    timestamp: float                # Unix时间戳
    step_index: int                 # 当前决策步索引
    
    # ========== 输入状态 ==========
    state_vector: List[float]       # 原始状态向量
    pupil_diameter: float           # 当前瞳孔直径 (便于直接查看)
    pupil_relative_change: float    # 瞳孔相对变化
    
    # ========== 核心输出 - 选定动作 ==========
    action: int                     # 选定的动作 (0-2)
    action_name: str                # 动作名称 (NONE/SHORT_5S/LONG_20S)
    
    # ========== Q值分析 ==========
    q_values: List[float]           # 所有动作的Q值 [Q(s,a0), Q(s,a1), Q(s,a2)]
    q_value_selected: float         # 选定动作的Q值
    q_value_max: float              # 最大Q值
    q_value_mean: float             # 平均Q值
    q_value_std: float              # Q值标准差 (反映动作区分度)
    
    # ========== 策略概率分析 ==========
    policy_probs: List[float]       # 策略网络输出的动作概率分布
    policy_entropy: float           # 策略熵 (高=不确定, 低=确信)
    policy_confidence: float        # 选定动作的置信度 (最高概率)
    
    # ========== 值函数分析 ==========
    v_value: float                  # 状态值估计 V(s)
    advantage: float                # 选定动作的优势函数 A(s,a) = Q(s,a) - V(s)
    advantages_all: List[float]     # 所有动作的优势函数
    
    # ========== 探索信息 ==========
    epsilon: float                  # 当前探索率
    is_random_action: bool          # 是否为随机探索动作
    
    # ========== 反馈执行信息 ==========
    feedback_executed: bool         # 反馈是否被执行
    feedback_intensity: float       # 反馈强度 (如果适用)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)
    
    def to_log_string(self) -> str:
        """转换为日志字符串"""
        return (
            f"[Step {self.step_index}] "
            f"Pupil={self.pupil_diameter:.3f}mm | "
            f"Action={self.action_name} | "
            f"Q={self.q_value_selected:.3f} | "
            f"V={self.v_value:.3f} | "
            f"A={self.advantage:.3f} | "
            f"Conf={self.policy_confidence:.2%} | "
            f"Entropy={self.policy_entropy:.3f}"
        )


@dataclass
class IQLEpisodeLog:
    """
    单个Episode（试次）的完整日志
    """
    # 元数据
    subject_id: str                     # 被试ID
    session_id: str                     # 会话ID
    condition: str                      # 实验条件 (RL/feedback/silence)
    group_type: str                     # 人群类型 (如 high_arousal/low_arousal)
    start_time: str                     # 开始时间
    end_time: str                       # 结束时间
    
    # 决策记录
    decisions: List[IQLDecisionOutput]  # 所有决策输出
    
    # Episode统计
    total_steps: int                    # 总决策步数
    total_feedbacks: int                # 总反馈次数
    feedback_rate: float                # 反馈率 (反馈次数/总步数)
    
    # 动作分布统计
    action_distribution: Dict[str, int]  # 各动作使用次数
    
    # 性能指标
    mean_q_value: float                 # 平均Q值
    mean_policy_entropy: float          # 平均策略熵
    mean_advantage_when_feedback: float # 给反馈时的平均优势值


class IQLRealtimeLogger:
    """
    IQL实时实验日志记录器
    
    用于在实时实验中记录模型的所有决策和状态
    """
    
    def __init__(self, 
                 subject_id: str,
                 session_id: str,
                 condition: str,
                 group_type: str,
                 output_dir: str = "iql_realtime_logs"):
        """
        初始化日志记录器
        
        Args:
            subject_id: 被试ID (如 S01)
            session_id: 会话ID (如 D01)
            condition: 实验条件 (RL/feedback/silence)
            group_type: 人群类型分类
            output_dir: 输出目录
        """
        self.subject_id = subject_id
        self.session_id = session_id
        self.condition = condition
        self.group_type = group_type
        
        # 创建输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化记录
        self.decisions: List[IQLDecisionOutput] = []
        self.start_time = datetime.now().isoformat()
        self.step_counter = 0
        
        # 创建文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.base_filename = f"{subject_id}_{session_id}_{condition}_{timestamp}"
        
    def log_decision(self, 
                     state: PupilState,
                     action: int,
                     q_values: np.ndarray,
                     policy_probs: np.ndarray,
                     v_value: float,
                     epsilon: float,
                     is_random: bool,
                     feedback_executed: bool = True,
                     feedback_intensity: float = 1.0) -> IQLDecisionOutput:
        """
        记录单次决策
        
        Args:
            state: 当前瞳孔状态
            action: 选定的动作
            q_values: Q值向量
            policy_probs: 策略概率分布
            v_value: 状态值估计
            epsilon: 当前探索率
            is_random: 是否为随机动作
            feedback_executed: 反馈是否被执行
            feedback_intensity: 反馈强度
            
        Returns:
            IQLDecisionOutput: 完整的决策输出记录
        """
        # 计算衍生指标
        q_values = np.array(q_values)
        policy_probs = np.array(policy_probs)
        
        advantages_all = q_values - v_value
        policy_entropy = -np.sum(policy_probs * np.log(policy_probs + 1e-8))
        
        # 创建输出记录
        output = IQLDecisionOutput(
            timestamp=datetime.now().timestamp(),
            step_index=self.step_counter,
            
            # 状态
            state_vector=state.to_vector().tolist(),
            pupil_diameter=state.pupil_diameter,
            pupil_relative_change=state.pupil_relative_change,
            
            # 动作
            action=action,
            action_name=FeedbackAction(action).name,
            
            # Q值
            q_values=q_values.tolist(),
            q_value_selected=float(q_values[action]),
            q_value_max=float(q_values.max()),
            q_value_mean=float(q_values.mean()),
            q_value_std=float(q_values.std()),
            
            # 策略
            policy_probs=policy_probs.tolist(),
            policy_entropy=float(policy_entropy),
            policy_confidence=float(policy_probs.max()),
            
            # 值函数
            v_value=float(v_value),
            advantage=float(advantages_all[action]),
            advantages_all=advantages_all.tolist(),
            
            # 探索
            epsilon=float(epsilon),
            is_random_action=is_random,
            
            # 反馈
            feedback_executed=feedback_executed,
            feedback_intensity=feedback_intensity
        )
        
        self.decisions.append(output)
        self.step_counter += 1
        
        return output
    
    def save_logs(self):
        """保存所有日志到文件"""
        if not self.decisions:
            print("No decisions to save.")
            return
        
        # 计算统计信息
        action_counts = {}
        for d in self.decisions:
            action_counts[d.action_name] = action_counts.get(d.action_name, 0) + 1
        
        total_feedbacks = sum(1 for d in self.decisions if d.action != 0)
        
        feedback_advantages = [
            d.advantage for d in self.decisions if d.action != 0
        ]
        
        episode_log = IQLEpisodeLog(
            subject_id=self.subject_id,
            session_id=self.session_id,
            condition=self.condition,
            group_type=self.group_type,
            start_time=self.start_time,
            end_time=datetime.now().isoformat(),
            decisions=self.decisions,
            total_steps=len(self.decisions),
            total_feedbacks=total_feedbacks,
            feedback_rate=total_feedbacks / len(self.decisions) if self.decisions else 0,
            action_distribution=action_counts,
            mean_q_value=np.mean([d.q_value_selected for d in self.decisions]),
            mean_policy_entropy=np.mean([d.policy_entropy for d in self.decisions]),
            mean_advantage_when_feedback=np.mean(feedback_advantages) if feedback_advantages else 0
        )
        
        # 保存JSON格式（完整详细日志）
        json_path = os.path.join(self.output_dir, f"{self.base_filename}_full.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            # 转换decisions为可序列化格式
            log_dict = {
                'subject_id': episode_log.subject_id,
                'session_id': episode_log.session_id,
                'condition': episode_log.condition,
                'group_type': episode_log.group_type,
                'start_time': episode_log.start_time,
                'end_time': episode_log.end_time,
                'total_steps': episode_log.total_steps,
                'total_feedbacks': episode_log.total_feedbacks,
                'feedback_rate': episode_log.feedback_rate,
                'action_distribution': episode_log.action_distribution,
                'mean_q_value': episode_log.mean_q_value,
                'mean_policy_entropy': episode_log.mean_policy_entropy,
                'mean_advantage_when_feedback': episode_log.mean_advantage_when_feedback,
                'decisions': [d.to_dict() for d in episode_log.decisions]
            }
            json.dump(log_dict, f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式（便于快速分析）
        csv_path = os.path.join(self.output_dir, f"{self.base_filename}_decisions.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if self.decisions:
                fieldnames = [
                    'timestamp', 'step_index', 
                    'pupil_diameter', 'pupil_relative_change',
                    'action', 'action_name',
                    'q_value_selected', 'q_value_max', 'q_value_std',
                    'policy_confidence', 'policy_entropy',
                    'v_value', 'advantage',
                    'epsilon', 'is_random_action',
                    'feedback_executed'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for d in self.decisions:
                    writer.writerow({
                        'timestamp': d.timestamp,
                        'step_index': d.step_index,
                        'pupil_diameter': d.pupil_diameter,
                        'pupil_relative_change': d.pupil_relative_change,
                        'action': d.action,
                        'action_name': d.action_name,
                        'q_value_selected': d.q_value_selected,
                        'q_value_max': d.q_value_max,
                        'q_value_std': d.q_value_std,
                        'policy_confidence': d.policy_confidence,
                        'policy_entropy': d.policy_entropy,
                        'v_value': d.v_value,
                        'advantage': d.advantage,
                        'epsilon': d.epsilon,
                        'is_random_action': d.is_random_action,
                        'feedback_executed': d.feedback_executed
                    })
        
        # 保存摘要统计
        summary_path = os.path.join(self.output_dir, f"{self.base_filename}_summary.json")
        summary = {
            'subject_id': episode_log.subject_id,
            'session_id': episode_log.session_id,
            'condition': episode_log.condition,
            'group_type': episode_log.group_type,
            'total_steps': episode_log.total_steps,
            'total_feedbacks': episode_log.total_feedbacks,
            'feedback_rate': f"{episode_log.feedback_rate:.2%}",
            'action_distribution': episode_log.action_distribution,
            'performance_metrics': {
                'mean_q_value': episode_log.mean_q_value,
                'mean_policy_entropy': episode_log.mean_policy_entropy,
                'mean_advantage_when_feedback': episode_log.mean_advantage_when_feedback
            }
        }
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 日志已保存:")
        print(f"  - 完整日志: {json_path}")
        print(f"  - 决策CSV: {csv_path}")
        print(f"  - 摘要统计: {summary_path}")
        
        return json_path, csv_path, summary_path


class IQLRealtimeInference:
    """
    IQL实时推理接口
    
    封装模型推理逻辑，输出完整的决策信息
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        加载IQL模型（使用与训练一致的网络结构）

        Args:
            model_path: 模型文件路径（iql_v2/train_neural_feedback.py 产生）
            device: 计算设备
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"未找到IQL模型: {model_path}\n请提供训练得到的 .pth 模型。"
            )
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=self.device)

        # 使用checkpoint内保存的配置还原网络结构（Dueling/策略网络等）
        from neural_feedback_iql import ImplicitQLearning, IQLConfig
        cfg = checkpoint.get('config', None)
        if not isinstance(cfg, IQLConfig):
            raise ValueError("模型checkpoint缺少IQLConfig配置，无法还原网络结构。")
        cfg.device = str(self.device)

        self.agent = ImplicitQLearning(cfg)
        self.agent.load_model(model_path)
        self.state_dim = cfg.state_dim
        self.action_dim = cfg.action_dim
        self.epsilon = 0.0  # 推理时不探索（论文式4-25贪心策略）
        if self.state_dim != 10:
            raise ValueError(
                f"统一口径为10维瞳孔状态，当前checkpoint为 {self.state_dim} 维，请使用10维模型。"
            )
        if self.action_dim != 3:
            raise ValueError(
                f"论文动作空间应为3维(NONE/SHORT_5S/LONG_20S)，当前checkpoint为 {self.action_dim} 维。"
            )

    def _build_model_state(self, state: PupilState) -> np.ndarray:
        return state.to_vector()

    def get_action_with_info(self, state: PupilState) -> Tuple[int, Dict]:
        """
        对状态做前向推理，返回动作与完整决策信息

        Args:
            state: 瞳孔状态

        Returns:
            action: 选定动作（贪心）
            info: 包含Q值、策略概率、值函数等信息的字典
        """
        state_vec = self._build_model_state(state)
        q_values = self.agent.get_q_values(state_vec)
        policy_probs = self.agent.get_policy_probs(state_vec)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v_value = float(self.agent.v_network(state_tensor).cpu().numpy().flatten()[0])

        action = int(np.argmax(q_values))
        info = {
            'q_values': q_values,
            'v_value': v_value,
            'policy_probs': policy_probs,
            'epsilon': self.epsilon,
            'is_random': False,
            'model_state_dim': self.state_dim,
        }
        return action, info


# ============================================================
# 策略分析指标说明
# ============================================================
"""
### 关键输出指标解读 ###

1. **Q值 (Q-values)**
   - 含义：每个动作在当前状态下的预期长期回报
   - 分析：Q值差异大 → 模型对动作选择更确信
   - 用途：比较不同人群的Q值模式差异

2. **策略概率 (Policy Probabilities)**
   - 含义：模型选择各动作的概率分布
   - 分析：概率集中 → 策略确定性高
   - 用途：分析不同状态下的决策偏好

3. **策略熵 (Policy Entropy)**
   - 含义：决策不确定性的度量
   - 低熵：模型很确定应该选什么动作
   - 高熵：模型对动作选择不确定
   - 用途：识别"困难"决策点

4. **状态值 V(s)**
   - 含义：当前状态的整体价值估计
   - 分析：V值高的状态是"好"状态
   - 用途：评估驾驶状态的安全性

5. **优势函数 A(s,a)**
   - 含义：选择某动作相比平均的额外收益
   - A > 0：该动作比平均更好
   - A < 0：该动作比平均更差
   - 用途：理解为什么选择某个反馈

### 策略分析建议 ###

1. **人群差异分析**
   - 比较不同人群（高/低唤醒组）的动作分布
   - 分析各组在相似状态下的Q值差异
   - 比较策略熵的分布（哪组决策更确定）

2. **时序分析**
   - 追踪Q值随时间的变化
   - 分析策略熵在不同实验阶段的变化
   - 识别模型策略转变的时间点

3. **状态-动作关联分析**
   - pupil_diameter vs action 的关系
   - pupil_change_rate vs advantage 的关系
   - 高熵决策发生时的瞳孔状态特征

4. **反馈效果分析**
   - 反馈后瞳孔的变化
   - 反馈后Q值/V值的变化
   - 连续反馈vs间隔反馈的效果差异
"""


def _read_pupil_csv(pupil_csv: str):
    """读取真实瞳孔CSV（需含列 timestamp, PupilDiameter）。"""
    import pandas as pd
    if not os.path.exists(pupil_csv):
        raise FileNotFoundError(f"瞳孔数据文件不存在: {pupil_csv}")
    df = pd.read_csv(pupil_csv)
    if 'timestamp' not in df.columns or 'PupilDiameter' not in df.columns:
        raise ValueError(
            f"{pupil_csv} 缺少必要列 timestamp / PupilDiameter。现有列: {list(df.columns)}"
        )
    df['pupil_mm'] = df['PupilDiameter'] * 1000.0 if df['PupilDiameter'].abs().median() < 0.1 \
        else df['PupilDiameter']
    return df.sort_values('timestamp').reset_index(drop=True)


def _build_state_at(df, t, baseline, time_since_last_feedback, recent_feedback_count):
    """从真实瞳孔时序在时间点 t 构造10维 PupilState（论文式4-9）。"""
    idx = (df['timestamp'] - t).abs().idxmin()
    current = float(df.loc[idx, 'pupil_mm'])
    win1 = df[(df['timestamp'] >= t - 1.0) & (df['timestamp'] <= t)]['pupil_mm']
    win5 = df[(df['timestamp'] >= t - 5.0) & (df['timestamp'] <= t)]['pupil_mm']
    if len(win5) == 0:
        return None
    mean_1s = float(win1.mean()) if len(win1) > 0 else current
    mean_5s = float(win5.mean())
    std_win = float(win5.std()) if len(win5) > 1 else 0.0
    change_rate = float(np.mean(np.diff(win1.values))) if len(win1) > 2 else 0.0
    rel_change = (current - baseline) / baseline * 100.0 if baseline > 0 else 0.0
    if len(win5) > 2:
        x = win5.index.values.astype(float)
        slope = float(np.polyfit(x - x.mean(), win5.values, 1)[0])
        trend = float(np.clip(slope, -1.0, 1.0))
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


def run_realtime_session(model_path: str, pupil_csv: str, subject_id: str = "S01",
                         session_id: str = "D01", condition: str = "RL",
                         group_type: str = "unknown", decision_interval_s: float = 1.0,
                         min_feedback_interval_s: float = 5.0, device: str = "cpu"):
    """对真实瞳孔时序按固定决策间隔运行已训练IQL模型并记录在线决策日志（论文4.4.2在线部署）。"""
    inferencer = IQLRealtimeInference(model_path, device=device)
    logger = IQLRealtimeLogger(subject_id=subject_id, session_id=session_id,
                               condition=condition, group_type=group_type)
    df = _read_pupil_csv(pupil_csv)

    t_start, t_end = float(df['timestamp'].min()), float(df['timestamp'].max())
    baseline = float(df[df['timestamp'] < (t_start + 5.0)]['pupil_mm'].mean())
    if not np.isfinite(baseline) or baseline <= 0:
        baseline = float(df['pupil_mm'].median())

    last_feedback_t = -np.inf
    recent_feedback_times: List[float] = []
    t = t_start + 5.0
    while t <= t_end:
        recent_feedback_times = [ft for ft in recent_feedback_times if t - ft <= 10.0]
        time_since = t - last_feedback_t if np.isfinite(last_feedback_t) else 999.0
        state = _build_state_at(df, t, baseline, time_since, len(recent_feedback_times))
        if state is None:
            t += decision_interval_s
            continue
        action, info = inferencer.get_action_with_info(state)
        if action != 0 and time_since < min_feedback_interval_s:
            action = 0  # 系统层频率限制：最小反馈间隔约束
        if action != 0:
            last_feedback_t = t
            recent_feedback_times.append(t)
        output = logger.log_decision(
            state=state, action=action,
            q_values=np.asarray(info['q_values']),
            policy_probs=np.asarray(info['policy_probs']),
            v_value=float(info['v_value']), epsilon=0.0, is_random=False,
        )
        print(output.to_log_string())
        t += decision_interval_s

    logger.save_logs()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IQL在线推理与实时决策日志（真实模型+真实瞳孔数据）")
    parser.add_argument("--model", required=True, help="已训练IQL模型权重 .pth")
    parser.add_argument("--pupil-csv", required=True, help="真实瞳孔CSV（含 timestamp,PupilDiameter）")
    parser.add_argument("--subject-id", default="S01")
    parser.add_argument("--session-id", default="D01")
    parser.add_argument("--condition", default="RL")
    parser.add_argument("--group-type", default="unknown")
    parser.add_argument("--decision-interval", type=float, default=1.0)
    parser.add_argument("--min-feedback-interval", type=float, default=5.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_realtime_session(
        model_path=args.model, pupil_csv=args.pupil_csv,
        subject_id=args.subject_id, session_id=args.session_id,
        condition=args.condition, group_type=args.group_type,
        decision_interval_s=args.decision_interval,
        min_feedback_interval_s=args.min_feedback_interval, device=args.device,
    )
