import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass
from enum import Enum

class FeedbackType(Enum):
    """反馈动作枚举（论文5.4.1 表5.7：按反馈时长离散）。"""
    NONE = 0
    SHORT_5S = 1
    LONG_20S = 2

@dataclass
class EEGMetrics:
    """脑电指标数据结构"""
    alpha_power: float  
    beta_power: float   
    theta_power: float  
    delta_power: float  
    gamma_power: float  
    alpha_beta_ratio: float  
    engagement_index: float  
    drowsiness_index: float  
    mental_workload: float  
    attention_level: float  
    eye_blink_rate: float   
    artifact_level: float   

class DrivingContext(Enum):
    """驾驶场景上下文"""
    HIGHWAY_CRUISE = 0
    CITY_TRAFFIC = 1
    NIGHT_DRIVING = 2
    HIGHWAY_CONGESTION = 3
    WEATHER_CONDITION = 4
    EMERGENCY_SITUATION = 5

class NeuralFeedbackEnvironment(gym.Env):
    """神经反馈优化环境"""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # 默认配置
        self.config = config or {
            'max_timesteps': 1000,
            'sampling_rate': 250,  # 250Hz采样率
            'arousal_target': 0.5,  # 目标arousal水平
            'feedback_cost': 0.01,  # 反馈成本
            'arousal_penalty_weight': 10.0,
            'safety_weight': 5.0,
            'comfort_weight': 2.0
        }
        
        # 状态空间定义 - 统一为10维瞳孔状态向量（论文式4-9）
        # 与在线部署 PupilState 完全同口径，训练/推理共用同一状态表示
        self.state_dim = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # 动作空间 - 是否发送反馈以及反馈时长
        self.action_space = spaces.Discrete(len(FeedbackType))
        
        # 内部状态
        self.current_timestep = 0
        self.arousal_history = []
        self.feedback_history = []
        self.eeg_buffer = []
        self.driving_context = DrivingContext.HIGHWAY_CRUISE
        
        # 模拟的生理状态（latent，仅用于奖励计算，不直接进入观测）
        self.base_arousal = 0.5
        self.arousal_trend = 0.0
        self.fatigue_accumulation = 0.0
        self.stress_level = 0.3
        self.current_arousal = self.config['arousal_target']

        # 瞳孔观测相关（10维瞳孔状态由此派生）
        self.pupil_baseline = 3.5
        self.pupil_history: List[float] = []
        self.steps_since_feedback = 999.0
        
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        self.current_timestep = 0
        self.arousal_history = [self.config['arousal_target']] * 5
        self.feedback_history = [0] * 5
        self.eeg_buffer = []
        
        # 随机初始化生理状态
        self.base_arousal = np.random.uniform(0.3, 0.7)
        self.arousal_trend = np.random.uniform(-0.1, 0.1)
        self.fatigue_accumulation = np.random.uniform(0.0, 0.2)
        self.stress_level = np.random.uniform(0.2, 0.6)
        
        # 随机驾驶场景
        self.driving_context = np.random.choice(list(DrivingContext))

        # 初始化瞳孔观测：以目标arousal预热5步瞳孔历史
        self.current_arousal = self.config['arousal_target']
        self.pupil_baseline = float(np.random.uniform(3.0, 4.5))
        self.pupil_history = [self._simulate_pupil(self.current_arousal) for _ in range(5)]
        self.steps_since_feedback = 999.0
        
        return self._get_observation(), {}
    
    def step(self, action: int):
        """执行一步"""
        feedback_type = FeedbackType(action)
        
        # 更新时间步
        self.current_timestep += 1
        
        # 模拟脑电数据变化
        eeg_metrics = self._simulate_eeg_metrics()
        
        # 计算当前arousal水平
        current_arousal = self._calculate_arousal(eeg_metrics, feedback_type)
        self.current_arousal = current_arousal
        
        # 更新历史记录
        self.arousal_history.append(current_arousal)
        self.arousal_history = self.arousal_history[-5:]  # 保持最近5个时间步
        
        self.feedback_history.append(action)
        self.feedback_history = self.feedback_history[-5:]

        # 由latent arousal派生瞳孔直径并更新瞳孔历史（观测来源）
        self.pupil_history.append(self._simulate_pupil(current_arousal))
        self.pupil_history = self.pupil_history[-10:]
        if feedback_type != FeedbackType.NONE:
            self.steps_since_feedback = 0.0
        else:
            self.steps_since_feedback += 1.0
        
        # 更新驾驶场景
        self._update_driving_context()
        
        # 计算奖励
        reward = self._calculate_reward(current_arousal, feedback_type, eeg_metrics)
        
        # 检查终止条件
        terminated = self.current_timestep >= self.config['max_timesteps']
        truncated = False
        
        # 更新生理状态趋势
        self._update_physiological_trends()
        
        return self._get_observation(), reward, terminated, truncated, {
            'arousal': current_arousal,
            'eeg_metrics': eeg_metrics,
            'driving_context': self.driving_context,
            'feedback_type': feedback_type
        }
    
    def _simulate_eeg_metrics(self) -> EEGMetrics:
        """模拟实时脑电指标"""
        # 基于当前生理状态和驾驶场景生成逼真的EEG数据
        
        # 基础噪声
        noise_scale = 0.1
        
        # 根据疲劳程度调整各频段功率
        fatigue_effect = self.fatigue_accumulation
        stress_effect = self.stress_level
        
        # Alpha波 (8-12Hz) - 与放松状态相关
        alpha_power = (0.6 - fatigue_effect * 0.3 - stress_effect * 0.2) + \
                     np.random.normal(0, noise_scale)
        
        # Beta波 (13-30Hz) - 与警觉状态相关
        beta_power = (0.4 + stress_effect * 0.3 + fatigue_effect * 0.1) + \
                    np.random.normal(0, noise_scale)
        
        # Theta波 (4-7Hz) - 与困倦相关
        theta_power = (0.3 + fatigue_effect * 0.4) + \
                     np.random.normal(0, noise_scale)
        
        # Delta波 (0.5-3Hz) - 与深度困倦相关
        delta_power = (0.2 + fatigue_effect * 0.5) + \
                     np.random.normal(0, noise_scale)
        
        # Gamma波 (30-100Hz) - 与认知负荷相关
        gamma_power = (0.3 + stress_effect * 0.2) + \
                     np.random.normal(0, noise_scale)
        
        # 计算衍生指标
        alpha_beta_ratio = alpha_power / (beta_power + 1e-6)
        engagement_index = beta_power / (alpha_power + theta_power + 1e-6)
        drowsiness_index = (theta_power + delta_power) / (alpha_power + beta_power + 1e-6)
        mental_workload = beta_power + gamma_power - alpha_power
        attention_level = beta_power / (theta_power + 1e-6)
        
        # 眨眼频率 - 与疲劳相关
        eye_blink_rate = 15 + fatigue_effect * 10 + np.random.normal(0, 2)
        
        # 伪影水平
        artifact_level = np.random.uniform(0.0, 0.2)
        
        return EEGMetrics(
            alpha_power=alpha_power,
            beta_power=beta_power,
            theta_power=theta_power,
            delta_power=delta_power,
            gamma_power=gamma_power,
            alpha_beta_ratio=alpha_beta_ratio,
            engagement_index=engagement_index,
            drowsiness_index=drowsiness_index,
            mental_workload=mental_workload,
            attention_level=attention_level,
            eye_blink_rate=eye_blink_rate,
            artifact_level=artifact_level
        )
    
    def _calculate_arousal(self, eeg_metrics: EEGMetrics, feedback_type: FeedbackType) -> float:
        """基于EEG指标和反馈计算arousal水平"""
        
        # 基础arousal来自EEG指标
        eeg_arousal = (
            eeg_metrics.beta_power * 0.3 +
            eeg_metrics.gamma_power * 0.2 +
            eeg_metrics.engagement_index * 0.2 +
            eeg_metrics.attention_level * 0.15 +
            (1 - eeg_metrics.drowsiness_index) * 0.15
        )
        
        # 反馈对arousal的即时影响（按反馈持续时间建模）
        feedback_effect = 0.0
        if feedback_type != FeedbackType.NONE:
            # 长时反馈激活更强，但代价也更高（在奖励函数中体现）
            feedback_effects = {
                FeedbackType.SHORT_5S: 0.15,
                FeedbackType.LONG_20S: 0.25,
            }
            feedback_effect = feedback_effects.get(feedback_type, 0.0)
            
            # 反馈效果随时间衰减
            feedback_effect *= np.exp(-0.1 * len([f for f in self.feedback_history[-3:] if f != 0]))
        
        # 驾驶场景对arousal的影响
        context_effects = {
            DrivingContext.HIGHWAY_CRUISE: -0.1,
            DrivingContext.CITY_TRAFFIC: 0.1,
            DrivingContext.NIGHT_DRIVING: -0.05,
            DrivingContext.HIGHWAY_CONGESTION: 0.15,
            DrivingContext.WEATHER_CONDITION: 0.2,
            DrivingContext.EMERGENCY_SITUATION: 0.4
        }
        context_effect = context_effects.get(self.driving_context, 0.0)
        
        # 疲劳累积降低arousal
        fatigue_effect = -self.fatigue_accumulation * 0.5
        
        # 最终arousal计算
        total_arousal = self.base_arousal + eeg_arousal + feedback_effect + \
                       context_effect + fatigue_effect + self.arousal_trend
        
        return np.clip(total_arousal, 0.0, 1.0)
    
    def _calculate_reward(self, current_arousal: float, feedback_type: FeedbackType, 
                         eeg_metrics: EEGMetrics) -> float:
        """计算奖励函数"""
        
        target_arousal = self.config['arousal_target']
        
        # 主要目标：保持arousal在目标水平附近
        arousal_error = abs(current_arousal - target_arousal)
        arousal_reward = -self.config['arousal_penalty_weight'] * arousal_error**2
        
        # 安全奖励：避免过低arousal（危险驾驶）
        safety_threshold = 0.2
        if current_arousal < safety_threshold:
            safety_penalty = -self.config['safety_weight'] * (safety_threshold - current_arousal)**2
        else:
            safety_penalty = 0.0
        
        # 舒适度奖励：避免过度反馈（时长越长舒适度代价越高）
        comfort_penalty = 0.0
        if feedback_type != FeedbackType.NONE:
            duration_cost_scale = {
                FeedbackType.SHORT_5S: 1.0,   # 5s
                FeedbackType.LONG_20S: 4.0,   # 20s = 4 * 5s
            }.get(feedback_type, 1.0)
            comfort_penalty = -self.config['comfort_weight'] * self.config['feedback_cost'] * duration_cost_scale
            
            # 频繁反馈额外惩罚
            recent_feedbacks = sum([1 for f in self.feedback_history[-3:] if f != 0])
            if recent_feedbacks >= 2:
                comfort_penalty *= 2.0
        
        # 驾驶表现奖励
        performance_reward = 0.0
        if self.driving_context == DrivingContext.EMERGENCY_SITUATION:
            # 紧急情况下需要高arousal
            if current_arousal > 0.7:
                performance_reward = 2.0
        elif current_arousal > 0.3 and current_arousal < 0.6:
            # 正常驾驶情况下的最佳arousal范围
            performance_reward = 1.0
        
        # EEG质量奖励：鼓励清洁的信号
        signal_quality_reward = -eeg_metrics.artifact_level * 0.5
        
        total_reward = arousal_reward + safety_penalty + comfort_penalty + \
                      performance_reward + signal_quality_reward
        
        return total_reward
    
    def _update_driving_context(self):
        """更新驾驶场景上下文"""
        # 场景转换概率
        transition_prob = 0.02  # 每步2%的概率切换场景
        
        if np.random.random() < transition_prob:
            # 加权随机选择新场景
            contexts = list(DrivingContext)
            weights = [0.3, 0.25, 0.15, 0.15, 0.1, 0.05]  # 高速巡航最常见
            self.driving_context = np.random.choice(contexts, p=weights)
    
    def _update_physiological_trends(self):
        """更新生理状态趋势"""
        # 疲劳累积
        self.fatigue_accumulation += 0.001  # 每步增加一点疲劳
        self.fatigue_accumulation = np.clip(self.fatigue_accumulation, 0.0, 1.0)
        
        # Arousal自然趋势
        self.arousal_trend += np.random.normal(0, 0.01)
        self.arousal_trend = np.clip(self.arousal_trend, -0.3, 0.3)
        
        # 压力水平变化
        stress_change = np.random.normal(0, 0.01)
        if self.driving_context in [DrivingContext.EMERGENCY_SITUATION, 
                                   DrivingContext.WEATHER_CONDITION]:
            stress_change += 0.02  # 困难场景增加压力
        
        self.stress_level += stress_change
        self.stress_level = np.clip(self.stress_level, 0.0, 1.0)
    
    def _simulate_pupil(self, arousal: float) -> float:
        """由latent生理状态派生瞳孔直径（mm）。

        瞳孔随唤醒度/认知负荷上升而扩张，随疲劳轻微收缩。该映射使智能体在
        训练阶段也仅观测瞳孔信号，与在线部署（仅瞳孔输入）保持同一口径。
        """
        dilation = (0.18 * (arousal - 0.5) * 2.0
                    + 0.08 * self.stress_level
                    - 0.06 * self.fatigue_accumulation)
        diameter = self.pupil_baseline * (1.0 + dilation) + np.random.normal(0, 0.03)
        return float(np.clip(diameter, 1.5, 8.0))

    def _get_observation(self) -> np.ndarray:
        """获取当前观测状态：10维瞳孔状态向量（论文式4-9）。

        分量顺序与在线部署 PupilState.to_vector 完全一致：
        [d_p, d_base, v_p, Δd_rel, σ_p, d̄_1s, d̄_5s, τ_p, Δt_fb, n_fb/10]
        """
        hist = self.pupil_history if self.pupil_history else [self.pupil_baseline]
        current = float(hist[-1])
        base = float(self.pupil_baseline)
        win5 = hist[-5:]

        mean_5s = float(np.mean(win5))
        mean_1s = current
        std_p = float(np.std(win5)) if len(win5) > 1 else 0.0
        change_rate = float(hist[-1] - hist[-2]) if len(hist) >= 2 else 0.0
        rel_change = (current - base) / base * 100.0 if base > 0 else 0.0
        if len(win5) > 2:
            x = np.arange(len(win5), dtype=float)
            slope = float(np.polyfit(x, win5, 1)[0])
            trend = float(np.clip(slope, -1.0, 1.0))
        else:
            trend = 0.0
        dt_fb = float(self.steps_since_feedback)
        n_fb = float(sum(1 for f in self.feedback_history if f != 0))

        state_vector = [
            current,        # 当前瞳孔直径 d_p
            base,           # 基线瞳孔直径 d_base
            change_rate,    # 瞳孔变化速率 v_p
            rel_change,     # 相对基线变化 Δd_rel (%)
            std_p,          # 窗口标准差 σ_p
            mean_1s,        # 近1s均值 d̄_1s
            mean_5s,        # 近5s均值 d̄_5s
            trend,          # 趋势 τ_p
            dt_fb,          # 距上次反馈时间 Δt_fb
            n_fb / 10.0,    # 近窗反馈次数 n_fb/10
        ]

        return np.array(state_vector, dtype=np.float32)
    
    def render(self, mode='human'):
        """渲染环境状态"""
        if mode == 'human':
            current_arousal = self.arousal_history[-1] if self.arousal_history else 0.5
            print(f"时间步: {self.current_timestep}")
            print(f"当前Arousal: {current_arousal:.3f}")
            print(f"目标Arousal: {self.config['arousal_target']:.3f}")
            print(f"疲劳程度: {self.fatigue_accumulation:.3f}")
            print(f"压力水平: {self.stress_level:.3f}")
            print(f"驾驶场景: {self.driving_context.name}")
            print(f"最近反馈: {[FeedbackType(f).name for f in self.feedback_history[-3:]]}")
            print("-" * 50) 