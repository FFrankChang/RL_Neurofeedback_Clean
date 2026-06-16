#!/usr/bin/env python3

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import random
from dataclasses import dataclass
from enum import Enum

# 导入原始环境和权重策略
from neural_feedback_env import NeuralFeedbackEnvironment, FeedbackType, EEGMetrics, DrivingContext
from step_weight_strategy import StepWeightStrategy, WeightComponent, WeightScheduleType, PerformanceMetrics, create_default_weight_strategy

class AdaptiveWeightNeuralFeedbackEnvironment(NeuralFeedbackEnvironment):
    """集成步级权重策略的神经反馈环境"""
    
    def __init__(self, config: Optional[Dict] = None, weight_strategy: Optional[StepWeightStrategy] = None):
        # 初始化基础环境
        super().__init__(config)
        
        # 初始化权重策略
        self.weight_strategy = weight_strategy or create_default_weight_strategy(
            total_episodes=self.config.get('total_episodes', 1000)
        )
        
        # 训练统计
        self.episode_count = 0
        self.episode_rewards = []
        self.episode_arousal_errors = []
        self.episode_losses = []
        self.step_count_in_episode = 0
        
        # 性能窗口（用于计算滑动平均）
        self.performance_window = 50
        
        print("初始化自适应权重神经反馈环境")
        print(f"权重策略包含 {len(self.weight_strategy.weight_components)} 个组件")
    
    def reset(self, seed=None, options=None):
        """重置环境并更新权重策略"""
        state, info = super().reset(seed, options)
        
        # 重置episode统计
        self.step_count_in_episode = 0
        
        # 如果不是第一个episode，更新权重策略
        if self.episode_count > 0:
            self._update_weight_strategy()
        
        return state, info
    
    def step(self, action: int):
        """执行一步并记录性能指标"""
        # 获取当前权重
        current_weights = self.weight_strategy.get_current_weights()
        
        # 执行环境步骤
        state, reward, terminated, truncated, info = super().step(action)
        
        # 使用自适应权重重新计算奖励
        adaptive_reward = self._calculate_adaptive_reward(
            info.get('arousal', 0.5), 
            FeedbackType(action), 
            info.get('eeg_metrics'), 
            current_weights
        )
        
        # 更新统计
        self.step_count_in_episode += 1
        self.episode_rewards.append(adaptive_reward)
        
        # 记录arousal误差
        arousal_error = abs(info.get('arousal', 0.5) - self.config['arousal_target'])
        self.episode_arousal_errors.append(arousal_error)
        
        # 更新info
        info['adaptive_reward'] = adaptive_reward
        info['current_weights'] = current_weights
        info['weight_summary'] = self.weight_strategy.get_weight_summary()
        
        return state, adaptive_reward, terminated, truncated, info
    
    def _calculate_adaptive_reward(self, current_arousal: float, feedback_type: FeedbackType, 
                                 eeg_metrics: EEGMetrics, weights: Dict[str, float]) -> float:
        """使用自适应权重计算奖励"""
        
        target_arousal = self.config['arousal_target']
        
        # 基础arousal奖励（使用动态权重）
        arousal_error = abs(current_arousal - target_arousal)
        arousal_weight = weights.get('arousal_penalty_weight', self.config.get('arousal_penalty_weight', 10.0))
        arousal_reward = -arousal_weight * arousal_error**2
        
        # 安全奖励（使用动态权重）
        safety_weight = weights.get('safety_weight', self.config.get('safety_weight', 5.0))
        safety_threshold = 0.2
        if current_arousal < safety_threshold:
            safety_penalty = -safety_weight * (safety_threshold - current_arousal)**2
        else:
            safety_penalty = 0.0
        
        # 舒适度奖励（使用动态权重）
        comfort_weight = weights.get('comfort_weight', self.config.get('comfort_weight', 2.0))
        comfort_penalty = 0.0
        if feedback_type != FeedbackType.NONE:
            comfort_penalty = -comfort_weight * self.config['feedback_cost']
            
            # 频繁反馈额外惩罚
            recent_feedbacks = sum([1 for f in self.feedback_history[-3:] if f != 0])
            if recent_feedbacks >= 2:
                comfort_penalty *= 2.0
        
        # 探索奖励（新增，鼓励适度探索）
        exploration_weight = weights.get('exploration_weight', 0.0)
        exploration_reward = 0.0
        if exploration_weight > 0:
            # 基于动作多样性给予探索奖励
            recent_actions = self.feedback_history[-5:]
            action_diversity = len(set(recent_actions)) / len(recent_actions) if recent_actions else 0
            exploration_reward = exploration_weight * action_diversity
        
        # 驾驶表现奖励（保持原逻辑）
        performance_reward = 0.0
        if self.driving_context == DrivingContext.EMERGENCY_SITUATION:
            if current_arousal > 0.7:
                performance_reward = 2.0
        elif current_arousal > 0.3 and current_arousal < 0.6:
            performance_reward = 1.0
        
        # EEG信号质量奖励（使用动态权重）
        signal_quality_weight = weights.get('signal_quality_weight', 0.5)
        signal_quality_reward = 0.0
        if eeg_metrics:
            signal_quality_reward = -signal_quality_weight * eeg_metrics.artifact_level
        
        # 总奖励
        total_reward = (arousal_reward + safety_penalty + comfort_penalty + 
                       exploration_reward + performance_reward + signal_quality_reward)
        
        return total_reward
    
    def _update_weight_strategy(self):
        """更新权重策略"""
        # 计算episode统计
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        avg_arousal_error = np.mean(self.episode_arousal_errors) if self.episode_arousal_errors else 0.5
        
        # 计算收敛和稳定性分数
        convergence_score = self._calculate_convergence_score()
        stability_score = self._calculate_stability_score()
        
        # 创建性能指标
        metrics = PerformanceMetrics(
            episode=self.episode_count,
            current_step=self.step_count_in_episode,
            total_steps=self.config.get('max_timesteps', 1000),
            avg_reward=avg_reward,
            avg_arousal_error=avg_arousal_error,
            training_loss=0.0,  # 需要从外部传入
            convergence_score=convergence_score,
            stability_score=stability_score,
            epsilon=1.0  # 需要从外部传入
        )
        
        # 更新权重
        updated_weights = self.weight_strategy.update_weights(metrics)
        
        # 清空episode统计
        self.episode_rewards = []
        self.episode_arousal_errors = []
        self.episode_count += 1
        
        # 打印权重更新信息
        if self.episode_count % 50 == 0:
            print(f"\nEpisode {self.episode_count} 权重更新:")
            for name, weight in updated_weights.items():
                print(f"  {name}: {weight:.3f}")
            print(f"  平均奖励: {avg_reward:.3f}")
            print(f"  平均Arousal误差: {avg_arousal_error:.3f}")
            print(f"  收敛分数: {convergence_score:.3f}")
    
    def _calculate_convergence_score(self) -> float:
        """计算收敛分数"""
        if len(self.episode_rewards) < 10:
            return 0.0
        
        # 计算最近奖励的趋势
        recent_rewards = self.episode_rewards[-10:]
        if len(set(recent_rewards)) == 1:  # 所有值相同
            return 1.0
        
        # 计算变异系数
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        if mean_reward == 0:
            return 0.0
        
        cv = std_reward / abs(mean_reward)
        convergence_score = max(0.0, 1.0 - cv)
        
        return convergence_score
    
    def _calculate_stability_score(self) -> float:
        """计算稳定性分数"""
        if len(self.episode_arousal_errors) < 10:
            return 0.0
        
        # 基于arousal误差的稳定性
        recent_errors = self.episode_arousal_errors[-10:]
        error_std = np.std(recent_errors)
        
        # 稳定性分数：误差标准差越小，稳定性越高
        stability_score = max(0.0, 1.0 - error_std)
        
        return stability_score
    
    def set_training_metrics(self, training_loss: float, epsilon: float):
        """设置外部训练指标"""
        self.current_training_loss = training_loss
        self.current_epsilon = epsilon
    
    def get_weight_evolution_data(self) -> Dict[str, Any]:
        """获取权重演化数据"""
        return {
            'weight_history': self.weight_strategy.weight_history,
            'performance_history': self.weight_strategy.performance_history,
            'current_weights': self.weight_strategy.get_current_weights()
        }
    
    def visualize_training_progress(self, save_path: Optional[str] = None):
        """可视化训练进度"""
        import matplotlib.pyplot as plt
        
        if not self.weight_strategy.weight_history:
            print("没有权重历史数据可以可视化")
            return
        
        # 提取数据
        episodes = [entry['episode'] for entry in self.weight_strategy.weight_history]
        rewards = [entry['metrics']['avg_reward'] for entry in self.weight_strategy.weight_history]
        arousal_errors = [entry['metrics']['avg_arousal_error'] for entry in self.weight_strategy.weight_history]
        convergence_scores = [entry['metrics']['convergence_score'] for entry in self.weight_strategy.weight_history]
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('自适应权重训练进度', fontsize=16, fontweight='bold')
        
        # 平均奖励
        axes[0, 0].plot(episodes, rewards, 'b-', linewidth=2)
        axes[0, 0].set_title('平均奖励趋势')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('平均奖励')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Arousal误差
        axes[0, 1].plot(episodes, arousal_errors, 'r-', linewidth=2)
        axes[0, 1].set_title('Arousal控制误差')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('平均Arousal误差')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 收敛分数
        axes[1, 0].plot(episodes, convergence_scores, 'g-', linewidth=2)
        axes[1, 0].set_title('收敛分数')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('收敛分数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 权重演化（选择主要权重）
        weight_names = ['arousal_penalty_weight', 'safety_weight', 'comfort_weight']
        for name in weight_names:
            if any(name in entry['weights'] for entry in self.weight_strategy.weight_history):
                weights = [entry['weights'].get(name, 0) for entry in self.weight_strategy.weight_history]
                axes[1, 1].plot(episodes, weights, linewidth=2, label=name)
        
        axes[1, 1].set_title('主要权重演化')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('权重值')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练进度图已保存到: {save_path}")
        
        plt.show()
    
    def save_training_data(self, filepath: str):
        """保存训练数据"""
        training_data = {
            'episode_count': self.episode_count,
            'weight_history': self.weight_strategy.weight_history,
            'performance_history': [{
                'episode': pm.episode,
                'avg_reward': pm.avg_reward,
                'avg_arousal_error': pm.avg_arousal_error,
                'convergence_score': pm.convergence_score,
                'stability_score': pm.stability_score
            } for pm in self.weight_strategy.performance_history],
            'config': self.config
        }
        
        import json
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"训练数据已保存到: {filepath}")

def create_neural_feedback_weight_strategy(total_episodes: int = 1000) -> StepWeightStrategy:
    """为神经反馈环境创建专门的权重策略"""
    
    weight_components = [
        # Arousal惩罚权重 - 初期较低，逐渐增加精度要求
        WeightComponent(
            name='arousal_penalty_weight',
            initial_weight=8.0,
            target_weight=18.0,
            schedule_type=WeightScheduleType.SIGMOID,
            schedule_params={'steepness': 6.0, 'midpoint': 0.3},
            min_weight=3.0,
            max_weight=25.0
        ),
        
        # 安全权重 - 始终保持较高水平
        WeightComponent(
            name='safety_weight',
            initial_weight=12.0,
            target_weight=8.0,
            schedule_type=WeightScheduleType.COSINE,
            min_weight=5.0,
            max_weight=15.0
        ),
        
        # 舒适度权重 - 后期增加，避免过度干预
        WeightComponent(
            name='comfort_weight',
            initial_weight=1.5,
            target_weight=4.0,
            schedule_type=WeightScheduleType.LINEAR,
            min_weight=0.5,
            max_weight=6.0
        ),
        
        # 探索权重 - 初期高，后期低
        WeightComponent(
            name='exploration_weight',
            initial_weight=3.0,
            target_weight=0.3,
            schedule_type=WeightScheduleType.EXPONENTIAL,
            schedule_params={'decay_rate': 3.0},
            min_weight=0.1,
            max_weight=5.0
        ),
        
        # 信号质量权重 - 阶梯式增加
        WeightComponent(
            name='signal_quality_weight',
            initial_weight=0.3,
            target_weight=1.5,
            schedule_type=WeightScheduleType.STEP,
            schedule_params={
                'steps': [0.25, 0.6, 0.85],
                'values': [0.3, 0.7, 1.1, 1.5]
            },
            min_weight=0.1,
            max_weight=2.0
        )
    ]
    
    return StepWeightStrategy(weight_components, total_episodes)

if __name__ == "__main__":
    # 演示使用
    print("自适应权重神经反馈环境演示")
    
    # 创建专门的权重策略
    weight_strategy = create_neural_feedback_weight_strategy(total_episodes=200)
    
    # 创建环境
    env = AdaptiveWeightNeuralFeedbackEnvironment(
        config={'max_timesteps': 100, 'total_episodes': 200},
        weight_strategy=weight_strategy
    )
    
    # 运行几个episode
    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(50):
            action = env.action_space.sample()  # 随机动作
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode}: 总奖励 = {episode_reward:.3f}")
        
        # 打印权重摘要
        if episode % 2 == 0:
            summary = info.get('weight_summary', {})
            if summary:
                print(f"  当前权重: {summary.get('current_weights', {})}")
    
    # 可视化结果
    env.visualize_training_progress() 