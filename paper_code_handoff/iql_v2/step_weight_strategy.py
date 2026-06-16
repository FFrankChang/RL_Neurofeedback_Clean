#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime

class WeightScheduleType(Enum):
    """权重调度类型"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"
    SIGMOID = "sigmoid"
    STEP = "step"

@dataclass
class WeightComponent:
    """权重组件配置"""
    name: str
    initial_weight: float
    target_weight: float
    schedule_type: WeightScheduleType
    schedule_params: Dict[str, Any] = field(default_factory=dict)
    min_weight: float = 0.0
    max_weight: float = 10.0
    adaptive_metric: Optional[str] = None  # 用于自适应调整的指标

@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    episode: int
    current_step: int
    total_steps: int
    avg_reward: float
    avg_arousal_error: float
    training_loss: float
    convergence_score: float
    stability_score: float
    epsilon: float
    recent_performance: List[float] = field(default_factory=list)

class StepWeightStrategy:
    """步级权重策略类"""
    
    def __init__(self, weight_components: List[WeightComponent], total_episodes: int = 1000):
        self.weight_components = {comp.name: comp for comp in weight_components}
        self.total_episodes = total_episodes
        self.current_weights = {}
        self.weight_history = []
        self.performance_history = []
        
        # 初始化权重
        for name, comp in self.weight_components.items():
            self.current_weights[name] = comp.initial_weight
        
        # 自适应参数
        self.adaptation_window = 50  # 用于计算趋势的窗口大小
        self.adaptation_threshold = 0.1  # 性能改善阈值
        self.adaptation_factor = 0.1  # 自适应调整因子
        
        print(f"初始化步级权重策略，包含 {len(self.weight_components)} 个权重组件")
        self._print_initial_config()
    
    def update_weights(self, metrics: PerformanceMetrics) -> Dict[str, float]:
        """更新权重并返回当前权重字典"""
        
        # 计算训练进度
        episode_progress = metrics.episode / self.total_episodes
        step_progress = metrics.current_step / metrics.total_steps if metrics.total_steps > 0 else 0
        
        # 更新每个权重组件
        updated_weights = {}
        for name, component in self.weight_components.items():
            new_weight = self._calculate_component_weight(
                component, episode_progress, step_progress, metrics
            )
            updated_weights[name] = new_weight
            self.current_weights[name] = new_weight
        
        # 记录历史
        self._record_history(metrics, updated_weights)
        
        return updated_weights.copy()
    
    def _calculate_component_weight(self, component: WeightComponent, 
                                  episode_progress: float, step_progress: float,
                                  metrics: PerformanceMetrics) -> float:
        """计算单个组件的权重"""
        
        if component.schedule_type == WeightScheduleType.LINEAR:
            weight = self._linear_schedule(component, episode_progress)
        
        elif component.schedule_type == WeightScheduleType.EXPONENTIAL:
            weight = self._exponential_schedule(component, episode_progress)
        
        elif component.schedule_type == WeightScheduleType.COSINE:
            weight = self._cosine_schedule(component, episode_progress)
        
        elif component.schedule_type == WeightScheduleType.SIGMOID:
            weight = self._sigmoid_schedule(component, episode_progress)
        
        elif component.schedule_type == WeightScheduleType.STEP:
            weight = self._step_schedule(component, episode_progress)
        
        elif component.schedule_type == WeightScheduleType.ADAPTIVE:
            weight = self._adaptive_schedule(component, metrics)
        
        else:
            weight = component.initial_weight
        
        # 应用权重约束
        weight = np.clip(weight, component.min_weight, component.max_weight)
        
        return weight
    
    def _linear_schedule(self, component: WeightComponent, progress: float) -> float:
        """线性调度"""
        return component.initial_weight + (component.target_weight - component.initial_weight) * progress
    
    def _exponential_schedule(self, component: WeightComponent, progress: float) -> float:
        """指数调度"""
        decay_rate = component.schedule_params.get('decay_rate', 2.0)
        if component.target_weight > component.initial_weight:
            # 指数增长
            factor = (component.target_weight / component.initial_weight) ** (progress ** (1/decay_rate))
            return component.initial_weight * factor
        else:
            # 指数衰减
            factor = np.exp(-decay_rate * progress)
            return component.target_weight + (component.initial_weight - component.target_weight) * factor
    
    def _cosine_schedule(self, component: WeightComponent, progress: float) -> float:
        """余弦调度"""
        return component.target_weight + 0.5 * (component.initial_weight - component.target_weight) * \
               (1 + np.cos(np.pi * progress))
    
    def _sigmoid_schedule(self, component: WeightComponent, progress: float) -> float:
        """sigmoid调度"""
        steepness = component.schedule_params.get('steepness', 10.0)
        midpoint = component.schedule_params.get('midpoint', 0.5)
        
        # 将进度映射到sigmoid函数
        x = steepness * (progress - midpoint)
        sigmoid_value = 1 / (1 + np.exp(-x))
        
        return component.initial_weight + (component.target_weight - component.initial_weight) * sigmoid_value
    
    def _step_schedule(self, component: WeightComponent, progress: float) -> float:
        """阶梯调度"""
        steps = component.schedule_params.get('steps', [0.25, 0.5, 0.75])
        values = component.schedule_params.get('values', None)
        
        if values is None:
            # 等步长变化
            step_size = (component.target_weight - component.initial_weight) / len(steps)
            values = [component.initial_weight + i * step_size for i in range(len(steps) + 1)]
        
        for i, step_point in enumerate(steps):
            if progress <= step_point:
                return values[i]
        
        return values[-1]
    
    def _adaptive_schedule(self, component: WeightComponent, metrics: PerformanceMetrics) -> float:
        """自适应调度"""
        current_weight = self.current_weights.get(component.name, component.initial_weight)
        
        # 根据指定的性能指标进行调整
        metric_name = component.adaptive_metric or 'avg_reward'
        current_metric = getattr(metrics, metric_name, 0.0)
        
        # 计算性能趋势
        if len(self.performance_history) >= self.adaptation_window:
            recent_metrics = [getattr(pm, metric_name, 0.0) 
                            for pm in self.performance_history[-self.adaptation_window:]]
            trend = np.polyfit(range(len(recent_metrics)), recent_metrics, 1)[0]
            
            # 根据趋势调整权重
            if component.name == 'arousal_penalty_weight':
                # 如果arousal误差增大，增加arousal惩罚权重
                if metric_name == 'avg_arousal_error' and trend > self.adaptation_threshold:
                    adjustment = self.adaptation_factor * trend
                    new_weight = current_weight + adjustment
                elif metric_name == 'avg_arousal_error' and trend < -self.adaptation_threshold:
                    adjustment = self.adaptation_factor * abs(trend)
                    new_weight = current_weight - adjustment
                else:
                    new_weight = current_weight
            
            elif component.name == 'exploration_weight':
                # 如果学习停滞，增加探索权重
                if metric_name == 'avg_reward' and abs(trend) < 0.01:
                    new_weight = current_weight + self.adaptation_factor
                else:
                    new_weight = current_weight - self.adaptation_factor * 0.5
            
            else:
                # 通用自适应规则
                if trend > self.adaptation_threshold:
                    new_weight = current_weight + self.adaptation_factor * trend
                else:
                    new_weight = current_weight
        else:
            new_weight = current_weight
        
        return new_weight
    
    def _record_history(self, metrics: PerformanceMetrics, weights: Dict[str, float]):
        """记录历史数据"""
        self.performance_history.append(metrics)
        self.weight_history.append({
            'episode': metrics.episode,
            'step': metrics.current_step,
            'weights': weights.copy(),
            'metrics': {
                'avg_reward': metrics.avg_reward,
                'avg_arousal_error': metrics.avg_arousal_error,
                'training_loss': metrics.training_loss,
                'convergence_score': metrics.convergence_score
            }
        })
        
        # 限制历史记录长度
        max_history = 5000
        if len(self.performance_history) > max_history:
            self.performance_history = self.performance_history[-max_history:]
            self.weight_history = self.weight_history[-max_history:]
    
    def _print_initial_config(self):
        """打印初始配置"""
        print("\n" + "="*60)
        print("步级权重策略配置")
        print("="*60)
        for name, comp in self.weight_components.items():
            print(f"{name}:")
            print(f"  初始权重: {comp.initial_weight}")
            print(f"  目标权重: {comp.target_weight}")
            print(f"  调度类型: {comp.schedule_type.value}")
            print(f"  权重范围: [{comp.min_weight}, {comp.max_weight}]")
            if comp.adaptive_metric:
                print(f"  自适应指标: {comp.adaptive_metric}")
            print()
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self.current_weights.copy()
    
    def get_weight_summary(self) -> Dict[str, Any]:
        """获取权重总结信息"""
        if not self.weight_history:
            return {}
        
        latest = self.weight_history[-1]
        summary = {
            'episode': latest['episode'],
            'current_weights': latest['weights'],
            'weight_changes': {},
            'performance_metrics': latest['metrics']
        }
        
        # 计算权重变化
        if len(self.weight_history) > 1:
            prev_weights = self.weight_history[-2]['weights']
            for name in latest['weights']:
                change = latest['weights'][name] - prev_weights.get(name, 0)
                summary['weight_changes'][name] = change
        
        return summary

def create_default_weight_strategy(total_episodes: int = 1000) -> StepWeightStrategy:
    """创建默认的权重策略"""
    
    weight_components = [
        # Arousal惩罚权重 - 随训练进展线性增加
        WeightComponent(
            name='arousal_penalty_weight',
            initial_weight=5.0,
            target_weight=15.0,
            schedule_type=WeightScheduleType.LINEAR,
            min_weight=1.0,
            max_weight=25.0
        ),
        
        # 安全权重 - 在训练初期较高，后期适度降低
        WeightComponent(
            name='safety_weight',
            initial_weight=10.0,
            target_weight=5.0,
            schedule_type=WeightScheduleType.EXPONENTIAL,
            schedule_params={'decay_rate': 2.0},
            min_weight=2.0,
            max_weight=15.0
        ),
        
        # 舒适度权重 - 使用sigmoid曲线，训练中期快速增加
        WeightComponent(
            name='comfort_weight',
            initial_weight=1.0,
            target_weight=5.0,
            schedule_type=WeightScheduleType.SIGMOID,
            schedule_params={'steepness': 8.0, 'midpoint': 0.4},
            min_weight=0.5,
            max_weight=8.0
        ),
        
        # 探索权重 - 自适应调整，基于奖励改善
        WeightComponent(
            name='exploration_weight',
            initial_weight=2.0,
            target_weight=0.5,
            schedule_type=WeightScheduleType.ADAPTIVE,
            adaptive_metric='avg_reward',
            min_weight=0.1,
            max_weight=5.0
        ),
        
        # 信号质量权重 - 阶梯式增加
        WeightComponent(
            name='signal_quality_weight',
            initial_weight=0.5,
            target_weight=2.0,
            schedule_type=WeightScheduleType.STEP,
            schedule_params={
                'steps': [0.2, 0.5, 0.8],
                'values': [0.5, 1.0, 1.5, 2.0]
            },
            min_weight=0.1,
            max_weight=3.0
        )
    ]
    
    return StepWeightStrategy(weight_components, total_episodes)

if __name__ == "__main__":
    # 演示用法
    print("步级权重策略演示")
    
    # 创建默认策略
    strategy = create_default_weight_strategy(total_episodes=500)
    
    # 模拟训练过程
    for episode in range(0, 500, 25):
        metrics = PerformanceMetrics(
            episode=episode,
            current_step=100,
            total_steps=200,
            avg_reward=np.random.normal(10 + episode*0.02, 2),
            avg_arousal_error=max(0.1, np.random.normal(0.5 - episode*0.001, 0.1)),
            training_loss=max(0.01, np.random.normal(2.0 - episode*0.003, 0.5)),
            convergence_score=min(1.0, episode/200),
            stability_score=min(1.0, episode/300),
            epsilon=max(0.01, 1.0 - episode/400)
        )
        
        weights = strategy.update_weights(metrics)
        
        if episode % 100 == 0:
            print(f"\nEpisode {episode}:")
            for name, weight in weights.items():
                print(f"  {name}: {weight:.3f}") 