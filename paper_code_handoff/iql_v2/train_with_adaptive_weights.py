#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
import pandas as pd
from datetime import datetime
import os
import time
import argparse

# 导入自定义模块
from adaptive_weight_environment import AdaptiveWeightNeuralFeedbackEnvironment, create_neural_feedback_weight_strategy
from step_weight_strategy import WeightComponent, WeightScheduleType, StepWeightStrategy
from compare_reward_experiments import SimpleDQNAgent

class AdaptiveDQNAgent(SimpleDQNAgent):
    """支持自适应权重的DQN代理"""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        super().__init__(state_dim, action_dim, learning_rate)
        
        # 添加权重感知的记录
        self.weight_history = []
        self.performance_metrics = {
            'episode_rewards': [],
            'weight_changes': [],
            'adaptation_events': []
        }
    
    def record_weights(self, weights: dict, episode: int):
        """记录权重变化"""
        self.weight_history.append({
            'episode': episode,
            'weights': weights.copy()
        })
    
    def record_adaptation_event(self, episode: int, event_type: str, details: dict):
        """记录权重适应事件"""
        self.performance_metrics['adaptation_events'].append({
            'episode': episode,
            'type': event_type,
            'details': details
        })

def train_with_adaptive_weights(env, agent, episodes=500, max_steps=200, 
                               save_interval=50, visualize_interval=100):
    """使用自适应权重训练代理"""
    
    print(f"开始自适应权重训练 - 总episodes: {episodes}")
    print(f"环境类型: {type(env).__name__}")
    
    # 训练历史记录
    training_history = {
        'episodes': [],
        'rewards': [],
        'losses': [],
        'arousal_errors': [],
        'episode_lengths': [],
        'weight_snapshots': [],
        'convergence_scores': [],
        'stability_scores': []
    }
    
    # 训练循环
    start_time = time.time()
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        arousal_error_sum = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.get_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 设置训练指标给环境
            if hasattr(agent, 'epsilon'):
                env.set_training_metrics(total_loss / max(step, 1), agent.epsilon)
            
            # 存储经验
            agent.store_experience(state, action, reward, next_state, done)
            
            # 训练代理
            loss = agent.train()
            
            # 累计统计
            total_reward += reward
            total_loss += loss
            steps += 1
            
            # 记录arousal误差
            if 'arousal' in info:
                arousal_error = abs(info['arousal'] - env.config['arousal_target'])
                arousal_error_sum += arousal_error
            
            # 记录权重信息
            if 'current_weights' in info:
                agent.record_weights(info['current_weights'], episode)
            
            state = next_state
            
            if done:
                break
        
        # 记录episode统计
        avg_reward = total_reward / max(steps, 1)
        avg_loss = total_loss / max(steps, 1)
        avg_arousal_error = arousal_error_sum / max(steps, 1)
        
        training_history['episodes'].append(episode)
        training_history['rewards'].append(avg_reward)
        training_history['losses'].append(avg_loss)
        training_history['arousal_errors'].append(avg_arousal_error)
        training_history['episode_lengths'].append(steps)
        
        # 获取当前权重快照
        weight_data = env.get_weight_evolution_data()
        if weight_data['current_weights']:
            training_history['weight_snapshots'].append({
                'episode': episode,
                'weights': weight_data['current_weights'].copy()
            })
        
        # 计算收敛和稳定性分数
        convergence_score = calculate_convergence_score(training_history['rewards'][-50:])
        stability_score = calculate_stability_score(training_history['arousal_errors'][-50:])
        
        training_history['convergence_scores'].append(convergence_score)
        training_history['stability_scores'].append(stability_score)
        
        # 定期输出训练进度
        if episode % 20 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode {episode}/{episodes}:")
            print(f"  平均奖励: {avg_reward:.3f}")
            print(f"  平均Arousal误差: {avg_arousal_error:.3f}")
            print(f"  训练损失: {avg_loss:.4f}")
            print(f"  收敛分数: {convergence_score:.3f}")
            print(f"  稳定性分数: {stability_score:.3f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  已用时间: {elapsed_time:.1f}s")
            
            # 显示当前权重
            current_weights = env.weight_strategy.get_current_weights()
            print("  当前权重:")
            for name, weight in current_weights.items():
                print(f"    {name}: {weight:.3f}")
            print()
        
        # 定期保存模型和可视化
        if episode % save_interval == 0 and episode > 0:
            save_training_checkpoint(agent, env, training_history, episode)
        
        if episode % visualize_interval == 0 and episode > 0:
            visualize_training_progress(training_history, env, episode)
    
    # 训练完成
    total_time = time.time() - start_time
    print(f"\n训练完成！总用时: {total_time:.1f}s")
    
    # 最终评估和保存
    final_evaluation(agent, env, training_history)
    
    return training_history

def calculate_convergence_score(rewards):
    """计算收敛分数"""
    if len(rewards) < 10:
        return 0.0
    
    # 计算趋势稳定性
    recent_rewards = rewards[-10:]
    if len(set(recent_rewards)) == 1:
        return 1.0
    
    # 使用变异系数
    mean_reward = np.mean(recent_rewards)
    std_reward = np.std(recent_rewards)
    if mean_reward == 0:
        return 0.0
    
    cv = std_reward / abs(mean_reward)
    return max(0.0, 1.0 - cv)

def calculate_stability_score(arousal_errors):
    """计算稳定性分数"""
    if len(arousal_errors) < 10:
        return 0.0
    
    recent_errors = arousal_errors[-10:]
    error_std = np.std(recent_errors)
    return max(0.0, 1.0 - error_std)

def save_training_checkpoint(agent, env, training_history, episode):
    """保存训练检查点"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_path = f"adaptive_model_ep{episode}_{timestamp}.pt"
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'episode': episode,
        'epsilon': agent.epsilon,
        'training_history': training_history
    }, model_path)
    
    # 保存训练数据
    data_path = f"adaptive_training_data_ep{episode}_{timestamp}.json"
    env.save_training_data(data_path)
    
    print(f"  检查点已保存: {model_path}, {data_path}")

def visualize_training_progress(training_history, env, episode):
    """可视化训练进度"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建综合训练报告
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'自适应权重训练进度报告 (Episode {episode})', fontsize=16, fontweight='bold')
    
    episodes = training_history['episodes']
    
    # 奖励趋势
    axes[0, 0].plot(episodes, training_history['rewards'], 'b-', linewidth=2)
    axes[0, 0].set_title('平均奖励趋势')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('平均奖励')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Arousal误差趋势
    axes[0, 1].plot(episodes, training_history['arousal_errors'], 'r-', linewidth=2)
    axes[0, 1].set_title('Arousal控制误差')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('平均Arousal误差')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 收敛和稳定性分数
    axes[1, 0].plot(episodes, training_history['convergence_scores'], 'g-', linewidth=2, label='收敛分数')
    axes[1, 0].plot(episodes, training_history['stability_scores'], 'orange', linewidth=2, label='稳定性分数')
    axes[1, 0].set_title('学习质量指标')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('分数')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 训练损失
    axes[1, 1].plot(episodes, training_history['losses'], 'purple', linewidth=2)
    axes[1, 1].set_title('训练损失')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('平均损失')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 权重演化
    if training_history['weight_snapshots']:
        weight_names = ['arousal_penalty_weight', 'safety_weight', 'comfort_weight']
        for name in weight_names:
            weights = []
            weight_episodes = []
            for snapshot in training_history['weight_snapshots']:
                if name in snapshot['weights']:
                    weights.append(snapshot['weights'][name])
                    weight_episodes.append(snapshot['episode'])
            
            if weights:
                axes[2, 0].plot(weight_episodes, weights, linewidth=2, label=name)
        
        axes[2, 0].set_title('权重演化')
        axes[2, 0].set_xlabel('Episode')
        axes[2, 0].set_ylabel('权重值')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
    
    # Episode长度分布
    axes[2, 1].hist(training_history['episode_lengths'], bins=20, alpha=0.7, color='skyblue')
    axes[2, 1].set_title('Episode长度分布')
    axes[2, 1].set_xlabel('Episode长度')
    axes[2, 1].set_ylabel('频次')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = f"adaptive_training_progress_ep{episode}_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  训练进度图已保存: {save_path}")
    
    plt.show()

def final_evaluation(agent, env, training_history):
    """最终评估和报告"""
    print("\n" + "="*80)
    print("训练完成 - 最终评估报告")
    print("="*80)
    
    # 计算最终统计
    final_episodes = 50
    final_rewards = training_history['rewards'][-final_episodes:]
    final_arousal_errors = training_history['arousal_errors'][-final_episodes:]
    final_convergence = training_history['convergence_scores'][-final_episodes:]
    
    print(f"\n最后{final_episodes}个episodes的表现:")
    print(f"平均奖励: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
    print(f"平均Arousal误差: {np.mean(final_arousal_errors):.3f} ± {np.std(final_arousal_errors):.3f}")
    print(f"平均收敛分数: {np.mean(final_convergence):.3f}")
    
    # 权重策略效果分析
    print(f"\n权重策略效果分析:")
    weight_data = env.get_weight_evolution_data()
    if weight_data['weight_history']:
        initial_weights = weight_data['weight_history'][0]['weights']
        final_weights = weight_data['weight_history'][-1]['weights']
        
        print("权重变化总结:")
        for name in initial_weights:
            if name in final_weights:
                change = final_weights[name] - initial_weights[name]
                change_pct = (change / initial_weights[name]) * 100 if initial_weights[name] != 0 else 0
                print(f"  {name}: {initial_weights[name]:.3f} → {final_weights[name]:.3f} "
                      f"(变化: {change:+.3f}, {change_pct:+.1f}%)")
    
    # 保存最终结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存训练历史
    df = pd.DataFrame(training_history)
    history_file = f"adaptive_training_history_{timestamp}.csv"
    df.to_csv(history_file, index=False, encoding='utf-8')
    print(f"\n训练历史已保存: {history_file}")
    
    # 保存最终模型
    final_model_path = f"adaptive_final_model_{timestamp}.pt"
    torch.save({
        'model_state_dict': agent.q_network.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'training_history': training_history,
        'final_performance': {
            'avg_reward': np.mean(final_rewards),
            'avg_arousal_error': np.mean(final_arousal_errors),
            'convergence_score': np.mean(final_convergence)
        }
    }, final_model_path)
    print(f"最终模型已保存: {final_model_path}")
    
    # 生成最终可视化
    env.visualize_training_progress(f"final_training_progress_{timestamp}.png")
    
    print("\n训练评估完成！")

def create_experiment_configs():
    """创建不同的实验配置"""
    configs = {
        'conservative': {
            'weight_strategy': 'conservative',
            'description': '保守策略 - 稳定优先'
        },
        'aggressive': {
            'weight_strategy': 'aggressive', 
            'description': '激进策略 - 快速学习'
        },
        'adaptive': {
            'weight_strategy': 'adaptive',
            'description': '自适应策略 - 动态调整'
        }
    }
    return configs

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自适应权重强化学习训练')
    parser.add_argument('--episodes', type=int, default=300, help='训练episodes数量')
    parser.add_argument('--max_steps', type=int, default=200, help='每个episode最大步数')
    parser.add_argument('--strategy', type=str, default='adaptive', 
                       choices=['conservative', 'aggressive', 'adaptive'],
                       help='权重策略类型')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--save_interval', type=int, default=50, help='保存间隔')
    parser.add_argument('--visualize_interval', type=int, default=100, help='可视化间隔')
    
    args = parser.parse_args()
    
    print("自适应权重强化学习训练系统")
    print(f"配置: Episodes={args.episodes}, Strategy={args.strategy}")
    print("="*60)
    
    # 创建权重策略
    if args.strategy == 'adaptive':
        weight_strategy = create_neural_feedback_weight_strategy(total_episodes=args.episodes)
    else:
        # 可以添加其他策略类型
        weight_strategy = create_neural_feedback_weight_strategy(total_episodes=args.episodes)
    
    # 创建环境
    env_config = {
        'max_timesteps': args.max_steps,
        'total_episodes': args.episodes,
        'arousal_target': 0.5,
        'feedback_cost': 0.01,
        'arousal_penalty_weight': 10.0,
        'safety_weight': 5.0,
        'comfort_weight': 2.0
    }
    
    env = AdaptiveWeightNeuralFeedbackEnvironment(
        config=env_config,
        weight_strategy=weight_strategy
    )
    
    # 创建代理
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = AdaptiveDQNAgent(state_dim, action_dim, args.lr)
    
    print(f"环境: {state_dim}维状态空间, {action_dim}维动作空间")
    print(f"代理: DQN网络, 学习率={args.lr}")
    
    # 开始训练
    training_history = train_with_adaptive_weights(
        env=env,
        agent=agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        save_interval=args.save_interval,
        visualize_interval=args.visualize_interval
    )

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    main() 