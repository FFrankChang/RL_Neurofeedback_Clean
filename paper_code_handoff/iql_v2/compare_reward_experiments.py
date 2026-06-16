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

# 导入自定义模块
from neural_feedback_env import NeuralFeedbackEnvironment
from neural_feedback_env_no_performance import NeuralFeedbackEnvironmentNoPerformance

class SimpleDQNAgent:
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.memory = deque(maxlen=10000)
        
        # 探索参数
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def get_action(self, state, training=True):
        """选择动作"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self, batch_size=32):
        """训练网络"""
        if len(self.memory) < batch_size:
            return 0.0
        
        # 随机采样
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(self.device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(self.device)
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(self.device)
        
        # 计算Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

def train_agent(env, agent, episodes=500, max_steps=200):
    """训练代理"""
    rewards_history = []
    losses_history = []
    arousal_errors = []
    episode_lengths = []
    
    print(f"开始训练 - 环境类型: {type(env).__name__}")
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        arousal_error_sum = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_experience(state, action, reward, next_state, done)
            loss = agent.train()
            
            total_reward += reward
            total_loss += loss
            steps += 1
            
            # 记录arousal误差
            if 'arousal' in info:
                arousal_error = abs(info['arousal'] - env.config['arousal_target'])
                arousal_error_sum += arousal_error
            
            state = next_state
            
            if done:
                break
        
        rewards_history.append(total_reward)
        losses_history.append(total_loss / max(steps, 1))
        arousal_errors.append(arousal_error_sum / max(steps, 1))
        episode_lengths.append(steps)
        
        if episode % 50 == 0:
            avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
            avg_arousal_error = np.mean(arousal_errors[-50:]) if len(arousal_errors) >= 50 else np.mean(arousal_errors)
            print(f"Episode {episode}: Avg Reward: {avg_reward:.3f}, Avg Arousal Error: {avg_arousal_error:.3f}, Epsilon: {agent.epsilon:.3f}")
    
    return {
        'rewards': rewards_history,
        'losses': losses_history,
        'arousal_errors': arousal_errors,
        'episode_lengths': episode_lengths
    }

def run_comparison_experiment():
    """运行对比实验"""
    print("=" * 60)
    print("强化学习奖励函数对比实验")
    print("实验目的：展示移除驾驶绩效指标后模型不收敛的结果")
    print("=" * 60)
    
    # 实验配置
    episodes = 300
    max_steps = 200
    
    # 创建两个环境
    print("\n1. 创建环境...")
    env_complete = NeuralFeedbackEnvironment()
    env_no_performance = NeuralFeedbackEnvironmentNoPerformance()
    
    # 创建代理
    print("2. 创建代理...")
    state_dim = env_complete.observation_space.shape[0]
    action_dim = env_complete.action_space.n
    
    agent_complete = SimpleDQNAgent(state_dim, action_dim)
    agent_no_performance = SimpleDQNAgent(state_dim, action_dim)
    
    # 训练完整奖励函数的代理
    print("\n3. 训练完整奖励函数代理（包含驾驶绩效指标）...")
    results_complete = train_agent(env_complete, agent_complete, episodes, max_steps)
    

def visualize_comparison_results(results_complete, results_no_performance, episodes):
    """可视化对比结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('强化学习奖励函数对比实验结果', fontsize=16, fontweight='bold')
    
    # 计算移动平均
    window = 20
    
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # 奖励对比
    axes[0, 0].plot(moving_average(results_complete['rewards'], window), 
                   label='完整奖励函数（含驾驶绩效指标）', linewidth=2, color='blue')
    axes[0, 0].plot(moving_average(results_no_performance['rewards'], window), 
                   label='移除驾驶绩效指标', linewidth=2, color='red', linestyle='--')
    axes[0, 0].set_title('训练奖励对比', fontweight='bold')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('平均奖励')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Arousal误差对比
    axes[0, 1].plot(moving_average(results_complete['arousal_errors'], window), 
                   label='完整奖励函数', linewidth=2, color='blue')
    axes[0, 1].plot(moving_average(results_no_performance['arousal_errors'], window), 
                   label='移除驾驶绩效指标', linewidth=2, color='red', linestyle='--')
    axes[0, 1].set_title('Arousal控制误差对比', fontweight='bold')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('平均Arousal误差')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 训练损失对比
    axes[1, 0].plot(moving_average(results_complete['losses'], window), 
                   label='完整奖励函数', linewidth=2, color='blue')
    axes[1, 0].plot(moving_average(results_no_performance['losses'], window), 
                   label='移除驾驶绩效指标', linewidth=2, color='red', linestyle='--')
    axes[1, 0].set_title('训练损失对比', fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('平均损失')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Episode长度对比
    axes[1, 1].plot(moving_average(results_complete['episode_lengths'], window), 
                   label='完整奖励函数', linewidth=2, color='blue')
    axes[1, 1].plot(moving_average(results_no_performance['episode_lengths'], window), 
                   label='移除驾驶绩效指标', linewidth=2, color='red', linestyle='--')
    axes[1, 1].set_title('Episode长度对比', fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('平均步数')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"reward_function_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"对比图表已保存为: {filename}")
    
    plt.show()

def save_experiment_results(results_complete, results_no_performance):
    """保存实验结果到CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建DataFrame
    df = pd.DataFrame({
        'episode': range(len(results_complete['rewards'])),
        'complete_reward': results_complete['rewards'],
        'no_performance_reward': results_no_performance['rewards'],
        'complete_arousal_error': results_complete['arousal_errors'],
        'no_performance_arousal_error': results_no_performance['arousal_errors'],
        'complete_loss': results_complete['losses'],
        'no_performance_loss': results_no_performance['losses'],
        'complete_episode_length': results_complete['episode_lengths'],
        'no_performance_episode_length': results_no_performance['episode_lengths']
    })
    
    filename = f"reward_comparison_results_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"详细结果已保存为: {filename}")

def print_experiment_summary(results_complete, results_no_performance):
    """打印实验总结"""
    print("\n" + "=" * 60)
    print("实验结果总结")
    print("=" * 60)
    
    # 计算最后50个episode的平均表现
    last_episodes = 50
    
    complete_final_reward = np.mean(results_complete['rewards'][-last_episodes:])
    no_perf_final_reward = np.mean(results_no_performance['rewards'][-last_episodes:])
    
    complete_final_error = np.mean(results_complete['arousal_errors'][-last_episodes:])
    no_perf_final_error = np.mean(results_no_performance['arousal_errors'][-last_episodes:])
    
    complete_stability = np.std(results_complete['rewards'][-last_episodes:])
    no_perf_stability = np.std(results_no_performance['rewards'][-last_episodes:])
    
    print(f"\n最后{last_episodes}个episode的平均表现:")
    print(f"{'指标':<20} {'完整奖励函数':<15} {'移除驾驶绩效':<15} {'差异':<15}")
    print("-" * 65)
    print(f"{'平均奖励':<20} {complete_final_reward:<15.3f} {no_perf_final_reward:<15.3f} {complete_final_reward-no_perf_final_reward:<15.3f}")
    print(f"{'Arousal误差':<20} {complete_final_error:<15.3f} {no_perf_final_error:<15.3f} {no_perf_final_error-complete_final_error:<15.3f}")
    print(f"{'奖励稳定性(std)':<20} {complete_stability:<15.3f} {no_perf_stability:<15.3f} {no_perf_stability-complete_stability:<15.3f}")
    
    print(f"\n关键发现:")
    if complete_final_reward > no_perf_final_reward:
        print("✓ 完整奖励函数的代理获得了更高的平均奖励")
    else:
        print("✗ 移除驾驶绩效指标的代理获得了更高的平均奖励")
    
    if complete_final_error < no_perf_final_error:
        print("✓ 完整奖励函数的代理实现了更好的Arousal控制")
    else:
        print("✗ 移除驾驶绩效指标的代理实现了更好的Arousal控制")
    
    if complete_stability < no_perf_stability:
        print("✓ 完整奖励函数的代理展现了更好的稳定性")
    else:
        print("✗ 移除驾驶绩效指标的代理展现了更好的稳定性")
    
    # 收敛性分析
    print(f"\n收敛性分析:")
    complete_trend = np.polyfit(range(len(results_complete['rewards'])), results_complete['rewards'], 1)[0]
    no_perf_trend = np.polyfit(range(len(results_no_performance['rewards'])), results_no_performance['rewards'], 1)[0]
    
    print(f"完整奖励函数训练趋势: {'上升' if complete_trend > 0 else '下降'} (斜率: {complete_trend:.6f})")
    print(f"移除驾驶绩效训练趋势: {'上升' if no_perf_trend > 0 else '下降'} (斜率: {no_perf_trend:.6f})")
    
    if abs(complete_trend) > abs(no_perf_trend):
        print("✓ 完整奖励函数展现了更强的学习趋势")
    else:
        print("✗ 移除驾驶绩效指标展现了更强的学习趋势")
    
    print("\n结论(基于本次运行的实测指标):")
    better_reward = complete_final_reward > no_perf_final_reward
    better_trend = complete_trend > no_perf_trend
    if better_reward and better_trend:
        print("完整奖励函数在最终平均奖励与学习趋势上均优于消融版本，")
        print("说明驾驶绩效指标对本次训练的收敛与表现有正向贡献。")
    elif better_reward or better_trend:
        print("完整奖励函数在部分指标上优于消融版本，驾驶绩效指标的作用为部分正向，")
        print("建议结合多次随机种子重复实验以确认显著性。")
    else:
        print("本次运行中消融版本未见明显劣化，驾驶绩效指标的作用不显著，")
        print("建议结合多次随机种子重复实验以确认结论。")
    print("=" * 60)

if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 运行对比实验
    run_comparison_experiment() 