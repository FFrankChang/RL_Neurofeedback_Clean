import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import random
from typing import Dict, List, Tuple, Optional, Union
import heapq
from dataclasses import dataclass
import math

# 定义经验元组
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

@dataclass
class IQLConfig:
    """IQL算法配置"""
    # 网络参数
    state_dim: int = 10  # 统一为10维瞳孔状态（论文式4-9）
    action_dim: int = 5
    hidden_dim: int = 512
    num_layers: int = 4
    
    # 训练参数
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005  # 软更新参数
    
    # IQL特定参数
    expectile: float = 0.8  # IQL期望分位数
    temperature: float = 3.0  # AWR温度参数
    clip_score: float = 100.0  # 优势函数裁剪
    
    # 探索参数
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # 优先级经验回放参数
    alpha: float = 0.6  # 优先级指数
    beta_start: float = 0.4  # 重要性采样初始值
    beta_frames: int = 100000  # beta增长帧数
    
    # 训练频率
    update_frequency: int = 4
    target_update_frequency: int = 1000
    
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class PriorityExperienceReplay:
    """优先级经验回放缓冲区"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def push(self, experience: Experience, priority: float = None):
        """添加经验到缓冲区"""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4):
        """优先级采样"""
        if len(self.buffer) < batch_size:
            return None, None, None
            
        # 计算采样概率
        priorities = np.array(self.priorities[:len(self.buffer)])
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 计算重要性权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # 获取经验
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
        
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)

class NoisyLinear(nn.Module):
    """噪声线性层，用于参数空间探索"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # 可学习参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # 噪声
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        """初始化参数"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def _scale_noise(self, size: int):
        """生成缩放噪声"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
        
    def forward(self, x: torch.Tensor):
        """前向传播"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)

class AttentionModule(nn.Module):
    """多头注意力模块，处理时序EEG数据"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """前向传播"""
        batch_size, seq_len, _ = x.shape
        
        # 生成Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 残差连接和层归一化
        output = self.layer_norm(x + self.w_o(attention_output))
        
        return output, attention_weights

class DuellingNetwork(nn.Module):
    """双流网络架构，分别估计状态值和优势函数"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512, 
                 num_layers: int = 4, use_attention: bool = True, use_noisy: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_attention = use_attention
        self.use_noisy = use_noisy
        
        # 特征提取层
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim
        
        for i in range(num_layers - 1):
            if use_noisy and i >= num_layers - 2:
                layer = NoisyLinear(prev_dim, hidden_dim)
            else:
                layer = nn.Linear(prev_dim, hidden_dim)
            self.feature_layers.append(layer)
            prev_dim = hidden_dim
            
        # 注意力模块（处理EEG特征）
        if use_attention:
            self.attention = AttentionModule(hidden_dim)
            
        # 值函数头
        if use_noisy:
            self.value_head = NoisyLinear(hidden_dim, 1)
        else:
            self.value_head = nn.Linear(hidden_dim, 1)
            
        # 优势函数头
        if use_noisy:
            self.advantage_head = NoisyLinear(hidden_dim, action_dim)
        else:
            self.advantage_head = nn.Linear(hidden_dim, action_dim)
            
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor):
        """前向传播"""
        # 特征提取
        for layer in self.feature_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
            
        if self.use_attention and x.dim() == 3:
            x, _ = self.attention(x)
            x = x.mean(dim=1)  # 全局平均池化
        elif x.dim() == 3:
            x = x.mean(dim=1)
            
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        
        # 合并为Q值
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values
        
    def reset_noise(self):
        """重置噪声层"""
        if self.use_noisy:
            for layer in self.feature_layers:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()
            if isinstance(self.value_head, NoisyLinear):
                self.value_head.reset_noise()
            if isinstance(self.advantage_head, NoisyLinear):
                self.advantage_head.reset_noise()

class ImplicitQLearning:
    
    def __init__(self, config: IQLConfig):
        self.config = config
        self.device = torch.device(config.device)
        
    
        self.q_network = DuellingNetwork(
            config.state_dim, config.action_dim, 
            config.hidden_dim, config.num_layers
        ).to(self.device)
        
        self.target_q_network = DuellingNetwork(
            config.state_dim, config.action_dim,
            config.hidden_dim, config.num_layers
        ).to(self.device)
        
        self.v_network = DuellingNetwork(
            config.state_dim, 1,
            config.hidden_dim, config.num_layers
        ).to(self.device)
        
        # 策略网络
        self.policy_network = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim)
        ).to(self.device)
        
        # 优化器
        self.q_optimizer = optim.AdamW(self.q_network.parameters(), lr=config.learning_rate)
        self.v_optimizer = optim.AdamW(self.v_network.parameters(), lr=config.learning_rate)
        self.policy_optimizer = optim.AdamW(self.policy_network.parameters(), lr=config.learning_rate)
        
        # 经验回放缓冲区
        self.memory = PriorityExperienceReplay(config.buffer_size, config.alpha)
        
        # 训练计数器
        self.update_count = 0
        self.epsilon = config.epsilon_start
        
        # 将目标网络参数复制过来
        self.update_target_network()
        
        # 性能追踪
        self.losses = {
            'q_loss': [],
            'v_loss': [],
            'policy_loss': [],
            'total_loss': []
        }
        
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作"""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.config.action_dim)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if training:
                # 训练时使用策略网络
                logits = self.policy_network(state_tensor)
                action_probs = F.softmax(logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()
            else:
                # 评估时使用Q网络
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
                
        return action
        
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储经验"""
        experience = Experience(state, action, reward, next_state, done)
        self.memory.push(experience)
        
    def expectile_loss(self, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """期望分位数损失"""
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff**2)
        
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.memory) < self.config.batch_size:
            return {}
            
        # 计算beta值
        beta = min(1.0, self.config.beta_start + 
                  (1.0 - self.config.beta_start) * self.update_count / self.config.beta_frames)
        
        # 采样经验
        experiences, indices, weights = self.memory.sample(self.config.batch_size, beta)
        if experiences is None:
            return {}
            
        # 转换为张量
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        weights_tensor = torch.FloatTensor(weights).to(self.device)
        
        # 重置噪声
        self.q_network.reset_noise()
        self.target_q_network.reset_noise()
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_v_values = self.v_network(next_states).squeeze(1)
            target_q_values = rewards + self.config.gamma * next_v_values * (~dones)
            
        # Q损失
        q_diff = target_q_values.unsqueeze(1) - current_q_values
        q_loss = (weights_tensor.unsqueeze(1) * F.mse_loss(
            current_q_values, target_q_values.unsqueeze(1), reduction='none'
        )).mean()
        
        with torch.no_grad():
            target_q_all = self.target_q_network(states)
            target_q_max = target_q_all.max(dim=1, keepdim=True)[0]
            
        current_v_values = self.v_network(states)
        v_diff = target_q_max - current_v_values
        v_loss = self.expectile_loss(v_diff, self.config.expectile).mean()
        
        # 策略损失
        with torch.no_grad():
            advantages = current_q_values - current_v_values.gather(1, actions.unsqueeze(1))
            advantages = torch.clamp(advantages, -self.config.clip_score, self.config.clip_score)
            exp_advantages = torch.exp(advantages / self.config.temperature)
            
        policy_logits = self.policy_network(states)
        log_probs = F.log_softmax(policy_logits, dim=-1).gather(1, actions.unsqueeze(1))
        policy_loss = -(exp_advantages * log_probs).mean()
        
        # 更新Q网络
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.q_optimizer.step()
        
        # 更新值网络
        self.v_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_network.parameters(), 1.0)
        self.v_optimizer.step()
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # 更新优先级
        with torch.no_grad():
            td_errors = torch.abs(q_diff.squeeze()).cpu().numpy()
            new_priorities = td_errors + 1e-6
            self.memory.update_priorities(indices, new_priorities)
        
        # 软更新目标网络
        if self.update_count % self.config.target_update_frequency == 0:
            self.soft_update_target_network()
            
        # 衰减探索
        self.epsilon = max(self.config.epsilon_end, 
                          self.epsilon * self.config.epsilon_decay)
        
        self.update_count += 1
        
        # 记录损失
        losses = {
            'q_loss': q_loss.item(),
            'v_loss': v_loss.item(),
            'policy_loss': policy_loss.item(),
            'total_loss': q_loss.item() + v_loss.item() + policy_loss.item()
        }
        
        for key, value in losses.items():
            self.losses[key].append(value)
            
        return losses
        
    def soft_update_target_network(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_q_network.parameters(), 
                                     self.q_network.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
            
    def update_target_network(self):
        """硬更新目标网络"""
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'v_network': self.v_network.state_dict(),
            'policy_network': self.policy_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'config': self.config,
            'update_count': self.update_count,
            'epsilon': self.epsilon
        }, filepath)
        
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.v_network.load_state_dict(checkpoint['v_network'])
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.update_count = checkpoint['update_count']
        self.epsilon = checkpoint['epsilon']
        
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """获取Q值用于分析"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy().flatten()
        
    def get_policy_probs(self, state: np.ndarray) -> np.ndarray:
        """获取策略概率分布"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy_network(state_tensor)
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy().flatten() 