import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import pickle
import os
import json
from datetime import datetime
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from neural_feedback_env import NeuralFeedbackEnvironment, FeedbackType
from neural_feedback_iql import ImplicitQLearning, IQLConfig
from group_wrappers import GroupRewardWrapper

try:
    from joblib import load as joblib_load
except ImportError:
    joblib_load = None

class NeuralFeedbackTrainer:
    """神经反馈强化学习训练器"""
    
    def __init__(self, config_overrides: dict = None):
        # 默认训练配置
        self.training_config = {
            'num_episodes': 10000,
            'max_steps_per_episode': 1000,
            'evaluation_frequency': 500,
            'save_frequency': 1000,
            'num_eval_episodes': 10,
            'render_eval': False,
            'early_stopping_patience': 2000,
            'target_reward': -5.0,  # 目标平均奖励
            # 论文5.4.1/5.4.2：默认启用人群分型适配闭环（分型 -> 差异化奖励 -> IQL）
            'enable_group_adaptation': True,
            'num_groups': 4,
            'subject_id': None,                      # 如 "S01"
            'fixed_group_id': None,                  # 指定后优先使用
            'group_mapping_json': None,              # subject_id -> group_id 的JSON映射
            'group_classifier_model_path': None,     # 3.7.2分类器 .joblib
            'group_classifier_input_path': None,     # 单被试SPD张量 .npy/.npz
            'fallback_group_strategy': 'round_robin' # round_robin | random | fixed_1
        }
        
        if config_overrides:
            self.training_config.update(config_overrides)
        
        # 创建环境配置
        self.env_config = {
            'max_timesteps': self.training_config['max_steps_per_episode'],
            'sampling_rate': 250,
            'arousal_target': 0.4,
            'feedback_cost': 0.01,
            'arousal_penalty_weight': 10.0,
            'safety_weight': 5.0,
            'comfort_weight': 2.0
        }
        
        # 创建环境（默认接入人群分型奖励包装器）
        self.enable_group_adaptation = bool(self.training_config.get('enable_group_adaptation', True))
        self.num_groups = int(self.training_config.get('num_groups', 4))
        self.group_mapping = self._load_group_mapping(self.training_config.get('group_mapping_json'))
        self.group_classifier = self._load_group_classifier(
            self.training_config.get('group_classifier_model_path')
        )
        self.subject_group_id = self._resolve_subject_group_id()

        base_env = NeuralFeedbackEnvironment(self.env_config)
        base_eval_env = NeuralFeedbackEnvironment(self.env_config)
        if self.enable_group_adaptation:
            self.env = GroupRewardWrapper(base_env, num_groups=self.num_groups)
            self.eval_env = GroupRewardWrapper(base_eval_env, num_groups=self.num_groups)
        else:
            self.env = base_env
            self.eval_env = base_eval_env
        
        # 创建IQL配置
        self.iql_config = IQLConfig(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            hidden_dim=512,
            num_layers=4,
            learning_rate=3e-4,
            batch_size=256,
            buffer_size=1000000,
            gamma=0.99,
            tau=0.005,
            expectile=0.8,
            temperature=3.0,
            clip_score=100.0,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.9995,  # 更慢的衰减
            alpha=0.6,
            beta_start=0.4,
            beta_frames=100000,
            update_frequency=4,
            target_update_frequency=1000
        )
        
        # 创建智能体
        self.agent = ImplicitQLearning(self.iql_config)
        
        # 训练统计
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'arousal_tracking_error': [],
            'feedback_usage_rate': [],
            'group_id_history': [],
            'evaluation_rewards': [],
            'evaluation_arousal_errors': [],
            'loss_history': {
                'q_loss': [],
                'v_loss': [],
                'policy_loss': [],
                'total_loss': []
            }
        }
        
        # 创建保存目录
        self.save_dir = f"models/neural_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train(self):
        """主训练循环"""
        print("开始神经反馈强化学习训练...")
        print(f"环境状态维度: {self.env.observation_space.shape[0]}")
        print(f"动作空间大小: {self.env.action_space.n}")
        print(f"设备: {self.iql_config.device}")
        if self.enable_group_adaptation:
            print(f"人群分型适配: 已启用 (num_groups={self.num_groups})")
            print(f"固定被试分组: {self.subject_group_id if self.subject_group_id is not None else '未固定，按回退策略分配'}")
        print(f"模型保存路径: {self.save_dir}")
        print("-" * 60)
        
        best_eval_reward = float('-inf')
        episodes_since_improvement = 0
        
        for episode in tqdm(range(self.training_config['num_episodes']), desc="训练进度"):
            group_id = self._assign_group_for_episode(episode, training=True)
            self._set_group(self.env, group_id)
            self._set_group(self.eval_env, group_id)

            # 训练一个episode
            episode_reward, episode_length, episode_stats = self._train_episode()
            
            # 记录统计信息
            self.training_stats['episode_rewards'].append(episode_reward)
            self.training_stats['episode_lengths'].append(episode_length)
            self.training_stats['arousal_tracking_error'].append(episode_stats['avg_arousal_error'])
            self.training_stats['feedback_usage_rate'].append(episode_stats['feedback_rate'])
            self.training_stats['group_id_history'].append(group_id)
            
            # 定期评估
            if (episode + 1) % self.training_config['evaluation_frequency'] == 0:
                eval_reward, eval_arousal_error = self._evaluate()
                self.training_stats['evaluation_rewards'].append(eval_reward)
                self.training_stats['evaluation_arousal_errors'].append(eval_arousal_error)
                
                print(f"\n第{episode+1}轮 - 评估奖励: {eval_reward:.3f}, Arousal误差: {eval_arousal_error:.4f}")
                print(f"训练奖励(最近100轮): {np.mean(self.training_stats['episode_rewards'][-100:]):.3f}")
                print(f"探索率: {self.agent.epsilon:.4f}")
                print(f"反馈使用率: {episode_stats['feedback_rate']:.3f}")
                if self.enable_group_adaptation:
                    print(f"当前分组ID: {group_id}")
                
                # 早停检查
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    episodes_since_improvement = 0
                    # 保存最佳模型
                    self.agent.save_model(os.path.join(self.save_dir, 'best_model.pth'))
                else:
                    episodes_since_improvement += self.training_config['evaluation_frequency']
                
                # 检查是否达到目标
                if eval_reward >= self.training_config['target_reward']:
                    print(f"达到目标奖励 {self.training_config['target_reward']}! 训练完成。")
                    break
                    
                # 早停
                if episodes_since_improvement >= self.training_config['early_stopping_patience']:
                    print(f"连续{self.training_config['early_stopping_patience']}轮无改善，早停训练。")
                    break
            
            # 定期保存模型
            if (episode + 1) % self.training_config['save_frequency'] == 0:
                self.agent.save_model(os.path.join(self.save_dir, f'model_episode_{episode+1}.pth'))
                self._save_training_stats()
        
        # 训练结束后的处理
        print(f"\n训练完成! 最佳评估奖励: {best_eval_reward:.3f}")
        
        # 保存最终模型和统计信息
        self.agent.save_model(os.path.join(self.save_dir, 'final_model.pth'))
        self._save_training_stats()
        
        # 生成训练报告
        self._generate_training_report()
        
        return best_eval_reward

    def _load_group_mapping(self, mapping_path: Optional[str]) -> dict:
        if not mapping_path:
            return {}
        if not os.path.exists(mapping_path):
            print(f"警告: group_mapping_json 不存在: {mapping_path}")
            return {}
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            print("警告: group_mapping_json 不是字典，忽略。")
            return {}
        return mapping

    def _load_group_classifier(self, model_path: Optional[str]):
        if not model_path:
            return None
        if joblib_load is None:
            print("警告: 未安装 joblib，无法加载人群分类器。")
            return None
        if not os.path.exists(model_path):
            print(f"警告: group_classifier_model_path 不存在: {model_path}")
            return None
        try:
            return joblib_load(model_path)
        except Exception as exc:
            print(f"警告: 加载人群分类器失败: {exc}")
            return None

    def _load_spd_input(self, path: Optional[str]) -> Optional[np.ndarray]:
        if not path or not os.path.exists(path):
            return None
        try:
            if path.endswith('.npz'):
                npz = np.load(path)
                key = 'X_spd' if 'X_spd' in npz else npz.files[0]
                x = np.asarray(npz[key], dtype=float)
            else:
                x = np.asarray(np.load(path), dtype=float)
            if x.ndim == 2:
                x = x[None, None, :, :]  # (1,1,C,C)
            elif x.ndim == 3:
                if x.shape[1] == x.shape[2]:
                    x = x[None, :, :, :]   # (1,V,C,C) 或 (1,C,C) 已扩展
                else:
                    return None
            elif x.ndim == 4:
                pass
            else:
                return None
            return x
        except Exception as exc:
            print(f"警告: 读取分类器输入SPD失败: {exc}")
            return None

    def _predict_group_id_by_classifier(self) -> Optional[int]:
        if self.group_classifier is None:
            return None
        spd_input_path = self.training_config.get('group_classifier_input_path')
        x = self._load_spd_input(spd_input_path)
        if x is None:
            print("警告: 未提供有效 group_classifier_input_path，无法自动分型。")
            return None
        try:
            pred = self.group_classifier.predict(x)
            raw_group = int(np.asarray(pred).reshape(-1)[0])
            # 分类器通常输出0-3，训练环境使用1-4
            if raw_group < 1:
                return raw_group + 1
            return raw_group
        except Exception as exc:
            print(f"警告: 分类器分型失败: {exc}")
            return None

    def _normalize_group_id(self, group_id: int) -> int:
        gid = int(group_id)
        if gid < 1:
            gid = 1
        if gid > self.num_groups:
            gid = ((gid - 1) % self.num_groups) + 1
        return gid

    def _resolve_subject_group_id(self) -> Optional[int]:
        if not self.enable_group_adaptation:
            return None
        fixed = self.training_config.get('fixed_group_id')
        if fixed is not None:
            return self._normalize_group_id(int(fixed))

        subject_id = self.training_config.get('subject_id')
        if subject_id and subject_id in self.group_mapping:
            return self._normalize_group_id(int(self.group_mapping[subject_id]))

        pred_group = self._predict_group_id_by_classifier()
        if pred_group is not None:
            pred_group = self._normalize_group_id(pred_group)
            print(f"已通过分类器预测 group_id={pred_group}")
            return pred_group
        return None

    def _assign_group_for_episode(self, episode_idx: int, training: bool = True) -> int:
        if not self.enable_group_adaptation:
            return 1
        if self.subject_group_id is not None:
            return self.subject_group_id

        strategy = str(self.training_config.get('fallback_group_strategy', 'round_robin')).lower()
        if strategy == 'random' and training:
            return int(np.random.randint(1, self.num_groups + 1))
        if strategy == 'fixed_1':
            return 1
        return (episode_idx % self.num_groups) + 1

    def _set_group(self, env_obj, group_id: int):
        if self.enable_group_adaptation and hasattr(env_obj, 'set_group'):
            env_obj.set_group(self._normalize_group_id(group_id))
    
    def _train_episode(self):
        """训练一个episode"""
        state, _ = self.env.reset()
        total_reward = 0
        step_count = 0
        arousal_errors = []
        feedback_actions = []
        
        while True:
            # 选择动作
            action = self.agent.get_action(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 存储经验
            self.agent.store_experience(state, action, reward, next_state, done)
            
            # 更新网络
            if len(self.agent.memory) >= self.iql_config.batch_size:
                if step_count % self.iql_config.update_frequency == 0:
                    losses = self.agent.update()
                    if losses:
                        for key, value in losses.items():
                            self.training_stats['loss_history'][key].append(value)
            
            # 记录统计信息
            total_reward += reward
            step_count += 1
            
            current_arousal = info['arousal']
            target_arousal = self.env_config['arousal_target']
            arousal_errors.append(abs(current_arousal - target_arousal))
            feedback_actions.append(1 if action != 0 else 0)
            
            state = next_state
            
            if done:
                break
        
        # 计算episode统计信息
        episode_stats = {
            'avg_arousal_error': np.mean(arousal_errors),
            'feedback_rate': np.mean(feedback_actions),
            'max_arousal_error': np.max(arousal_errors),
            'arousal_std': np.std(arousal_errors)
        }
        
        return total_reward, step_count, episode_stats
    
    def _evaluate(self):
        """评估当前策略"""
        total_rewards = []
        total_arousal_errors = []
        
        for _ in range(self.training_config['num_eval_episodes']):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            arousal_errors = []
            
            while True:
                # 使用确定性策略
                action = self.agent.get_action(state, training=False)
                next_state, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_reward += reward
                current_arousal = info['arousal']
                target_arousal = self.env_config['arousal_target']
                arousal_errors.append(abs(current_arousal - target_arousal))
                
                if self.training_config['render_eval']:
                    self.eval_env.render()
                
                if terminated or truncated:
                    break
                    
                state = next_state
            
            total_rewards.append(episode_reward)
            total_arousal_errors.extend(arousal_errors)
        
        avg_reward = np.mean(total_rewards)
        avg_arousal_error = np.mean(total_arousal_errors)
        
        return avg_reward, avg_arousal_error
    
    def _save_training_stats(self):
        """保存训练统计信息"""
        stats_path = os.path.join(self.save_dir, 'training_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(self.training_stats, f)
    
    def _generate_training_report(self):
        """生成训练报告"""
        print("\n生成训练报告...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('神经反馈强化学习训练报告', fontsize=16, fontweight='bold')
        
        # 1. Episode奖励
        axes[0, 0].plot(self.training_stats['episode_rewards'], alpha=0.6, color='blue')
        if len(self.training_stats['episode_rewards']) > 100:
            smoothed_rewards = self._smooth_curve(self.training_stats['episode_rewards'], window=100)
            axes[0, 0].plot(smoothed_rewards, color='red', linewidth=2, label='滑动平均(100)')
            axes[0, 0].legend()
        axes[0, 0].set_title('Episode奖励')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('奖励')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Arousal跟踪误差
        axes[0, 1].plot(self.training_stats['arousal_tracking_error'], alpha=0.7, color='green')
        if len(self.training_stats['arousal_tracking_error']) > 50:
            smoothed_errors = self._smooth_curve(self.training_stats['arousal_tracking_error'], window=50)
            axes[0, 1].plot(smoothed_errors, color='darkgreen', linewidth=2, label='滑动平均(50)')
            axes[0, 1].legend()
        axes[0, 1].set_title('Arousal跟踪误差')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('平均绝对误差')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 反馈使用率
        axes[0, 2].plot(self.training_stats['feedback_usage_rate'], alpha=0.7, color='orange')
        if len(self.training_stats['feedback_usage_rate']) > 50:
            smoothed_feedback = self._smooth_curve(self.training_stats['feedback_usage_rate'], window=50)
            axes[0, 2].plot(smoothed_feedback, color='darkorange', linewidth=2, label='滑动平均(50)')
            axes[0, 2].legend()
        axes[0, 2].set_title('反馈使用率')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('反馈比例')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 损失函数
        if self.training_stats['loss_history']['total_loss']:
            for loss_name, loss_values in self.training_stats['loss_history'].items():
                if loss_values and loss_name != 'total_loss':
                    axes[1, 0].plot(loss_values, label=loss_name, alpha=0.7)
            axes[1, 0].set_title('训练损失')
            axes[1, 0].set_xlabel('更新步数')
            axes[1, 0].set_ylabel('损失值')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # 5. 评估性能
        if self.training_stats['evaluation_rewards']:
            eval_episodes = np.arange(0, len(self.training_stats['evaluation_rewards'])) * \
                           self.training_config['evaluation_frequency']
            axes[1, 1].plot(eval_episodes, self.training_stats['evaluation_rewards'], 
                           'o-', color='purple', linewidth=2, markersize=4)
            axes[1, 1].set_title('评估奖励')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('平均奖励')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 评估Arousal误差
        if self.training_stats['evaluation_arousal_errors']:
            eval_episodes = np.arange(0, len(self.training_stats['evaluation_arousal_errors'])) * \
                           self.training_config['evaluation_frequency']
            axes[1, 2].plot(eval_episodes, self.training_stats['evaluation_arousal_errors'], 
                           's-', color='red', linewidth=2, markersize=4)
            axes[1, 2].set_title('评估Arousal误差')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('平均绝对误差')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_report.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 生成文本报告
        self._save_text_report()
    
    def _smooth_curve(self, data, window=50):
        """平滑曲线"""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def _save_text_report(self):
        """保存文本报告"""
        report_path = os.path.join(self.save_dir, 'training_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("神经反馈强化学习训练报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("训练配置:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.training_config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("环境配置:\n")
            f.write("-" * 20 + "\n")
            for key, value in self.env_config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("IQL算法配置:\n")
            f.write("-" * 20 + "\n")
            config_dict = self.iql_config.__dict__
            for key, value in config_dict.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("训练结果:\n")
            f.write("-" * 20 + "\n")
            if self.training_stats['episode_rewards']:
                f.write(f"总训练轮数: {len(self.training_stats['episode_rewards'])}\n")
                f.write(f"最终平均奖励(最近100轮): {np.mean(self.training_stats['episode_rewards'][-100:]):.3f}\n")
                f.write(f"最高单轮奖励: {np.max(self.training_stats['episode_rewards']):.3f}\n")
                f.write(f"最低单轮奖励: {np.min(self.training_stats['episode_rewards']):.3f}\n")
            
            if self.training_stats['evaluation_rewards']:
                f.write(f"最佳评估奖励: {np.max(self.training_stats['evaluation_rewards']):.3f}\n")
                f.write(f"最终评估奖励: {self.training_stats['evaluation_rewards'][-1]:.3f}\n")
            
            if self.training_stats['arousal_tracking_error']:
                f.write(f"最终Arousal跟踪误差: {np.mean(self.training_stats['arousal_tracking_error'][-100:]):.4f}\n")
                f.write(f"最佳Arousal跟踪误差: {np.min(self.training_stats['arousal_tracking_error']):.4f}\n")
            
            if self.training_stats['feedback_usage_rate']:
                f.write(f"平均反馈使用率: {np.mean(self.training_stats['feedback_usage_rate']):.3f}\n")
        
        print(f"训练报告已保存到: {report_path}")

def demonstrate_model(model_path: str, num_episodes: int = 5):
    """演示训练好的模型"""
    print(f"加载模型: {model_path}")
    
    # 创建环境
    env_config = {
        'max_timesteps': 1000,
        'sampling_rate': 250,
        'arousal_target': 0.4,
        'feedback_cost': 0.01,
        'arousal_penalty_weight': 10.0,
        'safety_weight': 5.0,
        'comfort_weight': 2.0
    }
    
    env = NeuralFeedbackEnvironment(env_config)
    
    # 创建智能体并加载模型
    iql_config = IQLConfig(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    agent = ImplicitQLearning(iql_config)
    agent.load_model(model_path)
    
    print(f"开始演示 {num_episodes} 个episodes...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        arousal_history = []
        action_history = []
        
        print(f"\n=== Episode {episode + 1} ===")
        
        while True:
            # 获取动作（非训练模式）
            action = agent.get_action(state, training=False)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 记录信息
            episode_reward += reward
            step_count += 1
            arousal_history.append(info['arousal'])
            action_history.append(FeedbackType(action).name)
            
            # 显示详细信息（每50步）
            if step_count % 50 == 0:
                print(f"步数: {step_count}, Arousal: {info['arousal']:.3f}, "
                      f"动作: {FeedbackType(action).name}, 奖励: {reward:.3f}")
            
            if terminated or truncated:
                break
                
            state = next_state
        
        # Episode结束统计
        avg_arousal = np.mean(arousal_history)
        arousal_error = np.mean([abs(a - env_config['arousal_target']) for a in arousal_history])
        feedback_rate = len([a for a in action_history if a != 'NONE']) / len(action_history)
        
        print(f"Episode {episode + 1} 结果:")
        print(f"  总奖励: {episode_reward:.3f}")
        print(f"  步数: {step_count}")
        print(f"  平均Arousal: {avg_arousal:.3f}")
        print(f"  Arousal跟踪误差: {arousal_error:.4f}")
        print(f"  反馈使用率: {feedback_rate:.3f}")

if __name__ == "__main__":
    # 训练参数
    training_config = {
        'num_episodes': 5000,  # 减少episode数量用于演示
        'max_steps_per_episode': 500,
        'evaluation_frequency': 250,
        'save_frequency': 500,
        'num_eval_episodes': 5,
        'render_eval': False,
        'early_stopping_patience': 1000,
        'target_reward': -3.0,
    }
    
    # 创建训练器
    trainer = NeuralFeedbackTrainer(training_config)
    
    # 开始训练
    best_reward = trainer.train()
    
    print(f"\n训练完成! 最佳奖励: {best_reward:.3f}")
    print(f"模型保存在: {trainer.save_dir}")
    
    # 演示最佳模型
    best_model_path = os.path.join(trainer.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print("\n演示最佳模型...")
        demonstrate_model(best_model_path, num_episodes=3)
    else:
        print("最佳模型文件不存在，使用最终模型演示...")
        final_model_path = os.path.join(trainer.save_dir, 'final_model.pth')
        if os.path.exists(final_model_path):
            demonstrate_model(final_model_path, num_episodes=3) 