"""
IQL 训练过程可视化
=================

本脚本读取由 `iql_v2/train_neural_feedback.py`
保存的 `training_stats.pkl`，按论文图4-5的子图布局绘制
训练收敛曲线（奖励/Arousal误差/反馈率/Q-V-Policy损失/评估曲线）。

使用方法：
    1) 先运行训练得到统计文件：
         cd iql_v2 && python train_neural_feedback.py
       训练会在 models/neural_feedback_<时间戳>/ 下保存 training_stats.pkl
    2) 用本脚本绘图：
         python iql_demo/plot_iql_training.py --stats <path>/training_stats.pkl
"""
import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _smooth(data, window):
    data = np.asarray(data, dtype=float)
    if len(data) < window or window <= 1:
        return None
    return np.convolve(data, np.ones(window) / window, mode='valid')


def load_stats(stats_path: str) -> dict:
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"未找到训练统计文件: {stats_path}\n"
            f"请先运行训练脚本 (iql_v2/train_neural_feedback.py) 生成 training_stats.pkl。"
        )
    with open(stats_path, 'rb') as f:
        return pickle.load(f)


def plot_training(stats: dict, out_dir: str, eval_frequency: int = 500):
    os.makedirs(out_dir, exist_ok=True)
    saved = []

    def _save(fig, name):
        path = os.path.join(out_dir, name)
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        saved.append(path)

    # 01 奖励曲线
    rewards = stats.get('episode_rewards', [])
    if rewards:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rewards, alpha=0.25, color='steelblue', linewidth=0.5)
        sm = _smooth(rewards, 50)
        if sm is not None:
            ax.plot(range(49, len(rewards)), sm, color='darkblue',
                    linewidth=2.5, label='滑动平均(50)')
            ax.legend(loc='lower right', fontsize=11)
        ax.set_xlabel('Episode'); ax.set_ylabel('累积奖励')
        ax.set_title('训练奖励曲线'); ax.grid(True, alpha=0.3)
        _save(fig, '01_reward_curve.png')

    # 02 Arousal 跟踪误差
    errs = stats.get('arousal_tracking_error', [])
    if errs:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(errs, alpha=0.25, color='seagreen', linewidth=0.5)
        sm = _smooth(errs, 50)
        if sm is not None:
            ax.plot(range(49, len(errs)), sm, color='darkgreen',
                    linewidth=2.5, label='滑动平均(50)')
            ax.legend(loc='upper right', fontsize=11)
        ax.set_xlabel('Episode'); ax.set_ylabel('平均Arousal误差')
        ax.set_title('Arousal跟踪误差'); ax.grid(True, alpha=0.3)
        _save(fig, '02_arousal_error.png')

    # 04/05/06 损失曲线
    loss_history = stats.get('loss_history', {})
    loss_specs = [
        ('q_loss', '04_q_loss.png', 'Q网络损失', 'darkred'),
        ('v_loss', '05_v_loss.png', '值网络损失', 'indigo'),
        ('policy_loss', '06_policy_loss.png', '策略网络损失', 'darkcyan'),
    ]
    for key, fname, title, color in loss_specs:
        vals = loss_history.get(key, [])
        if vals:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(vals, alpha=0.3, color=color, linewidth=0.5)
            sm = _smooth(vals, 80)
            if sm is not None:
                ax.plot(range(79, len(vals)), sm, color=color,
                        linewidth=2.5, label='滑动平均(80)')
                ax.legend(loc='upper right', fontsize=11)
            ax.set_xlabel('更新步数'); ax.set_ylabel(key)
            ax.set_title(title); ax.grid(True, alpha=0.3)
            _save(fig, fname)

    # 07 评估奖励
    eval_rewards = stats.get('evaluation_rewards', [])
    if eval_rewards:
        eval_x = np.arange(1, len(eval_rewards) + 1) * eval_frequency
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eval_x, eval_rewards, 'o-', color='royalblue',
                linewidth=2.5, markersize=8)
        ax.set_xlabel('Episode'); ax.set_ylabel('评估平均奖励')
        ax.set_title('评估性能趋势'); ax.grid(True, alpha=0.3)
        _save(fig, '07_eval_rewards.png')

    # 08 反馈使用率
    fb = stats.get('feedback_usage_rate', [])
    if fb:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fb, alpha=0.25, color='orange', linewidth=0.5)
        sm = _smooth(fb, 50)
        if sm is not None:
            ax.plot(range(49, len(fb)), sm, color='darkorange',
                    linewidth=2.5, label='滑动平均(50)')
            ax.legend(loc='upper right', fontsize=11)
        ax.set_xlabel('Episode'); ax.set_ylabel('反馈使用比例')
        ax.set_title('反馈效率优化'); ax.grid(True, alpha=0.3); ax.set_ylim([0, 1])
        _save(fig, '08_feedback_rate.png')

    print(f"已生成 {len(saved)} 张训练曲线图，保存到: {out_dir}")
    for p in saved:
        print(f"   - {os.path.basename(p)}")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="绘制IQL训练曲线（论文图4-5）"
    )
    parser.add_argument(
        "--stats", type=str, required=True,
        help="training_stats.pkl 路径"
    )
    parser.add_argument(
        "--eval-frequency", type=int, default=500,
        help="评估间隔（应与训练时 evaluation_frequency 一致，用于评估曲线横轴）"
    )
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    stats = load_stats(args.stats)
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.stats)), "training_plots"
    )
    plot_training(stats, out_dir, eval_frequency=args.eval_frequency)


if __name__ == "__main__":
    main()
