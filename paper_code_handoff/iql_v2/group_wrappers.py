import numpy as np
import gymnasium as gym
from typing import Dict, Optional, List

from .neural_feedback_env import NeuralFeedbackEnvironment, FeedbackType, EEGMetrics


def default_group_reward_profiles(num_groups: int = 4) -> Dict[int, Dict[str, float]]:
    """Return simple per-group reward weight profiles.

    Keys mirror the base environment's reward terms. Values are multipliers applied
    to the corresponding weights in the wrapped environment config.
    """
    profiles: Dict[int, Dict[str, float]] = {
        1: {
            'arousal_penalty_weight': 1.2,
            'safety_weight': 1.0,
            'comfort_weight': 0.8,
            'performance_bonus_scale': 1.0,
        },
        2: {
            'arousal_penalty_weight': 1.0,
            'safety_weight': 1.2,
            'comfort_weight': 1.0,
            'performance_bonus_scale': 1.0,
        },
        3: {
            'arousal_penalty_weight': 0.9,
            'safety_weight': 1.0,
            'comfort_weight': 1.2,
            'performance_bonus_scale': 0.8,
        },
        4: {
            'arousal_penalty_weight': 1.1,
            'safety_weight': 0.9,
            'comfort_weight': 1.0,
            'performance_bonus_scale': 1.2,
        },
    }
    # In case num_groups != 4, repeat or trim
    keys = list(profiles.keys())
    out: Dict[int, Dict[str, float]] = {}
    for i in range(1, num_groups + 1):
        out[i] = profiles[keys[(i - 1) % len(keys)]]
    return out


class GroupRewardWrapper(gym.Wrapper):
    """Wrapper that personalizes the reward using group-specific weight multipliers.

    The underlying environment computes its usual reward, but we recompute a
    personalized reward using the same components with per-group multipliers.
    """

    def __init__(self, env: NeuralFeedbackEnvironment,
                 group_profiles: Optional[Dict[int, Dict[str, float]]] = None,
                 num_groups: int = 4):
        super().__init__(env)
        self.num_groups = num_groups
        self.group_profiles = group_profiles or default_group_reward_profiles(num_groups)
        self.current_group_id: int = 1

    def set_group(self, group_id: int) -> None:
        self.current_group_id = int(group_id)

    def step(self, action: int):
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Compute personalized reward using current context in info
        eeg_metrics: EEGMetrics = info['eeg_metrics']
        feedback_type: FeedbackType = info['feedback_type']
        current_arousal: float = info['arousal']

        personalized_reward = self._compute_group_reward(current_arousal, feedback_type, eeg_metrics)

        info = dict(info)
        info['group_id'] = self.current_group_id
        info['base_reward'] = base_reward
        info['personalized_reward'] = personalized_reward

        return obs, personalized_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def _compute_group_reward(self, current_arousal: float, feedback_type: FeedbackType,
                              eeg_metrics: EEGMetrics) -> float:
        cfg = self.env.config
        prof = self.group_profiles.get(self.current_group_id, {})

        # Base weights
        arousal_w = cfg.get('arousal_penalty_weight', 10.0) * prof.get('arousal_penalty_weight', 1.0)
        safety_w = cfg.get('safety_weight', 5.0) * prof.get('safety_weight', 1.0)
        comfort_w = cfg.get('comfort_weight', 2.0) * prof.get('comfort_weight', 1.0)
        perf_scale = prof.get('performance_bonus_scale', 1.0)

        target_arousal = cfg.get('arousal_target', 0.5)
        arousal_error = abs(current_arousal - target_arousal)
        arousal_reward = -arousal_w * (arousal_error ** 2)

        safety_threshold = 0.2
        if current_arousal < safety_threshold:
            safety_penalty = -safety_w * ((safety_threshold - current_arousal) ** 2)
        else:
            safety_penalty = 0.0

        comfort_penalty = 0.0
        if feedback_type != FeedbackType.NONE:
            comfort_penalty = -comfort_w * cfg.get('feedback_cost', 0.01)
            recent_feedbacks = sum([1 for f in self.env.feedback_history[-3:] if f != 0])
            if recent_feedbacks >= 2:
                comfort_penalty *= 2.0

        performance_reward = 0.0
        if self.env.driving_context.name == 'EMERGENCY_SITUATION':
            if current_arousal > 0.7:
                performance_reward = 2.0 * perf_scale
        elif current_arousal > 0.3 and current_arousal < 0.6:
            performance_reward = 1.0 * perf_scale

        signal_quality_reward = -eeg_metrics.artifact_level * 0.5

        total = arousal_reward + safety_penalty + comfort_penalty + performance_reward + signal_quality_reward
        return float(total)


class GroupObservationAugment(gym.ObservationWrapper):
    """Append a group vector (one-hot or soft probs) to the observation.

    Use set_group_vector(...) before each episode to specify the current user's
    group context.
    """

    def __init__(self, env: gym.Env, num_groups: int = 4):
        super().__init__(env)
        self.num_groups = num_groups
        self.group_vector = np.zeros(self.num_groups, dtype=np.float32)

        low = np.concatenate([self.observation_space.low, np.zeros(self.num_groups, dtype=np.float32)])
        high = np.concatenate([self.observation_space.high, np.ones(self.num_groups, dtype=np.float32)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def set_group_id(self, group_id: int):
        vec = np.zeros(self.num_groups, dtype=np.float32)
        idx = int(group_id) - 1
        idx = max(0, min(idx, self.num_groups - 1))
        vec[idx] = 1.0
        self.group_vector = vec

    def set_group_vector(self, probs: List[float]):
        arr = np.array(probs, dtype=np.float32)
        if arr.shape[0] != self.num_groups:
            raise ValueError(f"group prob length {arr.shape[0]} != num_groups {self.num_groups}")
        # Normalize to sum 1
        s = float(arr.sum())
        self.group_vector = arr / s if s > 0 else np.ones(self.num_groups, dtype=np.float32) / self.num_groups

    def observation(self, observation):
        return np.concatenate([observation.astype(np.float32), self.group_vector], axis=0)


