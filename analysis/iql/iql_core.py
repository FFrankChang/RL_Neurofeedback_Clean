#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal Implicit Q-Learning components for recorded driving data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class IQLConfig:
    state_dim: int
    action_dim: int = 3
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    expectile: float = 0.8
    temperature: float = 3.0
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TransitionDataset(Dataset):
    def __init__(self, transitions: list[Transition]):
        if not transitions:
            raise ValueError("No transitions were provided.")
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int):
        item = self.transitions[idx]
        return (
            torch.tensor(item.state, dtype=torch.float32),
            torch.tensor(item.action, dtype=torch.long),
            torch.tensor(item.reward, dtype=torch.float32),
            torch.tensor(item.next_state, dtype=torch.float32),
            torch.tensor(item.done, dtype=torch.float32),
        )


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IQLTrainer:
    def __init__(self, config: IQLConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.q_net = MLP(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.v_net = MLP(config.state_dim, 1, config.hidden_dim).to(self.device)
        self.policy_net = MLP(config.state_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.q_optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.learning_rate)
        self.v_optimizer = torch.optim.AdamW(self.v_net.parameters(), lr=config.learning_rate)
        self.policy_optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=config.learning_rate)

    @staticmethod
    def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * diff.pow(2)

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        totals = {"q_loss": 0.0, "v_loss": 0.0, "policy_loss": 0.0}
        batches = 0

        for states, actions, rewards, next_states, dones in loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            with torch.no_grad():
                next_v = self.v_net(next_states).squeeze(-1)
                q_target = rewards + self.config.gamma * (1.0 - dones) * next_v

            q_values = self.q_net(states)
            selected_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            q_loss = F.mse_loss(selected_q, q_target)

            with torch.no_grad():
                q_for_value = self.q_net(states).max(dim=1).values
            v_values = self.v_net(states).squeeze(1)
            v_loss = self.expectile_loss(q_for_value - v_values, self.config.expectile).mean()

            with torch.no_grad():
                advantages = selected_q - v_values
                weights = torch.exp(advantages / self.config.temperature).clamp(max=100.0)
            logits = self.policy_net(states)
            log_probs = F.log_softmax(logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
            policy_loss = -(weights * log_probs).mean()

            self.q_optimizer.zero_grad()
            q_loss.backward()
            self.q_optimizer.step()

            self.v_optimizer.zero_grad()
            v_loss.backward()
            self.v_optimizer.step()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            totals["q_loss"] += float(q_loss.item())
            totals["v_loss"] += float(v_loss.item())
            totals["policy_loss"] += float(policy_loss.item())
            batches += 1

        return {key: value / max(1, batches) for key, value in totals.items()}

    def fit(self, transitions: list[Transition]) -> list[dict[str, float]]:
        dataset = TransitionDataset(transitions)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False)
        history = []
        for epoch in range(self.config.epochs):
            losses = self.train_epoch(loader)
            losses["epoch"] = epoch + 1
            history.append(losses)
        return history

    def save(self, path: str) -> None:
        torch.save({
            "config": self.config,
            "q_net": self.q_net.state_dict(),
            "v_net": self.v_net.state_dict(),
            "policy_net": self.policy_net.state_dict(),
        }, path)
