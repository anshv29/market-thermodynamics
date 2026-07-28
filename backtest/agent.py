"""
SAC (Soft Actor-Critic) Agent
==============================
Trains a portfolio management agent using SAC.
SAC is ideal for continuous action spaces and has
built-in entropy maximization for diverse strategies.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque
import random


# ── Neural Network Components ─────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Policy network — maps state to portfolio weights.
    Outputs mean and log_std of a Gaussian distribution over actions.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.LOG_STD_MAX  =  2
        self.LOG_STD_MIN  = -5

    def forward(self, state):
        h       = self.net(state)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x    = dist.rsample()

        # Squash to [0, 1] using sigmoid (portfolio weights must be positive)
        action   = torch.sigmoid(x)
        log_prob = dist.log_prob(x)

        # Correction for sigmoid squashing
        log_prob -= torch.log(action * (1 - action) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob, torch.sigmoid(mean)


class Critic(nn.Module):
    """
    Q-network — estimates Q(state, action).
    We use two critics (twin critics) to reduce overestimation bias.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        def make_q():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_value(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)


class ReplayBuffer:
    """Experience replay buffer — stores past transitions for training."""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class SACAgent:
    """
    Soft Actor-Critic agent.

    Key idea: maximize expected reward + entropy of policy.
    The entropy term encourages exploration and diverse strategies.
    """

    def __init__(self,
                 state_dim:   int,
                 action_dim:  int,
                 hidden_dim:  int   = 256,
                 lr:          float = 3e-4,
                 gamma:       float = 0.99,
                 tau:         float = 0.005,
                 alpha:       float = 0.2,
                 buffer_size: int   = 100_000,
                 batch_size:  int   = 256):

        self.gamma      = gamma
        self.tau        = tau
        self.alpha      = alpha
        self.batch_size = batch_size

        # Networks
        self.actor    = Actor(state_dim, action_dim, hidden_dim)
        self.critic   = Critic(state_dim, action_dim, hidden_dim)
        self.critic_t = Critic(state_dim, action_dim, hidden_dim)  # target

        # Copy weights to target
        self.critic_t.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt  = Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=lr)

        # Auto-tuning entropy coefficient
        self.target_entropy = -action_dim
        self.log_alpha      = torch.zeros(1, requires_grad=True)
        self.alpha_opt      = Adam([self.log_alpha], lr=lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                _, _, action = self.actor.sample(state_t)
            else:
                action, _, _ = self.actor.sample(state_t)
        return action.squeeze(0).numpy()

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.sample(next_states)
            q1_next, q2_next = self.critic_t(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * (1 - dones) * q_next

        # Critic update
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor update
        actions_new, log_pi, _ = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_new).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # Alpha update (entropy tuning)
        alpha_loss = -(
            self.log_alpha * (log_pi + self.target_entropy).detach()
        ).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        for param, target_param in zip(
            self.critic.parameters(), self.critic_t.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha,
        }

    def save(self, path: str):
        torch.save({
            "actor":    self.actor.state_dict(),
            "critic":   self.critic.state_dict(),
            "critic_t": self.critic_t.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_t.load_state_dict(ckpt["critic_t"])