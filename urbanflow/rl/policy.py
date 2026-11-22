"""Parameter-sharing actor and critic."""
from torch import nn

from .env import SCHEMA


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(len(SCHEMA), 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 2)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        hidden = self.body(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)
