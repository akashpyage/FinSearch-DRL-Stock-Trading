#!/usr/bin/env python3
"""
DQN for CartPole-v1 (Inverted Pendulum) — FinSearch Mid-Term, IIT Bombay
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
from typing import List

# ── CONFIG ────────────────────────────────────────────────
class Config:
    ENV_NAME      = "CartPole-v1"
    STATE_DIM     = 4
    ACTION_DIM    = 2
    HIDDEN        = [128, 128]
    EPISODES      = 500
    BATCH_SIZE    = 64
    LR            = 1e-3
    GAMMA         = 0.99
    EPS_START     = 1.0
    EPS_END       = 0.01
    EPS_DECAY     = 0.995
    MEM_SIZE      = 10000
    TARGET_SYNC   = 10
    SEED          = 42

# ── REPLAY BUFFER ─────────────────────────────────────────
Exp = namedtuple("Exp", ["s", "a", "r", "ns", "d"])

class ReplayBuffer:
    def __init__(self, cap=Config.MEM_SIZE):
        self.buf = deque(maxlen=cap)
    def push(self, s, a, r, ns, d):
        self.buf.append(Exp(s, a, r, ns, d))
    def sample(self, n):
        return random.sample(self.buf, min(n, len(self.buf)))
    def __len__(self):
        return len(self.buf)

# ── Q-NETWORK ─────────────────────────────────────────────
class QNet(nn.Module):
    def __init__(self, sd, ad, hd):
        super().__init__()
        layers = []
        inp = sd
        for h in hd:
            layers += [nn.Linear(inp, h), nn.ReLU()]
            inp = h
        layers.append(nn.Linear(inp, ad))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ── DQN AGENT ─────────────────────────────────────────────
class DQNAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = QNet(cfg.STATE_DIM, cfg.ACTION_DIM, cfg.HIDDEN).to(self.dev)
        self.tgt_net = QNet(cfg.STATE_DIM, cfg.ACTION_DIM, cfg.HIDDEN).to(self.dev)
        self.tgt_net.load_state_dict(self.q_net.state_dict())
        self.tgt_net.eval()
        self.opt = optim.Adam(self.q_net.parameters(), lr=cfg.LR)
        self.mem = ReplayBuffer(cfg.MEM_SIZE)
        self.eps = cfg.EPS_START
        self.rewards = []

    def act(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.eps:
            return random.randrange(self.cfg.ACTION_DIM)
        s = torch.FloatTensor(state).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            return self.q_net(s).argmax(1).item()

    def learn(self):
        if len(self.mem) < self.cfg.BATCH_SIZE:
            return 0.0
        batch = self.mem.sample(self.cfg.BATCH_SIZE)
        states  = torch.FloatTensor(np.array([e.s  for e in batch])).to(self.dev)
        actions = torch.LongTensor(np.array([e.a  for e in batch])).unsqueeze(1).to(self.dev)
        rewards = torch.FloatTensor(np.array([e.r  for e in batch])).unsqueeze(1).to(self.dev)
        nstates = torch.FloatTensor(np.array([e.ns for e in batch])).to(self.dev)
        dones   = torch.FloatTensor(np.array([e.d  for e in batch])).unsqueeze(1).to(self.dev)

        cur_q = self.q_net(states).gather(1, actions)
        with torch.no_grad():
            max_nq = self.tgt_net(nstates).max(1, keepdim=True)[0]
            tgt_q = rewards + (1 - dones) * self.cfg.GAMMA * max_nq
        loss = nn.MSELoss()(cur_q, tgt_q)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def train(self, env):
        cfg = self.cfg
        for ep in range(1, cfg.EPISODES + 1):
            s, _ = env.reset()
            ep_r, done = 0, False
            while not done:
                a = self.act(s)
                ns, r, term, trunc, _ = env.step(a)
                done = term or trunc
                self.mem.push(s, a, r, ns, done)
                self.learn()
                s = ns
                ep_r += r
            self.rewards.append(ep_r)
            self.eps = max(cfg.EPS_END, self.eps * cfg.EPS_DECAY)
            if ep % cfg.TARGET_SYNC == 0:
                self.tgt_net.load_state_dict(self.q_net.state_dict())
            if ep % 50 == 0:
                avg = np.mean(self.rewards[-50:])
                print(f"Ep {ep:4d} | Avg50: {avg:7.1f} | eps: {self.eps:.4f}")
        return self.rewards

    def evaluate(self, env, n=10):
        total = 0
        for _ in range(n):
            s, _ = env.reset()
            done = False
            while not done:
                a = self.act(s, eval_mode=True)
                s, r, term, trunc, _ = env.step(a)
                done = term or trunc
                total += r
        return total / n

# ── MAIN ──────────────────────────────────────────────────
def main():
    print("=" * 55)
    print(" DQN — Inverted Pendulum (CartPole-v1)")
    print(" FinSearch Mid-Term | Finance Club, IIT Bombay")
    print("=" * 55)

    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)

    env = gym.make(Config.ENV_NAME)
    print(f"\nState dim:  {Config.STATE_DIM}  [cart pos, cart vel, pole angle, pole ang vel]")
    print(f"Action dim: {Config.ACTION_DIM}  [0: push left, 1: push right]")
    print(f"Goal: keep pole upright for 500 steps (+1 per step)")

    agent = DQNAgent(Config)
    print(f"\nTraining {Config.EPISODES} episodes...\n")

    rewards = agent.train(env)

    print(f"\nDone! Final 50-ep avg: {np.mean(rewards[-50:]):.1f}")
    eval_r = agent.evaluate(env, 20)
    print(f"Evaluation (20 eps):     {eval_r:.1f}")

    # Plot
    plt.figure(figsize=(12, 4.5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.35, color='steelblue')
    ma = np.convolve(rewards, np.ones(30)/30, mode='valid')
    plt.plot(range(29, len(rewards)), ma, 'darkred', linewidth=2, label='30-ep MA')
    plt.axhline(y=475, color='green', linestyle='--', label='Solved threshold')
    plt.xlabel("Episode"); plt.ylabel("Reward")
    plt.title("DQN Training — CartPole-v1")
    plt.legend(); plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(rewards[-200:], bins=20, color='steelblue', edgecolor='white')
    plt.axvline(x=500, color='darkred', linestyle='--', label='Max (500)')
    plt.xlabel("Reward"); plt.ylabel("Count")
    plt.title("Final 200 Episodes Distribution")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("dqn_cartpole_results.png", dpi=150, bbox_inches="tight")
    print("Plot saved → dqn_cartpole_results.png")

    # Sample Q-values
    print("\n" + "=" * 55)
    print(" Sample Q-values (learned policy)")
    print("=" * 55)
    test_state = np.array([0.0, 0.0, 0.05, 0.0])
    with torch.no_grad():
        qv = agent.q_net(torch.FloatTensor(test_state).unsqueeze(0).to(agent.dev))
    print(f" State: [pos=0, vel=0, angle=0.05 rad, ang_vel=0]")
    print(f" Q(push left)  = {qv[0,0].item():.3f}")
    print(f" Q(push right) = {qv[0,1].item():.3f}")
    print(f" → Agent pushes {'RIGHT ↵' if qv.argmax().item()==1 else 'LEFT ↵'} (counteracts tilt)")

    env.close()

if __name__ == "__main__":
    main()
