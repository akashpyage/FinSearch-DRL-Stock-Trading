# FinSearch-DRL-Stock-Trading

> **FinSearch — Finance Club, IIT Bombay (June 20 – July 24, 2026)**

Deep Reinforcement Learning framework for optimising stock trading strategies to maximise risk-adjusted investment returns. DRL agents (DQN, PPO, DDPG) trained on NIFTY 50 data with a custom Gymnasium trading environment.

---

## Team

| Name | Roll No. | Checkpoints |
|:---|:---|:---|
| Akash Pyage | 24B2411 | CP 1 (RL/DL Foundations) & CP 4 (Data Collection) |
| Abhishek Gowda A L | 24B4512 | CP 2 (Algorithms) & CP 3 (CartPole DQN Implementation) |
| Yash Dudhade | 24B0066 | CP 5 (Training & Benchmarks) & CP 6 (Write-up & Findings) |

---

## Project Structure

```
FinSearch-DRL-Stock-Trading/
├── dqn_cartpole.py            # DQN implementation for CartPole-v1 (Checkpoint 3)
├── dqn_cartpole_results.png   # Training plots (learning curve + reward distribution)
├── FinSearch_Midterm_Report.md # Full mid-term report (July 6, 2026)
├── trading_env.py             # Custom Gymnasium trading environment (Checkpoint 4–5)
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Features

- **DQN from scratch** — Full implementation with experience replay, target network, and ε-greedy exploration (PyTorch)
- **Custom trading environment** — Gymnasium-compatible, NIFTY 50 data (2010–2019), 22-feature state representation
- **Risk-adjusted reward functions** — Log return + differential Sharpe ratio (multi-objective)
- **Benchmarking** — Comparison against ARIMA, LSTM, SMA Crossover, and Buy-and-Hold
- **Walk-forward validation** — Rolling window training to prevent look-ahead bias
- **7 evaluation metrics** — Cumulative Return, Annualised Return, Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio, Win Rate

---

## Tech Stack

| Library | Purpose |
|:---|:---|
| `torch` | Neural network implementation (DQN, PPO, DDPG) |
| `gymnasium` | RL environment interface (CartPole-v1 + custom trading env) |
| `stable-baselines3` | Production RL algorithms |
| `FinRL` | Standardised DRL framework for finance |
| `yfinance` | NIFTY 50 OHLCV data fetching |
| `numpy`, `pandas` | Data manipulation and preprocessing |
| `matplotlib` | Training visualisation |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/FinSearch-DRL-Stock-Trading.git
cd FinSearch-DRL-Stock-Trading

# Install dependencies
pip install -r requirements.txt

# Run DQN on CartPole-v1 (Checkpoint 3)
python dqn_cartpole.py
```

---

## Checkpoint Progress

| CP | Checkpoint | Status |
|:---:|:---|:---:|
| 1 | Introduction to RL and DL | ✅ Complete |
| 2 | Useful Algorithms (DQN, DDPG, PPO, A2C, TD3, SAC) | ✅ Complete |
| 3 | Inverted Pendulum — DQN Implementation | ✅ Complete |
| 4 | Data Collection & Preprocessing (NIFTY 50) | 🔄 In Progress |
| 5 | Training, Testing, Tuning & Comparison | 🔄 In Progress |
| 6 | Code, Write-up & Findings | 🔄 In Progress |

---

## DQN CartPole Results

| Metric | Value |
|:---|:---|
| Episodes trained | 500 |
| Initial avg reward (random) | ~21 |
| Final avg reward | ~117 |
| Evaluation score (ε=0) | ~105 |
| Q-Network | 4 → 128 → 128 → 2 |

The agent successfully learns to balance the pole and internalises the correct physics — verified through Q-value analysis.

---

## References

1. Mnih et al. (2015) — Human-level control through deep RL. *Nature*.
2. Sutton & Barto (2018) — *Reinforcement Learning: An Introduction*. MIT Press.
3. Schulman et al. (2017) — Proximal Policy Optimization. *arXiv:1707.06347*.
4. Lillicrap et al. (2015) — DDPG. *arXiv:1509.02971*.
5. Choudhary et al. (2025) — Risk-Adjusted DRL for Portfolio Optimization (IIT Mandi). *Int J Comp Int Sys*.
6. FinRL Contest (2025) — Benchmarking Financial RL. *arXiv:2504.02281*.

---

## Licence

This project is part of the FinSearch programme by Finance Club, IIT Bombay. For academic use only.
