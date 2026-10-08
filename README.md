# EP51 — Deep Q-Networks (DQN) Explained  
### When Reinforcement Learning Meets Deep Learning

**Deep Learning Series · Module 10 · Episode 51 of 72**

A Q-table cannot handle pixels or huge continuous states.  
This episode replaces the table with a neural network — the Deep Q-Network — and builds one from scratch in pure PyTorch.

---

## What you will learn

- Why a table breaks (a 64×64 image has ~10⁹⁸⁶⁴ possible states)
- The Q-network and the DQN learning step (Bellman target + Huber loss)
- Experience replay (breaks correlation + reuses transitions)
- The target network (keeps the target from moving)
- Full training loop with epsilon schedule
- Honest multi-seed evaluation on CartPole
- What DQN actually achieved on 49 Atari games (Nature 2015)

---

## Quick Start

```bash
pip install torch gymnasium numpy matplotlib
python ep51_dqn.py
