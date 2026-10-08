"""EP51 - Deep Q-Network (DQN) on CartPole: a neural network replaces the Q-table.
    python ep51_dqn.py        # train 50,000 steps (about 1-2 minutes on a CPU), evaluate, save a curve
pip install torch gymnasium numpy matplotlib
"""
import random
from collections import deque
import gymnasium as gym
import numpy as np
import torch
from torch import nn
# ── 1. CONFIG ─────────────────────────────────────────────────────────────
SEED = 0
STEPS = 50_000                       # environment steps to train for
GAMMA = 0.99                         # how much future reward counts
LR = 5e-4                            # Adam learning rate
BATCH = 64                           # transitions per learning step
BUFFER_SIZE = 10_000                 # experience replay: remember the last 10,000 steps
WARMUP = 1_000                       # collect experience before learning, so batches are not tiny
TARGET_SYNC = 250                    # copy online -> target network every 250 steps
EPS0, EPS_MIN, EPS_STEPS = 1.0, 0.05, 10_000   # epsilon: 100% random -> 5% random
HIDDEN = 128

# ── 2. NETWORK + MEMORY: a neural net replaces the table ──────────────────
def make_net(n_in=4, n_out=2):
    return nn.Sequential(nn.Linear(n_in, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, n_out))

def tensor(x, dtype=torch.float32):
    return torch.tensor(np.array(x), dtype=dtype)

def sample(memory):                  # a random batch breaks the order of play
    s, a, r, s2, d = zip(*random.sample(memory, BATCH))
    return tensor(s), tensor(a, torch.int64), tensor(r), tensor(s2), tensor(d)

# ── 3. AGENT: epsilon-greedy action, the DQN learning step ────────────────
def act(net, obs, eps):
    if random.random() < eps:
        return random.randrange(2)                      # explore
    with torch.no_grad():
        return int(net(tensor(obs)).argmax())           # exploit

def learn(online, target, opt, memory):
    s, a, r, s2, d = sample(memory)
    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)  # Q(s, a) from the online net
    with torch.no_grad():                               # the target comes from the FROZEN net
        y = r + GAMMA * target(s2).max(1).values * (1 - d)
    loss = nn.functional.smooth_l1_loss(q, y)
    opt.zero_grad(); loss.backward(); opt.step()

# ── 4. TRAIN: play, remember, replay, sync the target network ─────────────
def train(env, online, target, seed):
    opt = torch.optim.Adam(online.parameters(), lr=LR)
    memory, returns, total = deque(maxlen=BUFFER_SIZE), [], 0.0
    obs, _ = env.reset(seed=seed)
    for t in range(STEPS):
        eps = max(EPS_MIN, EPS0 - (EPS0 - EPS_MIN) * t / EPS_STEPS)
        a = act(online, obs, eps)
        obs2, r, terminated, truncated, _ = env.step(a)
        memory.append((obs, a, r, obs2, float(terminated)))
        obs, total = obs2, total + r
        if terminated or truncated:
            returns.append(total)
            total = 0.0
            obs, _ = env.reset()
        if t >= WARMUP:
            learn(online, target, opt, memory)
        if t % TARGET_SYNC == 0:
            target.load_state_dict(online.state_dict())
        if (t + 1) % 5000 == 0:
            print(f"step {t + 1:6d}  avg return (last 10): {np.mean(returns[-10:]):6.1f}  epsilon {eps:.2f}")
    return returns

# ── 5. EVALUATE: greedy play, no exploration; run() ties it all together ──
def evaluate(env, net, n=100):
    scores = []
    for k in range(n):
        obs, _ = env.reset(seed=10_000 + k)
        done, total = False, 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(act(net, obs, 0.0))
            total, done = total + r, terminated or truncated
        scores.append(total)
    return float(np.mean(scores))

def run(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    env, online, target = gym.make("CartPole-v1"), make_net(), make_net()
    return train(env, online, target, seed), evaluate(env, online)

if __name__ == "__main__":
    curve, score = run()
    print(f"greedy evaluation, 100 episodes: {score:.1f} / 500")
    try:
        import matplotlib.pyplot as plt
        plt.plot(np.convolve(curve, np.ones(20) / 20, mode="valid"))
        plt.savefig("ep51_learning_curve.png", dpi=120)
    except ImportError:
        print("matplotlib not installed - skipping the plot")
