'''
validation_dqn_cartpole.py

Edge case and validation testing for cartpole-v1.
'''

import sys, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.cartpole import cartpole_dqn as cd  # CartpoleDQN, ReplayBuffer, dqn_target
# sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.validations.val_common import check, section, finish

torch.set_num_threads(1)
RUN_ABLATIONS = True # True runs the retraining ablations at the bottom
GAMMA, SEEDS, STEPS = 0.95, (0, 1, 2), 100_000
CEIL = 1 / (1 - GAMMA)
FIG = Path("figures")
FIG.mkdir(exist_ok=True)
MODEL_PT = Path("src/cartpole/output/cartpole_dqn.pt") # model path

class SignFlip(gym.RewardWrapper):
    """Negates reward"""
    def reward(self, r): 
        return -r

def ma(x, w=100): 
    return np.convolve(x, np.ones(w) / w, mode="valid")

def trained_net():
   net = cd.CartpoleDQN()
   net.load_state_dict(torch.load(MODEL_PT, map_location="cpu"))
   return net.eval()

def collapse_fraction(rewards):
    """How unstable is training"""
    m = ma(rewards)
    peak = np.maximum.accumulate(m)
    live = peak > 100
    return float((m[live] < 0.5 * peak[live]).mean()) if live.any() else 0.0

def heldout_states(n=500):
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(123)
    env.reset(seed=123)
    out = []
    while len(out) < n:
        s, _, t, tr, _ = env.step(int(rng.integers(2)))
        out.append(s)
        if t or tr: 
            env.reset()
    return torch.as_tensor(np.array(out), dtype=torch.float32)

### configurable training
def train_cp(seed, steps=STEPS, gamma=GAMMA, double=True, done_on_trunc=False, buf_cap=1_000, sync_every=1000, flip=False, batch=64, lr=1e-3, eps_decay=10_000, log_every=500):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = gym.make("CartPole-v1")
    
    if flip: 
        env = SignFlip(env)
    env.action_space.seed(seed)
    online, target = cd.CartpoleDQN(), cd.CartpoleDQN()
    target.load_state_dict(online.state_dict())

    buf = cd.ReplayBuffer(buf_cap)
    cd.initialize_buffer(env, buf, buf_cap)
    
    for p in target.parameters(): 
        p.requires_grad_(False)
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    held = heldout_states()
    
    rewards, log = [], []  # log rows: step, mean max Q on held-out states, terminal fraction in buffer
    s, _ = env.reset(seed=seed)
    ep_r = 0.0
    for step in range(steps):
        eps = max(0.05, 1 - 0.95 * step / eps_decay)
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad(): 
                a = online(torch.as_tensor(s, dtype=torch.float32)).argmax().item()
        s2, r, term, trunc, _ = env.step(a)
        buf.push(s, a, s2, r, float(term or (done_on_trunc and trunc)))
        ep_r += r
        if term or trunc: 
            rewards.append(abs(ep_r))
            ep_r = 0.0
            s, _ = env.reset()
        else: 
            s = s2
            
        if len(buf) >= 1000:
            S, A, S2, R, D = buf.sample(batch)
            S, S2, R, D = [torch.as_tensor(x, dtype=torch.float32) for x in (S, S2, R, D)]
            A = torch.as_tensor(A, dtype=torch.long)
            y = cd.dqn_target(online, target, S2, R, D, gamma, double)
            loss = F.smooth_l1_loss(online(S).gather(1, A.unsqueeze(1)).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        if step % sync_every == 0: 
            target.load_state_dict(online.state_dict())
        if step % log_every == 0:
            with torch.no_grad(): 
                hq = online(held).max(1).values.mean().item()
            log.append((step, hq, np.mean([t[4] for t in buf.buffer])))
        if step % 5000 == 0:
             print(f"\t[train_cp seed={seed}] step {step}/{steps}", flush=True)
    return dict(rewards=np.array(rewards), log=np.array(log), online=online)

CACHE = {}
def get(name, **kw):
    if name not in CACHE: 
        CACHE[name] = [train_cp(s, **kw) for s in SEEDS]
    return CACHE[name]

def greedy(net, n=100, eps=0.0, seed0=10_000):
    """Rollouts. trace columns: t, |pole angle|, max Q, |Q(left) - Q(right)|."""
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(0)
    scores, trace = [], []
    
    for i in range(n):
        s, _ = env.reset(seed=seed0 + i)
        done, t = False, 0
        
        while not done:
            with torch.no_grad(): 
                q = net(torch.as_tensor(s, dtype=torch.float32))
            a = int(rng.integers(2)) if rng.random() < eps else q.argmax().item()
            trace.append((t, abs(s[2]), q.max().item(), abs((q[0] - q[1]).item())))
            s, _, term, trunc, _ = env.step(a)
            done = term or trunc
            t += 1
        scores.append(t)
    return np.array(scores), np.array(trace)

### ENVIRONMENT FACTS
def env_facts():
    section("ENVIRONMENT FACTS: what a failing policy scores")
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(0)
    def mean_len(fn, n=200):
        L = []
        for i in range(n):
            env.reset(seed=i)
            k, done = 0, False
            while not done:
                _, _, t, tr, _ = env.step(fn())
                k += 1
                done = t or tr
            L.append(k)
        return np.mean(L)
    print(f"\trandom policy: {mean_len(lambda: int(rng.integers(2))):.1f} steps")
    print(f"\talways push left: {mean_len(lambda: 0):.1f} steps, always push right: {mean_len(lambda: 1):.1f} steps")
    print(f"\tQ ceiling 1/(1-gamma) = {CEIL:.1f} time limit = 500 steps")

### Edge Cases - default runs only 
def e1_time_limit():
    section("E1 TIME LIMIT IS NOT A FAILURE (we bootstrap through step 500)")
    sc, tr = greedy(trained_net(), 50)
    early, late = tr[tr[:, 0] < 400, 2], tr[tr[:, 0] >= 450, 2]
    print(f"\tgreedy mean = {sc.mean():.1f}, episodes reaching 500 = {(sc >= 500).mean():.0%}")
    print(f"\tmean max Q at t<400 = {early.mean():.2f}, at t>=450 = {late.mean():.2f}  (late Q should not drop)")

def e2_action_gap():
    """The pole angle gap drives the policy so Q should grow as it tips."""
    section("E2 CONSTANT REWARD: Q is about 20 everywhere, the action gap carries the policy")
    
    tr = greedy(trained_net(), 100, eps=0.2)[1]
    bins = np.linspace(0, 0.21, 8)
    idx = np.digitize(tr[:, 1], bins)
    xs, gap, q = [], [], []
    
    print(f"\t{'|angle| rad':>12}{'mean max Q':>12}{'mean gap':>10}{'n':>8}")
    for k in range(1, len(bins)):
        m = idx == k
        if m.any():
            xs.append(bins[k - 1])
            q.append(tr[m, 2].mean())
            gap.append(tr[m, 3].mean())
            print(f"\t{xs[-1]:>12.3f}{q[-1]:>12.2f}{gap[-1]:>10.3f}{m.sum():>8}")
            
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(xs, q, "o-")
    ax[0].axhline(CEIL, color="k", ls="--")
    ax[0].set_xlabel("|pole angle| (rad)")
    ax[0].set_ylabel("mean max Q")
    ax[1].plot(xs, gap, "o-")
    ax[1].set_xlabel("|pole angle| (rad)")
    ax[1].set_ylabel("mean |Q(left) - Q(right)|")
    plt.tight_layout()
    plt.savefig(FIG / "e2_action_gap.png", dpi=150)
    plt.close()

def e3_e4_training_dynamics():
    section("E3/E4 Q CEILING AND FAILURE EXAMPLES IN THE BUFFER OVER TRAINING")
    runs = get("default")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for k, r in enumerate(runs):
        ax[0].plot(np.arange(99, len(r["rewards"])), ma(r["rewards"]), label=f"seed {k}")
        ax[1].plot(r["log"][:, 0], r["log"][:, 1])
        ax[2].plot(r["log"][:, 0], r["log"][:, 2])
        print(f"\tseed {k}: final held-out max Q = {r['log'][-1, 1]:.2f}, peak = {r['log'][:, 1].max():.2f}, terminal share of buffer {r['log'][0, 2]:.3f} -> {r['log'][-1, 2]:.3f}, collapse fraction = {collapse_fraction(r['rewards']):.2f}")
        
    ax[1].axhline(CEIL, color="k", ls="--", label="1/(1-gamma)")
    ax[0].set_xlabel("episode")
    ax[0].set_ylabel("avg100 reward")
    ax[0].legend()
    ax[1].set_xlabel("env step")
    ax[1].set_ylabel("held-out mean max Q")
    ax[1].legend()
    ax[2].set_xlabel("env step")
    ax[2].set_ylabel("terminal share of buffer")
    plt.tight_layout()
    plt.savefig(FIG / "e3_e4_dynamics.png", dpi=150)
    plt.close()

def e5_exploration():
    section("E5 EXPLORATION NOISE: greedy vs eps-greedy on the trained model")
    net = trained_net()
    for eps in (0.0, 0.05, 0.1, 0.2):
        sc = greedy(net, 100, eps=eps)[0]
        print(f"\teps = {eps:<4} mean = {sc.mean():.1f}, std = {sc.std():.1f}, reached 500: {(sc >= 500).mean():.0%}")

### Validation tests
def v_chain():
    section("V1 KNOWN MDP: 3-state chain with analytic Q*")
    E = np.eye(4)
    S, A, S2, R, D = [], [], [], [], []
    for s in range(3):
        for a in (0, 1): # action 1 moves right, action 0 stays; reward 1 on reaching state 3
            s2 = s + 1 if a == 1 else s
            S.append(E[s])
            A.append(a)
            S2.append(E[s2])
            R.append(float(s2 == 3))
            D.append(float(s2 == 3))
            
    f = lambda x, dt=torch.float32: torch.as_tensor(np.array(x), dtype=dt)
    S, A, S2, R, D = f(S), f(A, torch.long), f(S2), f(R), f(D)
    g = GAMMA
    qstar = np.array([[g**3, g**2], [g**2, g], [g, 1.0]]) # Q*[s][a]
    
    def fit(gamma):
        torch.manual_seed(0)
        o, t = cd.CartpoleDQN(), cd.CartpoleDQN()
        t.load_state_dict(o.state_dict())
        
        for p in t.parameters(): 
            p.requires_grad_(False)
            
        opt = torch.optim.Adam(o.parameters(), 1e-3)
        
        for i in range(4000):
            y = cd.dqn_target(o, t, S2, R, D, gamma)
            loss = F.smooth_l1_loss(o(S).gather(1, A.unsqueeze(1)).squeeze(1), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 50 == 0:
                t.load_state_dict(o.state_dict())
        with torch.no_grad(): 
            return o(f(E[:3])).numpy()
            
    q = fit(g)
    err = np.abs(q - qstar).max()
    
    print("\tlearned Q:\n", np.round(q, 3), "\n\tanalytic Q*:\n", np.round(qstar, 3))
    check("V1a learned Q matches the analytic Q*", err < 0.05, f"max error = {err:.3f}")
    err_bad = np.abs(fit(0.5) - qstar).max()
    check("V1b the test can fail: a wrong gamma (0.5) in the update is detected", err_bad > 0.1, f"max error = {err_bad:.3f}")

def v_trace():
    section("V2 TRACE OF 10 CONSECUTIVE TRANSITIONS: hand-computed target vs dqn_target")
    torch.manual_seed(0)
    online = cd.CartpoleDQN()
    torch.manual_seed(1)
    target = cd.CartpoleDQN()
    
    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(0)
    
    s, _ = env.reset(seed=0)
    tr = []
    for _ in range(60):
        a = int(rng.integers(2))
        s2, r, term, trunc, _ = env.step(a)
        tr.append((s, a, s2, float(r), float(term)))
        s = env.reset()[0] if (term or trunc) else s2
        
    k = next((i for i, x in enumerate(tr) if x[4] == 1.0), 10)
    win = tr[max(0, k - 6):max(0, k - 6) + 10]
    print(f"\t{'a':>2}{'r':>5}{'done':>6}{'Q(s,a)':>9}{'a*':>4}{'Qt(s2,a*)':>11}{'y by hand':>11}")
    y_hand = []
    for (s, a, s2, r, d) in win:
        with torch.no_grad():
            q = online(torch.as_tensor(s, dtype=torch.float32))[a].item()
            astar = online(torch.as_tensor(s2, dtype=torch.float32)).argmax().item()
            qt = target(torch.as_tensor(s2, dtype=torch.float32))[astar].item()
        y_hand.append(r + GAMMA * (1 - d) * qt)
        print(f"\t{a:>2}{r:>5.0f}{d:>6.0f}{q:>9.3f}{astar:>4}{qt:>11.3f}{y_hand[-1]:>11.3f}")
        
    col = lambda i, dt=torch.float32: torch.as_tensor(np.array([x[i] for x in win]), dtype=dt)
    y_vec = cd.dqn_target(online, target, col(2), col(3), col(4), GAMMA)
    check("V2a dqn_target equals the hand-computed targets", torch.allclose(y_vec, torch.tensor(y_hand, dtype=torch.float32), atol=1e-5))
    done = col(4) == 1
    check("V2b at terminal transitions y equals r exactly", (not done.any()) or torch.allclose(y_vec[done], col(3)[done]))

def v_determinism():
    section("V3 REDUCED STOCHASTICITY: greedy policy, fixed seeds")
    net = trained_net()
    a, b = greedy(net, 20)[0], greedy(net, 20)[0]
    check("V3a greedy rollouts with the same env seeds repeat exactly", np.array_equal(a, b))
    print(f"\tspread over 20 start states (the only randomness left): mean = {a.mean():.1f}, std = {a.std():.1f}, min = {a.min()}")

def v_direction():
    section("V4 LEARNING DIRECTION: gamma = 0 and negated reward")
    g0 = train_cp(0, 10_000, gamma=0.0)
    check("V4a gamma = 0: Q converges to the immediate reward 1", 0.8 <= g0["log"][-1, 1] <= 1.2, f"held-out max Q = {g0['log'][-1, 1]:.2f}")
    fl = train_cp(0, 15_000, flip=True)
    f0, f1 = fl["rewards"][:100].mean(), fl["rewards"][-100:].mean()
    check("V4b negated reward: the agent learns to fail faster", f1 < 0.75 * f0, f"episode length {f0:.1f} to {f1:.1f}")
    check("V4c negated reward: Q goes negative", fl["log"][-1, 1] < 0, f"held-out max Q = {fl['log'][-1, 1]:.2f}")

# ABLATIONS - we retrain models, to run set RUN_ABLATIONS =True #
def ablations():
    section("ABLATIONS")
    
    # A1 time limit stored as terminal
    print(f"\t{'A1 done rule':<28}{'greedy mean ± std (seeds)':>28}{'seeds reaching t>=450':>24}")
    for name, key, kw in (("terminated only", "default", {}), ("terminated or truncated", "trunc", dict(done_on_trunc=True))):
        gm, late = [], []
        for r in get(key, **kw):
            sc, tr = greedy(r["online"], 30)
            gm.append(sc.mean())
            m = tr[:, 0] >= 450
            if m.any():
                late.append(tr[m, 2].mean())
        late_s = f"{len(late)}/{len(gm)}" + (f" (late Q {np.mean(late):.2f})" if late else "")
        print(f"\t{name:<28}{np.mean(gm):>14.1f} ± {np.std(gm):<10.1f}{late_s:>24}")

    # A2 Double DQN vs vanilla DQN: What changes in the training curve
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, key, kw in (("Double DQN", "default", {}), ("vanilla DQN", "vanilla", dict(double=False))):
        for k, r in enumerate(get(key, **kw)): 
            ax.plot(r["log"][:, 0], r["log"][:, 1], color="C0" if key == "default" else "C1", label=name if k == 0 else None)
    ax.axhline(CEIL, color="k", ls="--")
    ax.set_xlabel("env step")
    ax.set_ylabel("held-out mean max Q")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "a2_double_vs_vanilla.png", dpi=150)
    plt.close()
    
    # A3 buffer size: how does buffer size change our training behavior
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, key, kw in (("buffer 1k (real)", "default", {}), ("buffer 50k", "buf50k", dict(buf_cap=50_000))):
        runs = get(key, **kw)
        for k, r in enumerate(runs):
            ax.plot(np.arange(99, len(r["rewards"])), ma(r["rewards"]), color="C0" if key == "default" else "C1", label=name if k == 0 else None)
        print(f"\t{name}: final avg100 = {np.mean([r['rewards'][-100:].mean() for r in runs]):.1f}, collapse = {np.mean([collapse_fraction(r['rewards']) for r in runs]):.2f}")
        
    ax.set_xlabel("episode")
    ax.set_ylabel("avg100 reward")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "a3_buffer.png", dpi=150)
    plt.close()
    
    # A4 target sync interval
    fig, ax = plt.subplots(figsize=(7, 4))
    print(f"\t{'sync every':>11}{'final avg100':>14}{'collapse':>10}")
    for n in (1, 100, 1000, 5000):
        runs = get("default") if n == 1000 else get(f"sync{n}", sync_every=n)
        for k, r in enumerate(runs):
            ax.plot(np.arange(99, len(r["rewards"])), ma(r["rewards"]), color=f"C{[1,100,1000,5000].index(n)}", label=f"sync {n}" if k == 0 else None)
        print(f"\t{n:>11}{np.mean([r['rewards'][-100:].mean() for r in runs]):>14.1f}{np.mean([collapse_fraction(r['rewards']) for r in runs]):>10.2f}")
        
    ax.set_xlabel("episode")
    ax.set_ylabel("avg100 reward")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "a4_sync.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    env_facts()
    print("TEST: UPDATE")
    e1_time_limit()
    e2_action_gap()
    e3_e4_training_dynamics()
    e5_exploration()
    
    v_chain()
    v_trace()
    v_determinism()
    v_direction()
    
    if RUN_ABLATIONS: 
        ablations()
        
    finish()