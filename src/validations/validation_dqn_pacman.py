'''
validation_dqn_pacman.py

Edge cases and validation runs for MsPacman-v5.
'''

import sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, gymnasium as gym, ale_py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.mspacman.model import PacmanDQN
from src.mspacman.replay_buffer import ReplayBuffer # preallocated version with .s .a .s2 .r .d .n
from src.mspacman.frame_stack import FrameStack
from src.mspacman.mspacman_dqn import preprocess_observation, update, device
# sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.validations.val_common import check, section, finish

gym.register_envs(ale_py)
GAMMA, CAP = 0.99, 100_000
LIFE_LOSS_TERMINAL = True # We count losing a life as a terminal state, as we did during training.
RUN_SLOW = True # V4 takes about 2 to 4 minutes on a GPU
CKPT = Path("src/mspacman/output/pacman_dqn_final_run_1.pt") # modify for correct path
TARGET_PT = Path("src/mspacman/output/pacman_dqn_run_1.pt") # modify for correct path
FIG = Path("figures"); FIG.mkdir(exist_ok=True)
pre = lambda o: preprocess_observation(o.copy())[:, :, 0]
T = lambda x: torch.as_tensor(x, device=device)

def load(p):
    """Open a Pacman trained model, either the final or the target."""
    net = PacmanDQN(9).to(device)
    net.load_state_dict(torch.load(p, map_location=device))
    return net.eval()

def make(sticky=0.25): 
    """Modify the stochasticity of the environment."""
    return gym.make("ALE/MsPacman-v5", repeat_action_probability=sticky)

def rollout(env, net, n, eps=0.0, keep_states=0):
    """
    Rollout some episodes for the given network and environment. Record corresponding stats for each episode.

    Args:
        env (Gymnasium Environment Object): The environment we are running
        net (PyTorch Neural Network Class): The model
        n (Integer): The number of episodes to run
        eps: Epsilon
        keep_states: How many states to retain
    """
    stack, rng, kept = FrameStack(4), np.random.default_rng(0), []
    R = {k: [] for k in ("score", "length", "maxq", "ghosts", "big_share", "nonzero", "stall")}
    for i in range(n):
        s = stack.reset(pre(env.reset(seed=i)[0]))
        tot = big = 0.0
        t = nz = ghosts = gap = worst = 0
        qmax, over = -1e9, False
        while not over:
            with torch.no_grad(): 
                q = net(T(s)[None])[0]
            qmax = max(qmax, q.max().item())
            if len(kept) < keep_states and rng.random() < 0.02: 
                kept.append(s)
            a = int(rng.integers(9)) if rng.random() < eps else q.argmax().item() # Greediness, exploration v exploitation
            o, r, te, tr, _ = env.step(a)
            s = stack.step(pre(o))
            over = te or tr
            tot += r
            t += 1
            if r > 0:
                nz += 1
                worst = max(worst, gap)
                gap = 0
                if r >= 200: 
                    ghosts += 1
                    big += r
            else: 
                gap += 1
        worst = max(worst, gap)
        for k, v in zip(R, (tot, t, qmax, ghosts, big / max(tot, 1), nz / t, worst)):
            R[k].append(v)
    return {k: np.array(v) for k, v in R.items()}, kept

# ++++++++++++++++++++++++++ EDGE CASES (on the saved checkpoint) ++++++++++++++++++++++++++ #
def p1_reward_clipping(net):
    """How much of the raw score comes from rewards of 200 or more, the clipping hides this fact from the network."""
    section("P1 REWARD CLIPPING: what the network cannot see")
    R, _ = rollout(make(), net, 8, eps=0.05)
    print(f"\tmax Q seen = {R['maxq'].max():.1f} (clipped ceiling {1/(1-GAMMA):.0f})")
    print(f"\trewards >= 200 per episode = {R['ghosts'].mean():.2f}; share of raw score from them = {R['big_share'].mean():.0%}")
    print("\tclipping gives a 1600-point ghost the same target as a 10-point pellet")

def p2_motion(net):
    """If the model is in fact tracking motion, then we should notice the greedy action flip flopping."""
    section("P2 SINGLE FRAME IS NOT MARKOV: does the network use history?")
    _, kept = rollout(make(), net, 3, eps=0.05, keep_states=800)
    X = T(np.stack(kept))
    with torch.no_grad():
        Q = net(X)
        Qs = net(X[:, 3:4].repeat(1, 4, 1, 1))
        Qr = net(torch.flip(X, dims=[1]))
    flip = lambda A: (A.argmax(1) != Q.argmax(1)).float().mean().item()
    print(f"\tstatic history (newest frame x4): mean |dQ| = {(Qs-Q).abs().mean():.3f}, greedy action changes on {flip(Qs):.0%} of states")
    print(f"\ttime-reversed history: mean |dQ| = {(Qr-Q).abs().mean():.3f}, greedy action changes on {flip(Qr):.0%} of states")

def p3_intro():
    """Plot changed pixels per step at episode start"""
    section("P3 FROZEN INTRO AT THE START OF EACH EPISODE")
    env, rng = make(), np.random.default_rng(0)
    prev = pre(env.reset(seed=0)[0])
    changed = []
    for _ in range(200):
        o, *_ = env.step(int(rng.integers(9)))
        f = pre(o)
        changed.append(np.count_nonzero(f != prev))
        prev = f
    changed = np.array(changed)
    print(f"\tof the first 200 steps, {(changed < 20).sum()} change fewer than 20 pixels; first step with > 100 changed pixels: {int(np.argmax(changed > 100))}")
    plt.figure(figsize=(7, 3))
    plt.plot(changed)
    plt.xlabel("step in episode")
    plt.ylabel("changed pixels")
    plt.tight_layout()
    plt.savefig(FIG / "p3_intro.png", dpi=150)
    plt.close()

def p4_sparse_rewards(net):
    """Fraction of steps with reward for random play against the trained model. Run 6 episodes."""
    section("P4 SPARSE REWARDS")
    for name, eps in (("random", 1.0), ("trained eps 0.05", 0.05)):
        R, _ = rollout(make(), net, 6, eps=eps)
        print(f"\t{name:<18} fraction of steps with reward = {R['nonzero'].mean():.3f}, mean score = {R['score'].mean():.0f}")

def p5_sticky(net):
    """Evaluates the checkpoint, target model, with sticky actions on and off. """
    section("P5 STICKY ACTIONS MAKE EVALUATION NOISY")
    for sticky in (0.25, 0.0):
        R, _ = rollout(make(sticky), net, 50, eps=0.05)
        print(f"\trepeat_action_probability = {sticky}: mean = {R['score'].mean():.0f}, std = {R['score'].std():.0f}")

def p6_preprocessing(net):
    """Raw vs. pre-processed frames, including 15 steps after power pellet."""
    section("P6 CAN PREPROCESSING STILL SEPARATE PAC-MAN, GHOSTS, PELLETS, BLUE GHOSTS?")
    env, stack = make(), FrameStack(4)
    s = stack.reset(pre(env.reset(seed=0)[0])); shots, t, pellet, over = {}, 0, None, False
    while not over and t < 3000:
        with torch.no_grad():
            a = net(T(s)[None]).argmax().item()
        o, r, te, tr, _ = env.step(a)
        s = stack.step(pre(o))
        over = te or tr
        t += 1
        if t in (5, 300): shots[f"step {t}"] = o.copy()
        if r == 50 and pellet is None: pellet = t
        if pellet and t == pellet + 15: 
            shots["15 steps after power pellet"] = o.copy()
    fig, ax = plt.subplots(2, len(shots), figsize=(4 * len(shots), 6), squeeze=False)
    for k, (name, o) in enumerate(shots.items()):
        ax[0, k].imshow(o)
        ax[0, k].set_title(f"raw: {name}")
        ax[1, k].imshow(pre(o), cmap="gray")
        ax[1, k].set_title("preprocessed")
        ax[0, k].axis("off"); ax[1, k].axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "p6_preprocessing.png", dpi=150)
    plt.close()
    print(f"\tsaved figures/p6_preprocessing.png ({'power pellet frame included' if pellet else 'no power pellet eaten this episode'}); check the blue ghost is visible")

def p7_stall_and_staleness(net):
    "Episode length and longest no reward stretch for greedy vs some exploration. Also checks how many episodes the buffer holds."
    section("P7 GREEDY STALLING AND BUFFER STALENESS")
    for name, eps in (("greedy", 0.0), ("eps 0.05", 0.05)):
        R, _ = rollout(make(), net, 10, eps=eps)
        print(f"\t{name:<9} mean length = {R['length'].mean():.0f}, max length = {R['length'].max()}, longest stretch with no reward = {R['stall'].max()} steps")
    print(f"\tbuffer of {CAP} transitions holds about {CAP / R['length'].mean():.0f} episodes; sampled transitions are on average {CAP // 2} env steps old")

### ++++++++++++++++++++++++++ VALIDATION ++++++++++++++++++++++++++
def v_trace(online, target, target_note):
    """
    Hand computes a small portion of the loss of an episode and ensures that our network matches it.
    """
    section(f"V1 TRACE OF 10 CONSECUTIVE TRANSITIONS: hand-computed target vs update() ({target_note})")
    env, stack, rng = make(0.0), FrameStack(4), np.random.default_rng(0)
    o0, info0 = env.reset(seed=0)
    s = stack.reset(pre(o0))
    lives = info0["lives"]
    tr = []
    for _ in range(800):
        with torch.no_grad(): 
            a = online(T(s)[None]).argmax().item() if rng.random() > 0.2 else int(rng.integers(9))
        o, r, te, trn, info = env.step(a)
        s2 = stack.step(pre(o))
        lost = info["lives"] < lives
        lives = info["lives"]
        d = float(te or (LIFE_LOSS_TERMINAL and lost))
        tr.append((s, a, s2, float(np.sign(r)), r, float(d)))
        s = s2
        if te or trn: 
            break
    k = next((i for i, x in enumerate(tr) if x[5] == 0), 0)
    first = max(0, k - 6)
    win = tr[first:first + 10]
    print(f"\twindow starts at step {first}\n\t{'a':>2}{'raw r':>7}{'clip r':>8}{'done':>6}{'Q(s,a)':>9}{'a*':>4}{'Qt(s2,a*)':>11}{'y by hand':>11}")
    
    y_hand = []
    for (s, a, s2, rc, rr, d) in win:
        with torch.no_grad():
            q = online(T(s)[None])[0, a].item()
            astar = online(T(s2)[None])[0].argmax().item()
            qt = target(T(s2)[None])[0, astar].item()
        y_hand.append(rc + GAMMA * (1 - d) * qt)
        print(f"\t{a:>2}{rr:>7.0f}{rc:>8.0f}{q:>9.3f}{astar:>4}{qt:>11.3f}{y_hand[-1]:>11.3f}")
        
    S, A, S2, R, D = [T(np.array([x[i] for x in win])) for i in (0, 1, 2, 3, 5)]
    with torch.no_grad():
        y_vec = R.float() + GAMMA * (1 - D.float()) * target(S2).gather(1, online(S2).argmax(1, keepdim=True)).squeeze(1)
    check("V1a vectorised Double DQN target equals the hand-computed targets", torch.allclose(y_vec.cpu(), torch.tensor(y_hand, dtype=torch.float32), atol=1e-4))
    buf = ReplayBuffer(16)
    for (s, a, s2, rc, rr, d) in win: 
        buf.push(s, a, s2, rc, d)
    opt = torch.optim.SGD(online.parameters(), lr=0.0) # lr 0: weights untouched
    
    np.random.seed(7)
    Sb, Ab, S2b, Rb, Db = [T(x) for x in buf.sample(10)]
    
    with torch.no_grad(): 
        yb = Rb + GAMMA * (1 - Db) * target(S2b).gather(1, online(S2b).argmax(1, keepdim=True)).squeeze(1)
        
    with torch.no_grad(): 
        exp = F.smooth_l1_loss(online(Sb).gather(1, Ab[:, None]).squeeze(1), yb).item()
    np.random.seed(7)
    got = update(online, target, opt, buf, GAMMA, 10)
    check("V1b update() loss equals the hand-computed Double DQN loss", abs(got - exp) / max(abs(exp), 1e-8) < 1e-4, f"{got:.6f} vs {exp:.6f}")

def v_determinism(net):
    """NO STOCHASTICITY, do we get identical results?"""
    section("V2 REDUCED STOCHASTICITY: greedy policy, no sticky actions, fixed seeds")
    a, _ = rollout(make(0.0), net, 3)
    b, _ = rollout(make(0.0), net, 3)
    check("V2a identical seeds give identical scores and lengths", np.array_equal(a["score"], b["score"]) and np.array_equal(a["length"], b["length"]),
          f"scores = {a['score'].tolist()} vs {b['score'].tolist()}")

def v_direction():
    """
    Fits update() on a fixed random buffer. With normal rewards the Q of the taken actions should go positive, and with negated rewards negative. It also prints the terminal flags per game, which should be about 3 with life loss as terminal.

    Is our network learning from the reward sign? If the sign goes negative, then the Q-Values should end up negative.
    """
    section("V3 LEARNING DIRECTION: fit on a fixed random buffer, then negate the rewards")
    
    env = make()
    env.action_space.seed(0)
    stack = FrameStack(4)
    buf = ReplayBuffer(10_000)
    s = stack.reset(pre(env.reset(seed=0)[0]))
    lives, dones, games = 3, 0.0, 0
    
    for _ in range(10_000):
        # Fill with random steps, track terminal flags and how many games completed
        a = env.action_space.sample()
        o, r, te, tr, info = env.step(a)
        s2 = stack.step(pre(o))
        
        lost = info["lives"] < lives
        lives = info["lives"]
        d = float(te or (LIFE_LOSS_TERMINAL and lost))
        dones += d
        
        buf.push(s, a, s2, float(np.sign(r)), d)
        s = s2
        if te or tr: 
            games += 1
            s = stack.reset(pre(env.reset()[0]))
            lives = 3
            
    print(f"\tterminal flags per finished game in random data: {dones / max(games, 1):.1f} ({'expected about 3 with life loss as terminal' if LIFE_LOSS_TERMINAL else 'expected 1'}) nonzero reward share = {(buf.r[:buf.n] != 0).mean():.3f}")
    
    def fit():
        torch.manual_seed(0)
        o, t = PacmanDQN().to(device), PacmanDQN().to(device)
        t.load_state_dict(o.state_dict())
        for p in t.parameters(): 
            p.requires_grad_(False)
        opt = torch.optim.Adam(o.parameters(), 1e-4)
        for i in range(3000):
            update(o, t, opt, buf, GAMMA, 32)
            if i % 200 == 0: 
                t.load_state_dict(o.state_dict())
        with torch.no_grad(): 
            return o(T(buf.s[:1000])).gather(1, T(buf.a[:1000])[:, None]).mean().item()
            
    q_pos = fit()
    buf.r *= -1
    q_neg = fit()
    check("V3a normal rewards: Q of the taken actions goes positive", q_pos > 0.02, f"mean Q = {q_pos:+.3f}")
    check("V3b negated rewards: Q of the taken actions goes negative", q_neg < -0.02, f"mean Q = {q_neg:+.3f}")

if __name__ == "__main__":
    online = load(CKPT)
    target, note = (load(TARGET_PT), "saved target") if TARGET_PT.exists() else (load(CKPT), "target missing, online net used as stand-in")
    p1_reward_clipping(online)
    p2_motion(online)
    p3_intro()
    p4_sparse_rewards(online)
    p5_sticky(online)
    p6_preprocessing(online)
    p7_stall_and_staleness(online)
    
    v_trace(online, target, note)
    v_determinism(online)
    if RUN_SLOW: v_direction()
        
    finish()