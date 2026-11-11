# Ms. Pacman DQN Validation: Test Definitions

## Edge-Case Probes (run on the saved checkpoint)

### P1: Reward Clipping
**Function:** `p1_reward_clipping(net)`
Runs 8 episodes (`eps=0.05`) and measures how much of the raw score comes from rewards of 200 or more (ghost eats). Training clips rewards to their sign, so a 1600-point ghost gets the same target as a 10-point pellet, and the network cannot see the difference. Reports the max Q seen compared with the clipped ceiling (`1/(1-GAMMA) = 100`), the average number of big rewards per episode, and their share of the raw score.

### P2: Single Frame Is Not Markov
**Function:** `p2_motion(net)`
Tests whether the network actually uses frame history. Collects up to 800 states over 3 episodes and compares Q-values on the original stack with two altered stacks:
- **Static history:** the newest frame repeated 4 times.
- **Time-reversed history:** the frame order flipped.

Reports mean absolute Q change and the percentage of states where the greedy action changes. If the network tracks motion, both alterations should change its behavior noticeably.

### P3: Frozen Intro
**Function:** `p3_intro()`
Takes 200 random steps after a reset and counts the pixels that change between consecutive frames. Shows that the start of each episode is a near-static intro. Prints how many steps change fewer than 20 pixels and the first step with more than 100 changed pixels. Saves `figures/p3_intro.png`.

### P4: Sparse Rewards
**Function:** `p4_sparse_rewards(net)`
Runs 6 episodes each for random play (`eps=1.0`) and the trained policy (`eps=0.05`). Reports the fraction of steps with a reward and the mean score, to quantify how sparse the reward signal is.

### P5: Sticky Actions Make Evaluation Noisy
**Function:** `p5_sticky(net)`
Evaluates the checkpoint over 50 episodes with sticky actions on (`0.25`) and off (`0.0`). Reports the mean and standard deviation of the score, showing how much sticky actions add evaluation noise.

### P6: Preprocessing Still Separates Objects
**Function:** `p6_preprocessing(net)`
Plays one greedy episode (up to 3000 steps) and saves raw and preprocessed frames side by side at step 5, at step 300, and 15 steps after the first power pellet (reward of 50). Saves `figures/p6_preprocessing.png`. It is a visual check that Pac-Man, ghosts, pellets, and blue (edible) ghosts remain distinguishable after preprocessing.

### P7: Greedy Stalling and Buffer Staleness
**Function:** `p7_stall_and_staleness(net)`
Compares greedy (`eps=0`) and `eps=0.05` play over 10 episodes each. Reports mean length, max length, and the longest stretch with no reward, which reveals whether the greedy policy stalls. Then estimates how many episodes fit in the replay buffer (`CAP / mean length`) and the average age of a sampled transition (`CAP // 2` environment steps).

## Validations

### V1: Hand-Computed Target vs. `update()`
**Function:** `v_trace(online, target, target_note)`
Plays up to 800 steps (no sticky actions, 20% random actions), then takes a window of 10 consecutive transitions. For each one it computes by hand `Q(s,a)`, the online net's best next action `a*`, `Q_target(s2, a*)`, and the Double DQN target `y = r_clipped + GAMMA * (1 - done) * Q_target(s2, a*)`. Prints a table of these values.

- **V1a:** The vectorised Double DQN target matches the hand-computed targets (`atol=1e-4`).
- **V1b:** The loss returned by `update()` equals the hand-computed Smooth L1 (Huber) loss on the same sampled batch. The optimizer uses `lr=0.0` so weights are untouched, and the NumPy seed is reset so both sample the same batch.

*Note:* the window start `k` is the first index where `done == 0`, which is usually index 0. The window therefore may not include a terminal transition. To center the window on a terminal one, search for `x[5] == 1` instead.

### V2: Reduced Stochasticity
**Function:** `v_determinism(net)`
Runs the greedy policy for 3 episodes twice with no sticky actions and fixed seeds.

- **V2a:** Scores and lengths are identical across the two runs.

### V3: Learning Direction
**Function:** `v_direction()`
Checks that the network learns from the sign of the reward. Fills a 10,000-transition buffer with random play (clipped rewards, life loss as terminal) and prints terminal flags per finished game (about 3 expected) and the nonzero-reward share. Then `fit()` trains a fresh network (seed 0, Adam `lr=1e-4`, 3000 updates, batch 32, target sync every 200 updates) and returns the mean Q of the taken actions on the first 1000 stored transitions. It fits once on normal rewards, negates `buf.r`, and fits again.

- **V3a:** Mean Q with normal rewards is greater than `0.02`.
- **V3b:** Mean Q with negated rewards is less than `-0.02`.

Only runs when `RUN_SLOW = True` (about 2 to 4 minutes on a GPU).