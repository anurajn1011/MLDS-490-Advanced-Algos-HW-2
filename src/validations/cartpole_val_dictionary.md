# CartPole DQN Validation: Test Definitions

## Environment Facts

### ENV FACTS: What a Failing Policy Scores
**Function:** `env_facts()`
Establishes baselines by measuring mean episode length over 200 episodes for a random policy, always pushing left, and always pushing right. Also prints the Q ceiling (`1/(1-GAMMA) = 20` at `GAMMA = 0.95`) and the 500-step time limit. These numbers show what a failing policy scores, so a trained agent's results can be judged against them.

## Edge Cases (run by default, on the saved model)

### E1: Time Limit Is Not a Failure
**Function:** `e1_time_limit()`
Runs 50 greedy episodes and reports the mean score and the share of episodes reaching 500 steps. Compares the mean max Q early in episodes (`t < 400`) with late (`t >= 450`). The 500-step limit is a truncation, not a real failure, so the agent should bootstrap through it. If it does, late Q should not drop.

### E2: Constant Reward, Action Gap Carries the Policy
**Function:** `e2_action_gap()`
Runs 100 episodes with `eps=0.2` and bins the visited states by absolute pole angle (0 to 0.21 rad, 7 bins). For each bin it reports mean max Q, mean action gap `|Q(left) - Q(right)|`, and sample count. Since the reward is a constant 1 every step, Q is roughly the same everywhere, so the policy is determined by the small gap between actions, which should grow as the pole tips. Saves `figures/e2_action_gap.png` (max Q with the ceiling line, and gap, each against angle).

### E3/E4: Q Ceiling and Failure Examples in the Buffer Over Training
**Function:** `e3_e4_training_dynamics()`
Uses the cached default training runs (3 seeds, 100,000 steps each) and plots three curves per seed:
- Average-100 reward per episode.
- Held-out mean max Q over env steps, with the `1/(1-gamma)` ceiling.
- The terminal share of the replay buffer over env steps.

Prints for each seed the final and peak held-out Q, how the terminal share changed from start to end, and the collapse fraction (the share of time the moving average fell below half its peak after exceeding 100). This reveals Q overestimation past the ceiling, and whether the small buffer fills with too few failure examples, which causes instability. Saves `figures/e3_e4_dynamics.png`.

### E5: Exploration Noise
**Function:** `e5_exploration()`
Evaluates the trained model over 100 episodes at `eps` of 0.0, 0.05, 0.1, and 0.2. Reports mean, standard deviation, and the share of episodes reaching 500. Shows how robust the policy is to random action noise.

## Validations

### V1: Known MDP With Analytic Q*
**Function:** `v_chain()`
Builds a 3-state chain with one-hot states. Action 1 moves right and action 0 stays, with a reward of 1 on reaching state 3, which is terminal. The analytic optimal values are `Q* = [[g^3, g^2], [g^2, g], [g, 1]]`. Trains a network with `dqn_target` (4000 Adam steps, `lr=1e-3`, target sync every 50 steps).

- **V1a:** The learned Q matches the analytic `Q*` (max error below 0.05).
- **V1b:** The test can fail. Training with a wrong gamma of 0.5 produces a max error above 0.1, proving the test detects a broken update.

### V2: Trace of 10 Consecutive Transitions
**Function:** `v_trace()`
Collects 60 random-action transitions with two untrained, differently seeded networks as online and target. Takes a 10-transition window around the first terminal transition. For each it computes by hand `Q(s,a)`, the online net's best next action `a*`, `Q_target(s2, a*)`, and the target `y = r + GAMMA * (1 - done) * Q_target(s2, a*)`. Prints a table of these values.

- **V2a:** `dqn_target` equals the hand-computed targets (`atol=1e-5`).
- **V2b:** At terminal transitions, `y` equals `r` exactly (passes trivially if the window has no terminal transition).

### V3: Reduced Stochasticity
**Function:** `v_determinism()`
Runs the greedy trained policy for 20 episodes twice with the same env seeds.

- **V3a:** The two runs give identical scores.

Also prints the spread (mean, std, min) across the 20 start states, which is the only randomness left.

### V4: Learning Direction
**Function:** `v_direction()`
Checks that the agent responds correctly to changes in the reward signal.

- **V4a:** With `gamma = 0` (10,000 steps), Q converges to the immediate reward of 1 (held-out max Q between 0.8 and 1.2).
- **V4b:** With negated reward (15,000 steps), the agent learns to fail faster (last-100 mean episode length is below 75% of the first-100 mean).
- **V4c:** With negated reward, held-out max Q goes negative.

Each of the following ablations use 3 seeds, and results are cached so shared configurations are trained once.

### Ablation 1: Time Limit Stored as Terminal
Compares `done = terminated` only against `done = terminated or truncated`. Reports greedy mean ± std across seeds and how many seeds reach `t >= 450`, with the mean late Q for those. Shows the harm of treating the time limit as a true failure, which teaches the network that Q should drop near step 500.

### Ablation 2: Double DQN vs. Vanilla DQN
Plots held-out mean max Q over training for both variants against the `1/(1-gamma)` ceiling. Shows whether vanilla DQN overestimates Q more than Double DQN. Saves `figures/a2_double_vs_vanilla.png`.

### Ablation 3: Buffer Size
Compares the real 1,000-transition buffer against a 50,000 buffer. Plots average-100 reward curves and prints final average-100 reward and collapse fraction for each. Shows how buffer size changes training stability. Saves `figures/a3_buffer.png`.

### Ablation 4: Target Sync Interval
Compares syncing the target network every 1, 100, 1000 (default), and 5000 steps. Plots average-100 reward curves and prints final average-100 reward and collapse fraction for each interval. Shows how sync frequency affects stability and learning speed. Saves `figures/a4_sync.png`.