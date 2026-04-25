# Team Poker: Mathematical Foundations for SeqComm Testing

## 1. Game Setup

### 1.1 Per-Hand Structure

N homogeneous agents (parameter-sharing) play repeated rounds against a dealer. Each hand:

- Agent i privately observes hand strength `H_i ~ Uniform[0,1]` i.i.d.
- Dealer draws `D ~ Uniform[0,1]` independently
- Team wins iff `H̄ > D`, where `H̄ = (1/N) Σ H_i`
- Win probability: `p(H̄) = P(D < H̄) = H̄` (since D is uniform)
- Agent i chooses bet fraction `b_i ∈ [0,1]` of current coffers `C_i`

**Payoff per hand (normalized by C_i):**

```
Win (prob H̄):   ΔC_i / C_i = M · b_i
Lose (prob 1-H̄): ΔC_i / C_i = α - b_i(1+α)
```

where `M` is the net profit multiplier on a win, and `α · (1 - b_i)` is the consolation on a loss (the fraction of unbet chips returned). This gives two boundary cases:
- `b_i = 0`: guaranteed return `α` regardless of outcome (risk-free consolation)
- `b_i = 1`: win returns `M`, lose returns `-1` (bankruptcy)

**Expected per-hand reward:**

```
EV(b_i, H̄) = H̄ · M · b_i  +  (1 - H̄) · [α - b_i(1+α)]
            = b_i · [H̄(M+1+α) - (1+α)]  +  (1 - H̄) · α
```

This is affine in `b_i` given `H̄`. The slope changes sign at the critical threshold:

```
θ_opt = (1 + α) / (M + 1 + α)
```

---

## 2. World Model Approximation

SeqComm's world model `M(h_t, a_t)` must predict `r̂_{t+1}` given joint hidden states and actions. In this environment the world model function is exact and closed-form.

### 2.1 What the World Model Learns

The hidden state `h_i` encodes `H_i`. With perfect encoding, the world model needs to compute:

```
r̂(h_t, a_t) = H̄ · M · b_i  +  (1 - H̄) · [α - b_i(1+α)]
```

This is a bilinear function of `H̄` and `b_i`. Concretely, expanding:

```
r̂ = b_i · H̄(M+1+α)  -  b_i(1+α)  +  α - α · H̄
```

The world model decomposes into two learned scalars: the slope coefficient `(M+1+α)` and the intercept term `(1+α)`. A single linear layer on `(H̄, b_i)` can represent this exactly.

### 2.2 Intention Value Under the World Model

In the negotiation phase, agent i treats itself as first mover and simulates F trajectories of length H. For a trajectory where agent i goes first and the remaining order is sampled uniformly:

```
v_i = (1/F) Σ_f  Σ_{t'=1}^{H} γ^{t'-1} r̂(ĥ_{t+t'}, â_{t+t'})
```

where `â_{t+t'}` is the predicted joint action under the world model. Since the world model is exact and the EV is increasing in `H̄` for `b_i > 0`, the intention value is monotonically increasing in `H_i`:

**Claim:** Under a fully accurate world model and a policy that bets more when H̄ is higher, the intention value satisfies `v_i > v_j` whenever `H_i > H_j`.

**Argument:** Agent i as first mover sets `â_i = π(H_i, ∅)`. Since π is increasing in H_i (it should bet more with a stronger hand), higher `H_i` → higher `â_i` → higher `H̄_eff` (because â_i is the leading signal) → higher predicted reward. Agents with lower `H_j` as followers condition on â_i and update upward, but agent i's signal is the anchor. Therefore `v_i` is strictly increasing in `H_i` for the relevant regime. ∎

This is the key connection: SeqComm's intention-value ordering is equivalent to ordering agents by hand strength under a perfect world model.

### 2.3 World Model Error Bound (from Theorem 1)

With observation noise `ε_i ~ N(0, σ²)` on the hidden states, the world model error on each step is:

```
|r̂ - r| ≤ |ΔH̄| · (M + 1 + α) · max_i(b_i)
```

where `ΔH̄ = (1/N) Σ ε_i ~ N(0, σ²/N)`. From Theorem 1 in SeqComm, the ordering is reliable when the model error C(ε_m, ε_π) is less than the ordering advantage. Setting the noise condition:

```
σ / √N  <  θ_opt  =  (1+α) / (M+1+α)
```

This gives the **maximum noise tolerance** for which SeqComm's ordering is meaningfully better than random. For `M=2, α=0.1, N=2`: tolerance is `σ < 0.355 · √2 ≈ 0.50`. Wide margin for the "fully accurate world model + added noise" setting.

---

## 3. Optimal Betting Policy

### 3.1 Threshold Under Full Information (EV objective)

Given knowledge of `H̄`, since `EV(b_i, H̄)` is affine in `b_i`, the optimum is a corner solution:

```
b*_EV(H̄) = 1   if H̄ > θ_opt
           = 0   if H̄ < θ_opt
```

where `θ_opt = (1+α) / (M+1+α)`.

At `b_i = 0`: `EV = (1-H̄) · α`
At `b_i = 1`: `EV = H̄(M+1) - 1`

These are equal at `H̄ = θ_opt`, confirming the threshold.

### 3.2 Kelly Fraction Under Log-Utility Objective

For the proportional-ante episode structure (Section 5), the log-utility objective is more appropriate. With `b_i = ρ · C_i` (bet fraction ρ of current coffers):

```
Win:  C_i → C_i(1 + Mρ)
Lose: C_i → C_i · (1 + α)(1 - ρ)
```

Per-hand log-return:

```
g(ρ, H̄) = H̄ · log(1 + Mρ)  +  (1 - H̄) · [log(1+α) + log(1-ρ)]
```

Setting `dg/dρ = 0`:

```
H̄ · M / (1 + Mρ)  =  (1 - H̄) / (1 - ρ)
```

Solving (the `log(1+α)` term drops out because it has no ρ dependence):

```
ρ*(H̄) = max(0,  (H̄(M+1) - 1) / M)
```

**Kelly threshold:** `θ_k = 1/(M+1)` (where `ρ*` becomes positive)

Note that `θ_k < θ_opt` for any `α > 0`:
```
1/(M+1)  <  (1+α)/(M+1+α)
```
since `M+1+α < (1+α)(M+1) = M+1+αM+α` iff `0 < αM` which is always true.

**Implication:** The Kelly objective starts betting at lower `H̄` than the EV objective because it values the upside asymmetrically and accounts for the compounding effect. The `α` consolation does not affect the optimal fraction — it only affects the log-utility at `b=0`.

The world model needs to learn `ρ*(H̄) = max(0, (H̄(M+1)-1)/M)`: a continuous, piecewise-linear function of `H̄`. This is easier to learn than the binary EV policy (denser gradient signal).

---

## 4. Overgeneralization Analysis

### 4.1 Independent Equilibrium Threshold

Without coordination, each agent knows only `H_i` and must estimate `H̄`. For `N` agents with `H_j ~ Uniform[0,1]` for `j ≠ i`:

```
E[H̄ | H_i] = H_i/N + (N-1)/(2N)
```

Agent i bets when the expected marginal EV is positive:

```
E[H̄ | H_i] · (M+1+α) > (1+α)
```

Substituting and solving:

```
θ_ind(N) = N · θ_opt  -  (N-1)/2
```

**Check:**
- N=1: `θ_ind = θ_opt` ✓ (single agent has full information)
- N=2: `θ_ind = 2θ_opt - 1/2`
- N→∞: `θ_ind → -∞` (with many agents, individual hand strength is uninformative about H̄)

For `N=2, M=2, α=0.1`:  `θ_opt ≈ 0.355`,  `θ_ind ≈ 0.210`

Independent agents bet aggressively at `H_i > 0.210`, when the team threshold requires `H̄ > 0.355`.

### 4.2 The Overgeneralization Gap

The gap between the optimal threshold and the independent threshold:

```
Δθ(N) = θ_opt - θ_ind(N)
       = (N-1) · (1/2 - θ_opt)
       = (N-1)(M - 1 - α) / (2(M + 1 + α))
```

This is the "region of incorrect betting": when `H̄ ∈ [θ_ind, θ_opt]`, independent agents bet but the team shouldn't.

**Key properties of Δθ:**

```
∂(Δθ)/∂M = (N-1)(M+1+α+M-1-α) / (2(M+1+α)²)
           = (N-1) · 2M / (2(M+1+α)²)
           > 0
```
Gap increases with M (higher amplifier → stronger coordination incentive).

```
∂(Δθ)/∂α = (N-1)[-(M+1+α) - (M-1-α)] / (2(M+1+α)²)
           = (N-1) · (-2M) / (2(M+1+α)²)
           < 0
```
Gap decreases with α (higher consolation → safe play is more attractive → independent agents are less aggressive → gap shrinks).

```
∂(Δθ)/∂N = (M-1-α) / (2(M+1+α))  > 0  (when M > 1+α)
```
Gap increases with N (more agents → worse individual estimates of H̄ → larger coordination value).

### 4.3 Expected Value Under Each Regime

For `N=2`, `H̄` has the triangular distribution: `f(h) = 4h` for `h ∈ [0, 0.5]`, `f(h) = 4(1-h)` for `h ∈ [0.5, 1]`.

**Optimal EV** (knows `H̄`, uses `θ_opt`):

For `θ_opt < 0.5` (guaranteed when `M > 1+α`):

```
EV_opt = 4α ∫₀^{θ_opt} h(1-h) dh
       + 4 ∫_{θ_opt}^{0.5} h²(M+1) - h dh
       + 4 ∫_{0.5}^1 [h(M+1)-1](1-h) dh
```

Evaluating:
```
EV_opt = 4α(θ²/2 - θ³/3)
       + 4[(M+1)/24 - 1/8 - (M+1)θ³/3 + θ²/2]
       + (2M-1)/6
```
where `θ = θ_opt = (1+α)/(M+1+α)`.

**Independent EV** (each agent uses `θ_ind = 2θ_opt - 0.5`):

Per agent i, conditioning on their own H_i:

```
EV_ind = (1 - θ_ind) · [ ((θ_ind+2)/4)(M+1) - 1 ]
       + θ_ind        · [ α(3 - θ_ind)/4         ]
```

Derivation: `E[H̄ | H_i > θ_ind] = ((θ_ind+1)/2 + 0.5)/2 = (θ_ind+2)/4` and `E[H̄ | H_i ≤ θ_ind] = (θ_ind/2+0.5)/2 = (θ_ind+1)/4`, giving `E[1-H̄ | H_i ≤ θ_ind] = (3-θ_ind)/4`.

**Overgeneralization penalty:**

```
Π(M, α) = EV_opt - EV_ind
```

Numerical evaluation for N=2:

| M   | α   | θ_opt | θ_ind | Δθ    | EV_opt | EV_ind | Π      | Π/EV_opt |
|-----|-----|-------|-------|-------|--------|--------|--------|----------|
| 1.5 | 0.1 | 0.406 | 0.313 | 0.094 | 0.287  | 0.264  | 0.023  | 8.0%     |
| 2.0 | 0.1 | 0.355 | 0.210 | 0.145 | 0.587  | 0.534  | 0.054  | 9.1%     |
| 3.0 | 0.1 | 0.275 | 0.050 | 0.225 | 1.176  | 1.053  | 0.123  | 10.5%    |
| 2.0 | 0.0 | 0.333 | 0.167 | 0.167 | 0.611  | 0.546  | 0.065  | 10.6%    |
| 2.0 | 0.3 | 0.394 | 0.288 | 0.106 | 0.559  | 0.523  | 0.036  | 6.4%     |

**Reading the table:**
- Increasing M (higher win multiplier) increases both the absolute and relative overgeneralization penalty. The sweet spot is M=2 to M=3: the penalty is meaningful (~9-10%) without making the game so lopsided that agents always bet regardless.
- Increasing α (larger consolation) shrinks Π. High α makes the safe play (b=0) so attractive that independent agents become *more* conservative, closing the gap. Low α forces agents to bet more aggressively to overcome the zero-bet baseline.
- The recommended regime is **M=2, α=0.1**: large enough gap to learn from, small enough that the EV calculation remains tractable and the consolation doesn't dominate.

### 4.4 Overgeneralization in SeqComm Terms

The above gap is the *maximum* recoverable by perfect coordination. SeqComm recovers this by:

1. **Negotiation phase:** Agent with higher `H_i` wins the intention-value comparison → goes first
2. **Launching phase:** First mover bets based on `H_i`, second mover updates `E[H̄ | b_1, H_2]` and bets accordingly

The residual gap (how much SeqComm leaves on the table vs. perfect knowledge of `H̄`) depends on how informative `b_1` is about `H_1`. With a threshold policy where `b_1 ∈ {0,1}`, the second mover updates:

```
E[H_1 | b_1=1] = (θ_1 + 1)/2    →    E[H̄ | b_1=1, H_2] = ((θ_1+1)/2 + H_2)/2
E[H_1 | b_1=0] = θ_1/2          →    E[H̄ | b_1=0, H_2] = (θ_1/2 + H_2)/2... wait, 
                                        E[H_1 | b_1=0] = θ_1/2
```

This Bayesian update is exactly what the launching phase enables. With continuous bets (Kelly fraction), `b_1 = ρ*(H_1)` is an injective mapping, so the follower can invert and recover `H_1` exactly — full information sharing, zero residual gap.

---

## 5. Episode Structure

### 5.1 The Case for Proportional Ante

Under a **fixed ante** `b_0`:
- Optimal policy depends on `C` (when `C` is small, `b_0` is a large fraction and risk matters; when `C` is large, `b_0/C → 0` and the agent is effectively risk-neutral)
- The policy must learn `b*(H̄, C)` — a function of both hand strength and current wealth
- With parameter sharing, all agents run the same policy regardless of `C`, which breaks stationarity

Under a **proportional ante** `b = ρ · C`:
- The coffer process is a multiplicative random walk: `C_{t+1} = C_t · R_t` where `R_t` is the per-hand return ratio
- Per-hand log-return `g(ρ, H̄) = H̄ · log(1+Mρ) + (1-H̄) · [log(1+α) + log(1-ρ)]` is independent of `C`
- The optimal policy `ρ*(H̄)` is stationary — same at any wealth level
- Parameter sharing is valid: every agent's optimal action depends only on `H̄` (through hand strengths), not on absolute chip count

**Scale-invariance is sufficient for stationarity.** The proportional ante achieves this; the fixed ante does not. Therefore, proportional ante strictly dominates fixed ante in terms of policy learnability.

### 5.2 Why Episodic Reward Helps Learn Ordering

Consider the gradient of episode reward with respect to the ordering decision at hand `t`:

**Single-hand reward signal:**

```
∇_{order_t} R_t  =  EV(b_first*(H_{i*}), b_second*(H_{j*} | b_first)) - EV(uniform ordering)
```

This is small: roughly proportional to `Δθ · (M+1)`, the marginal value of one correct ordering.

**Episodic reward signal** (K hands with proportional ante):

Let `G_t = log(C_t/C_0) = Σ_{t'=1}^t g(ρ_{t'}, H̄_{t'})` be the log-growth up to hand t.

The ordering decision at hand `t` affects the log-growth at hand `t` directly, AND affects all future hand sizes (because `C_t` compounds). The total gradient:

```
∇_{order_t} G_K  =  ∇_{order_t} g_t  +  Σ_{t'>t} ∇_{order_t} g_{t'}
```

The second term is nonzero because `g_{t'} = H̄_{t'} · log(1 + Mρ · C_{t'})` and `C_{t'}` depends on all decisions before `t'`. Under proportional ante, the indirect effect compounds:

```
Σ_{t'>t} ∇_{order_t} g_{t'}  ≈  (K - t) · ∂g/∂ρ · ∇_{order_t} ρ_t
```

So the ordering gradient scales approximately as `O(K - t)` — early ordering decisions have much larger gradients than late ones. This gives a natural credit assignment structure: get the ordering right in the early hands of an episode, and the reward signal is amplified by all subsequent hands.

With per-hand reward, the effective gradient is `O(1)` regardless of when the ordering decision is made.

### 5.3 Episode Termination Condition

**Option A: Fixed K hands**

- Episode length is constant → policy gradient variance is bounded
- The gradient for ordering is `O(K)` total, but early decisions are weighted more (from the compounding argument)
- Weakness: agents who learn poor ordering policies don't face any harder episodes — no automatic curriculum

**Option B: Bankruptcy / Target threshold**

- Episode ends when `C < C_floor` (bankruptcy) or `C > C_target`
- Natural curriculum: poor ordering → frequent bankruptcies → short episodes early in training → many episodes → fast early learning
- As policy improves, episodes get longer → richer ordering signal
- Weakness: high variance in episode length → high variance policy gradient

**Option C: Hybrid (recommended)**

```
Terminate if:  C < C_floor  OR  C > C_target  OR  t = K_max
```

The expected episode length under optimal policy:

```
T_opt ≈ log(C_target / C_0) / E[g(ρ*(H̄), H̄)]
```

where `E[g] = ∫ g(ρ*(h), h) · f(h) dh` and f is the triangular distribution for N=2.

For `M=2, α=0.1`: `ρ*(h) = max(0, (3h-1)/2)` and the threshold is at `h = 1/3`.

```
E[g] = ∫_{1/3}^1 [h·log(1+2ρ*(h)) + (1-h)·log((1+α)(1-ρ*(h)))] · f(h) dh
     + ∫_0^{1/3} log(1+α) · f(h) dh
```

Numerically for M=2, α=0.1: `E[g] ≈ 0.087` (about 8.7% log-growth per hand). Under overgeneralization (wrong ordering, θ_ind policy): `E[g] ≈ 0.073`. The bankruptcy rate difference is measurable and provides a direct training signal for ordering quality.

**Setting `C_target / C_floor = 100`:** At 8.7% per hand, the optimal policy reaches target in ~53 hands. The overgeneralized policy takes ~63 hands and goes bankrupt more often. This ~19% episode length difference is a clean signal.

**Setting K_max:** `K_max = 2 × T_opt ≈ 100` hands per episode. Episodes that run to K_max are ones where agents aren't improving — use this as a "reset" to maintain learning velocity.

### 5.4 Recommended Configuration

```
M          = 2.0      # win multiplier: large coordination incentive, tractable math
α          = 0.1      # consolation fraction: meaningful safe-play but not dominant
N          = 2-4      # agents: 2 for analysis, 4 for demonstrating scaling
ρ_max      = 1.0      # proportional ante: full bet allowed (Kelly handles fraction)
C_0        = 100      # starting coffers (arbitrary since scale-invariant)
C_floor    = 10       # bankruptcy threshold (10% of start)
C_target   = 1000     # success threshold (10× start)
K_max      = 150      # max hands per episode
```

**What to measure:**
1. `Δθ_empirical`: difference between the hand strength threshold at which trained agents bet vs θ_opt — measures how close to optimal
2. **Ordering accuracy**: fraction of episodes where arg max H_i goes first — measures SeqComm's ordering quality
3. **Bankruptcy rate**: fraction of episodes ending at C_floor — decreases as ordering improves
4. **Episode length**: increases as ordering improves (compounding effect)
5. **EV gap**: `EV_opt - EV_empirical` per hand — overall performance gap

These five metrics collectively isolate the overgeneralization signal without confounding it with game-strategy learning.

---

## 6. Summary

The game's mathematics lock in the following structure:

```
Overgeneralization gap:  Δθ = (N-1)(M-1-α) / (2(M+1+α))
EV penalty:              Π ≈ 9% relative for M=2, α=0.1, N=2
World model function:    r̂ = b · [H̄(M+1+α) - (1+α)] + (1-H̄)α  [exact, bilinear]
Kelly fraction:          ρ*(H̄) = max(0, (H̄(M+1) - 1) / M)       [α-independent]
Ordering claim:          v_i > v_j iff H_i > H_j under perfect world model
Noise tolerance:         σ/√N < (1+α)/(M+1+α)
```

The proportional-ante, hybrid-termination episode structure is optimal for learning because:
1. Scale-invariance → stationary policy → compatible with parameter sharing
2. Episodic reward → ordering gradient scales as O(K-t) vs O(1) for per-hand reward
3. Bankruptcy condition → automatic curriculum that accelerates early learning
