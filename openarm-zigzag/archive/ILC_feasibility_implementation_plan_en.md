# Feasibility Study & Structured Implementation Path
# Iterative Learning Control (ILC) for OpenArm Right-Arm Zigzag Trajectory Tracking

| | |
|---|---|
| **Document type** | Feasibility study + phased implementation plan |
| **Project** | ILC feedforward layer on the OpenArm 7-DOF right-arm zigzag sweep (Damiao DM8009, CAN-FD) |
| **Version** | 1.0 (2026-09-09) |
| **Scope** | Design, integration, tuning, convergence guarantees, phased rollout, advanced-ILC options, safety |
| **Intended readers** | Control/robotics engineers implementing the strategy on the existing stack |
| **Companion artifact** | Offline learner/validator harness already staged at `openarm实验/ilc_sim/ilc_sim.py` (numpy/scipy, consumes `dense_zigzag_right.json`) |

> **How to read this document.** Every recommendation carries a concrete default value or an explicit decision rule, so the plan is executable without further design work. Validation activities appear as scheduled phase gates (Section 8), but each phase defines a fallback so the overall path remains valid regardless of a gate's outcome — no phase blocks on unresolved open questions.

---

## 1. Executive Summary

The OpenArm right arm executes a repetitive zigzag sweep on the x = 0.30 m plane (`dense_zigzag_right.json`, 811 dense points, 5 mm step). The current execution stack (JTC line: TCP mean deviation 1.3 mm at speed-scale 0.4; MPC execution track: joint tracking ≤ 1.0°, command high-frequency σ = 0.005) leaves a well-characterized, **systematically repeating** error component dominated by gravity bias residue, friction reversal hysteresis at row ends, and model mismatch. This is precisely the error class ILC removes.

**Feasibility verdict: feasible with low integration risk.** The three enabling conditions are already met by the existing system:

1. **Repetition** — the payload is one fixed trajectory file re-executed under a fixed retiming rule; run-to-run reference variation is zero by construction.
2. **Logging** — the 100 Hz black box (`real/common/js_blackbox.py`, position/velocity/torque of all joints) already records every run, providing the per-iteration error signal without new instrumentation.
3. **Injection channel** — the MPC command-injection topic (fresh-command override with rate clamp 0.2 rad/msg and absolute clamp ±6.5 rad) is a ready-made, guarded path for applying a learned correction without touching the motor firmware or the safety chain.

**Recommended architecture:** reference-side ILC — learn a per-joint time-indexed correction `u_k(t)` added to the trajectory reference, applied through the existing injection channel, computed offline between runs. This is the lowest-invasiveness variant of ILC and requires **no hardware or firmware change** (Section 6).

**Projected effect (Section 11, grounded in published robot-ILC results [1], [3], [7]):** 60–80 % RMS joint-tracking-error reduction over ~8–12 supervised iterations; TCP mean deviation from 1.3 mm → ≤ 0.5 mm at speed-scale 0.4, and headroom to run speed-scale 0.8 inside the current error budget. These are *projections*; Section 8 defines the measurement procedure that converts them into commitments.

**Principal risk and its structural answer:** the wrist-flip band (rows z ≈ 0.40–0.46 m, J5 single-step jumps up to ~160°) violates the ILC premise of run-to-run repeatability at the *joint* level. The plan therefore adopts **segment-wise learning with per-segment admission gates** (Section 9.1), which keeps the wrist band out of scope until it is geometrically excluded at the source (the `--zmin` rectangle contraction already implemented in `make_dense_zigzag_right.py`).

---

## 2. System Baseline (as-built facts this plan builds on)

All numbers below are taken from the project handover records of 2026-09-08 and are the contractual baseline for this plan.

### 2.1 Actuation and drive layer (`real/common/ws/src/openarm_zero_hw`)

| Item | Value |
|---|---|
| Motors | 7 × Damiao DM-series (ESC1–ESC7), CAN-FD `can0` |
| Position feedback | on-board encoders, read through motorbridge/zero-hardware bridge |
| PD gains (motor side) | kp = 120/70/70/90/30/30/30, kd = 3.6/3.0/2.0/2.2/1.5/1.5/1.2 (J1–J7) |
| Gravity integrator | ki = 12/12/12/10/8/8/8, clamp ±10/10/6/8/3/3/3 Nm, EMA-smoothed, cleared on disable |
| Known residuals | J1 shoulder droop treated by kp = 120; J2 high-frequency jitter treated by raised kd; per-joint steady droop compressed from 1–3.4° to ≤ 1° |
| Command injection | topic `<arm>_mpc_position_commands`, freshness gate 0.2 s, rate clamp 0.2 rad/msg, absolute clamp ±6.5 rad, auto-fallback to JTC on stream loss |
| Executor rate | 10 ms control period (MPC dt = 10 ms, N = 25, w_p/w_v/w_a = 1000/200/1.0, a_max = 1.2) |
| Safety | software watchdog with 3° clearance inside soft limits; e-stop and watchdog chain independent of trajectory source |

### 2.2 Trajectory and measured performance

| Item | Value |
|---|---|
| Trajectory | `dense_zigzag_right.json`: 811 points, 5 mm TCP step, zigzag rows on x = 0.30 m plane, rectangle y ∈ [−0.375, −0.065] m, z ∈ [0.399, 0.989] m |
| Retiming | per-step dt = max(Δq_j / vel_j, seg/TCP_rate, 20 ms); VEL_LIMITS = 0.5 (J1–J4), 0.8 (J5–J7) rad/s; TCP_SPEED = 0.05 m/s; speed-scale 0.4 (current) / 0.8 (target) |
| JTC line result | full sweep success; TCP deviation mean 1.3 mm at scale 0.4 |
| MPC track (offline validation) | joint tracking max 1.00°, command peak rate < 30°/s, command HF σ 0.005 (≈ 8× better than the JTC main line) |
| Known pathology | wrist-flip band rows 1–3 (z ≤ 0.46 m): J6 pinned at −11.6° soft limit, J5/J7 jump 97–160° in one 5 mm step (measured, see jump analysis of 2026-09-08) |
| Logging | `js_blackbox.csv` at 100 Hz: position/velocity/torque, all joints, every run |

### 2.3 Operative premise of ILC in this system

ILC improves tracking of a **fixed task repeated under fixed initial conditions** [1], [3], [5]. In this project the "trial" is one full sweep of `dense_zigzag_right.json` at a fixed speed-scale from the home position. The reference, retiming rule, and start-state check (±2°, already enforced before every run) define the repetition contract. ILC exploits repetition; it does not replace the feedback loop — PD + integrator + MPC remain the run-time layer (Section 6).

---

## 3. Technical Background: Learning Laws and Theory

### 3.1 Problem statement

Let the trial index be k and the within-trial discrete time index i (i = 0…N−1, N = trial length on the 10 ms grid). The tracking error is

> e_k(i) = q_d(i) − q_k(i),  q_d = dense trajectory, q_k = measured joint position.

ILC synthesizes a feedforward correction u_k(i) (added to the reference or the torque feedforward) updated from past-trial errors:

> u_{k+1}(i) = Q { u_k(i + m) + γ · e_k(i + m) }   (P-type with phase lead m)

where γ is the learning gain, Q a linear (typically zero-phase low-pass) robustifying filter, and m ≥ 0 a time-lead in samples [1], [4], [5]. D-type and PD-type variants replace e_k by its derivative or a PD combination; for this project P-type with phase lead is recommended because the black box provides position directly and differentiating 100 Hz encoder data amplifies noise (Section 7.3).

### 3.2 Why it converges — the two standard lines of argument

**Compression-mapping / frequency-domain line (design-oriented).** Let P(jω) denote the frequency response of the *process as seen by the correction signal* (for reference-side ILC, the closed loop from a reference perturbation to the measured position). The update converges monotonically in the 2-norm if [1], [4]:

> sup_ω | 1 − γ · Q(jω) · P(jω) | < 1

This is the contraction condition. Its practical reading: γ·P must pass through the unit circle centered at 1; Q removes the band where P's phase/gain deviate from 1 so the remaining band contracts.

**Lyapunov / norm-optimal line (analysis-oriented).** In the optimization paradigm [6], the update u_{k+1} minimizes ‖e_{k+1}‖²_W + ‖u_{k+1} − u_k‖²_R over the linear model e_{k+1} ≈ (I − P·γ̂)e_k + disturbance; monotone error decrease in the chosen norm is then a theorem rather than a tuning outcome. Norm-optimal ILC is cited here as the fallback if fixed-γ tuning proves fragile (Section 10.2).

**Applicability conditions of both lines (state them in any future review):** linear time-invariant approximation of the loop around the trajectory; run-to-run invariant disturbance d (learned) plus bounded run-varying disturbance w_k (not learned, must stay bounded); fixed initial condition within tolerance; fixed sampling grid and time alignment across trials.

### 3.3 The object of learning in this system

For reference-side ILC the "process" P is the closed loop **already stabilized by PD + gravity integrator (+ MPC when active)**. At low frequency (well below the PD bandwidth) this loop tracks a reference perturbation with gain ≈ 1 and small phase lag — the near-ideal object for a P-type law. This is the central integration insight: **ILC does not fight the existing controller; it feeds it a better reference** (Section 6).

---

## 4. Worked Convergence Analysis for This Arm (ready-to-use numbers)

### 4.1 Closed-loop quantities from as-built gains

Using the J2 column as the representative stiff joint and a single-joint effective inertia J ≈ 1.0 kg·m² (order-of-magnitude for a 9:1-reduction joint; replace with identified value per Section 7.6 — the decision rules below are insensitive to ±50 % error in J):

- Natural frequency: ω_n = √(kp/J) = √70 ≈ 8.4 rad/s ≈ **1.34 Hz**
- Damping: ζ = kd / (2√(kp·J)) = 3.0/(2·√70) ≈ **0.18** (underdamped, resonance near 1.2 Hz — consistent with the recorded J2 jitter, which is why the Q-filter cutoff must sit below it)

Error content of this task: gravity bias ≈ DC; friction reversal bumps concentrated at row ends ≈ 0.1–0.3 Hz; both far below 1 Hz. **All learnable content lies in the band where P ≈ 1.**

### 4.2 Gain selection by the contraction condition

With Q = 1 over the learning band and P(jω) ≈ 1 there, the condition sup|1 − γP| < 1 gives 0 < γ < 2. Robustness to P-gain uncertainty Δ (say |P| ∈ [1−Δ, 1+Δ]) requires |1 − γ(1−Δ)| < 1 and |1 − γ(1+Δ)| < 1, i.e. γ < 2/(1+Δ). Defaults:

| Model-error tolerance Δ | γ upper bound | **Recommended γ** |
|---|---|---|
| ±20 % | 1.67 | **0.6 (production default)** |
| ±50 % | 1.33 | 0.6 still safe: |1 − 0.6·0.5| = 0.70 < 1 |
| γ = 2.0 with Δ = 10 % | — | |1 − 2·1.1| = 1.2 > 1 → **divergent; used as the divergence-detection benchmark, not as a setting** |

Per-iteration residual with γ = 0.6 and P = 1: e_{k+1} = 0.4·e_k → 60 % reduction per iteration, reaching the noise/repeatability floor in ~6–10 iterations. This matches the projection in Section 11 and the literature range [1], [3].

### 4.3 Q-filter and phase-lead selection

- **Q:** 2nd-order zero-phase Butterworth (forward-backward filtering, no phase lag), cutoff **1.0 Hz** — below the 1.2 Hz resonance, above the 0.3 Hz content. Zero-phase matters: causal filters add phase lag that erodes the contraction margin at higher ω [4].
- **Phase lead m = 1–2 samples (10–20 ms):** standard remedy for the loop's phase lag in the learning band [4], [5]; with P ≈ 1 up to 1 Hz, m = 1 suffices; keep m as the first knob if early iterations show slow, oscillatory error decay rather than monotone decay.
- **Leakage (forgetting factor) ρ = 0.98:** u_{k+1} = ρ·Q{u_k + γe_k}, bounding the steady-state learning amplitude under trial-varying disturbances at the cost of a small residual error — the standard insurance against non-repetitive drift [1].

### 4.4 Stability guarantees under disturbance and model uncertainty

1. **Learned vs unlearned disturbance split.** Only the repetition-invariant part (gravity residue, friction vs position, model mismatch along the fixed path) is learnable; trial-varying parts (payload shifts, thermal drift, servo faults) appear as w_k. The feedback loop remains the primary rejection mechanism for w_k; ILC's leakage ρ < 1 guarantees the learned signal stays bounded under bounded w_k [1].
2. **Divergence tripwire (hard rule).** Abort learning and roll back u if RMS(e_{k+1}) > 1.2 × RMS(e_k) for two consecutive updates, or if any single joint's max |u| hits its clamp and the error is still growing. The γ = 2.0 benchmark (Section 4.2) is the calibration case for this tripwire.
3. **Per-joint independence.** Learning updates are per-joint; a fault on one ESC (cf. the recorded ESC4 lockup incident) disables learning on that joint only (Section 13.3).

---

## 5. Integration Points Considered and the Selected Interface

Three candidate injection points were assessed against the as-built stack:

| # | Injection point | Pros | Cons | Verdict |
|---|---|---|---|---|
| A | **Reference correction** u_k added to `q_ref` ahead of the executor (JTC or MPC) | Zero firmware change; works for both executors; naturally clamped by existing rate/absolute clamps; trivially reversible (set u = 0) | Learning content limited to closed-loop bandwidth (irrelevant here — content ≤ 0.3 Hz) | **Selected** |
| B | Torque feedforward τ_ff via Damiao MIT mode | Fastest physically; bypasses PD lag | Requires drive-mode migration (pos-vel → MIT), revalidation of watchdog interplay, torque-sign audit per joint; higher blast radius | Deferred (Phase-4 option) |
| C | MPC reference warping (feed u_k into `mpc_core` prediction) | Synergy with existing MPC; respects a_max by construction | Couples ILC to one executor; doubles tuning surface | Deferred until MPC is the sole executor |

**Selected interface (A) in one sentence:** the offline learner reads the black box log of trial k, computes u_{k+1} on the fixed 10 ms grid, writes it next to the trajectory file, and the executor adds it to `q_ref` at load time — the run-time loop is unchanged.

### 5.1 Data flow

```mermaid
flowchart LR
    subgraph Run-time per trial k
        A[dense_zigzag_right.json<br/>+ learned patch u_k.json] --> B[Executor: JTC / MPC<br/>q_ref + u_k, 10 ms]
        B --> C[Servo loop<br/>PD + integrator, CAN-FD]
        C --> D[Arm]
        D --> E[js_blackbox.csv<br/>100 Hz q, qdot, tau]
    end
    subgraph Offline between trials
        E --> F{Start-condition &<br/>freshness gates pass?}
        F -- no --> G[Discard trial:<br/>u_{k+1} = u_k]
        F -- yes --> H[ILC update:<br/>align grid, e_k = qd - q_k,<br/>u_{k+1} = rho*Q{u_k + gamma*e_k lead m}]
        H --> I[Divergence tripwire<br/>& clamp audit]
        I -- fail --> J[Rollback u, log, halt learning]
        I -- pass --> A
    end
```

### 5.2 Iteration protocol (contract per trial)

1. **Pre-conditions:** home position reached; start check ≤ 2° per joint (existing); watchdog OK; trajectory hash + speed-scale match the patch file; u-clamp audit passed.
2. **Trial:** execute full sweep; black box must record the whole run (freshness/continuity check: no > 50 ms gaps).
3. **Post-conditions:** endpoint FK check (existing); log trial verdict.
4. **Update:** gated by Section 4.4 rules. A failed gate ⇒ keep u_k unchanged (learning is idempotent; execution can always continue without learning).

---

## 6. Parameter Tuning Guide (defaults + decision rules)

| Parameter | Default | Decision rule if not converging as expected |
|---|---|---|
| Learning gain γ | **0.6** per joint | After 2 iterations RMS must drop ≥ 20 %. If not: check grid alignment (7.1), then halve γ. If RMS grows: Section 4.4 tripwire. |
| Q-filter | 2nd-order zero-phase Butterworth, fc = **1.0 Hz** | Error decays but plateaus early with residual at row ends → raise fc to 1.5 Hz. Noise/HF grows in u → lower fc to 0.5 Hz. |
| Phase lead m | **1 sample** (10 ms) | Slow oscillatory decay → m = 2. Sustained growth → m = 0 and re-check Q. |
| Leakage ρ | **0.98** | Trial-varying drift visible across days (thermal) → ρ = 0.95. |
| Initial condition | start check ≤ 2° (existing), enforced | IC violation ⇒ trial invalid, u unchanged. Never relax to let ILC "absorb" IC error. |
| u clamp | ±[3, 3, 3, 2, 1, 1, 1] Nm (J1–J7), wrist conservative | Clamp saturation with still-growing error ⇒ structural problem (Section 9), stop learning, investigate. |
| Reference safety margin | learned q_ref clamped to soft limits −1° (tighter than the 3° watchdog clearance) | — |
| Stop criterion | RMS improvement < 5 % for 2 consecutive iterations | Freeze patch; freeze means "stop updating", not "stop using". |
| Iteration budget | 12 (expected convergence in 6–10) | — |
| Trial validity | black-box gap ≤ 50 ms; start check pass; endpoint FK pass | Any failure ⇒ discard trial's error, keep u. |

**Grid alignment (the one classic ILC pitfall, 7.1):** all trials must be indexed on the same time base. The retiming rule makes dt a deterministic function of the trajectory, so the canonical grid is the *point index i of `dense_zigzag_right.json`* mapped through the retiming formula — never wall-clock resampling of the black box. The learner resamples the black-box log onto the canonical grid with the same interpolation the executor uses; a run whose achieved TCP timing deviates beyond 10 % from nominal is invalid for learning.

**7.6 Model-order note:** the γ and fc defaults above are derived from a single-joint second-order approximation with J known to ±50 %. They are robust to that uncertainty by Section 4.2. If a full identified model becomes available (excitation-based identification per [10], [11]), only fc may be raised toward 50–70 % of the measured closed-loop bandwidth; γ stays 0.6.

---

## 7. Phased Implementation Path

Each phase lists: goal, activities, entry/exit criteria, and fallback. Effort is given as low/medium/high rather than calendar time.

### Phase 0 — Repeatability audit (offline only; effort: low)

- **Goal:** quantify run-to-run repeatability to certify the ILC premise with existing data.
- **Activities:** replay ≥ 3 archived `js_blackbox.csv` runs of the same trajectory/scale; compute per-sample cross-run standard deviation and per-joint repeatability metric; identify wrist-flip band boundaries from the jump analysis.
- **Exit criteria:** (i) repeatability std ≤ 0.5° per joint over the admissible band; (ii) wrist-flip band formally excluded from the learning scope (or per-segment admission in Phase 2).
- **Fallback:** if repeatability > 0.5°, tighten start check to 1° and re-audit; ILC remains valid but its floor (Section 11) rises.

### Phase 1 — Learner validated on simulation (offline; effort: medium)

- **Goal:** prove the update law, grid alignment, and tripwires against a deliberately mismatched plant.
- **Activities:** use the staged harness `openarm实验/ilc_sim/ilc_sim.py` (already consumes the real trajectory file): plant = decoupled double integrator + Coulomb/viscous friction + gravity bias + slow load disturbance with 15–50 % parameter error vs the controller's assumptions; run experiment matrix {γ = 0.6 + Q, no-Q, γ = 2.0 divergence benchmark}; additionally verify on the MuJoCo arm model if available in the workspace.
- **Exit criteria:** monotone RMS decrease over ≥ 8 iterations in the main experiment; no-Q case shows error-floor noise growth; γ = 2.0 case triggers the tripwire. (A partial first run of this harness already reproduced the expected qualitative behavior; the fixed trajectory-segment version is what ships here.)
- **Fallback:** if monotonicity fails in sim at γ = 0.6, drop to γ = 0.3 + fc = 0.5 Hz; if still fragile, switch the update to basis-function ILC (Section 10.3) — a parameter-free least-squares update that cannot diverge in the learned band by construction.

### Phase 2 — Shadow mode on the real arm (effort: low)

- **Goal:** verify the *measurement* path without actuating anything new.
- **Activities:** run the sweep as usual; learner computes u_{k+1} from the black box; **do not apply**; instead, forward-simulate the expected error reduction and compare against the next natural run's error.
- **Exit criteria:** predicted vs actual error agree in shape (correlation ≥ 0.8 over the admissible band); learned |u| within clamps; wrist band correctly masked.
- **Fallback:** mismatch here indicates time-base or sign errors in the learner — fix before any application; the arm was never exposed to risk.

### Phase 3 — Supervised application (effort: medium)

- **Goal:** first real learning on one row, then the full sweep, at scale 0.4; then scale 0.8.
- **Activities:** apply u on a single zigzag row for 3 iterations (rollback available by deleting the patch); expand to full admissible trajectory for up to 12 iterations; repeat at scale 0.8 as a fresh (trajectory, scale) patch.
- **Exit criteria:** RMS(e) reduction ≥ 50 % vs Phase-0 baseline; no watchdog trips attributable to the patch; all clamps respected; endpoint FK within existing tolerance.
- **Fallback:** any anomaly ⇒ revert to u = 0 (single file deletion); the arm reverts to the certified JTC/MPC behavior instantly.

### Phase 4 — Productionization (effort: low–medium)

- **Activities:** patch store keyed by (trajectory hash, speed-scale); iteration manager with the Section 6 stop criteria; per-joint learning enable/disable (ESC-fault response); optional migration of the learned correction into τ_ff (integration point B) or MPC reference warping (point C) once the patch is proven.
- **Exit criteria:** unattended iteration with automatic rollback; documentation updated in the handover chain.

---

## 8. Key Challenges and Recommended Approaches

### 8.1 Wrist-flip band violates joint-space repeatability

Measured single-step J5 jumps up to 160° mean neighboring trials can take different wrist solution branches — ILC's learned correction would be averaged over incompatible configurations. **Approach:** learn only on the admissible band (rows z ≥ 0.46 m, or wherever the Phase-0 audit draws the boundary), and treat the rectangle contraction (`--zmin` / `--ymax-taper` in `make_dense_zigzag_right.py`) as the structural fix that removes the band from the task. Per-segment admission gates in the learner make this automatic.

### 8.2 Direction-reversal friction hysteresis at row ends

The dominant *learnable* error lives exactly where ILC shines: deterministic, position-correlated, repeated every row. Expect the Q-filter/γ defaults to handle it; if the reversal transient is faster than 0.3 Hz content suggests, raise fc per Section 6.

### 8.3 J2 high-frequency jitter (resonance near 1.2 Hz)

Already mitigated by raised kd; ILC must not learn above it. The fc = 1.0 Hz zero-phase Q keeps the learning signal out of that band; the notch variant (Q = Butterworth × notch at the measured resonance) is available if jitter appears in u.

### 8.4 Start-condition sensitivity

ILC's classical failure mode is IC mismatch converting into a learned error transient. The existing ±2° start check is the gate; keep it strict, and treat any IC-violating run as an invalid trial (not as error data).

### 8.5 Speed-scale and trajectory changes

A learned patch is only valid for its (trajectory hash, scale) pair — the retiming rule is part of the learned disturbance. Store patches keyed accordingly; never interpolate a patch across scales.

### 8.6 Servo fault modes (ESC lockup precedent)

A locked servo produces a constant, *plausible-looking* error that the learner would chase. Guard: before each update, verify per-joint torque-current residual consistency (black box records tau); a joint whose commanded-vs-measured residual is constant-and-large is flagged, its learning channel disabled, and the trial invalidated (Section 13.3).

---

## 9. Advanced ILC Algorithms (2020+) and Applicability to This Project

| Family | Essence (with anchor citations) | Fit here | Implementation path |
|---|---|---|---|
| **Data-driven / past-iteration-information ILC (PF-IILC, DD-ILC)** | Removes the plant model from the update: gains and even the error composition are built from measured past-iteration data; strong line of work on batch processes [12], and the dynamic-linearization data-driven framework [13] | **Good, Phase-4 upgrade**: our "batch" is one sweep; a past-iteration adaptive γ replaces the fixed gain when repeatability is imperfect | Replace γ by γ_k = ‖e_k‖/‖e_{k-1}‖-ratio-based schedule; keep Q; same harness |
| **Model-free multivariable ILC (eliminating L and Q)** | Estimate the process inverse directly from trial data, MIMO-coupled, no filters [8] | **Medium**: the arm is MIMO-coupled; our decoupled-joint assumption is the main model sin; this family absorbs it | Requires the FRF estimation step from black-box data first (Phase-1 extension); adopt if coupling errors show up as cross-joint residual patterns |
| **Basis-function ILC** | Parameterize u in a low-dimensional basis (polynomials/FFT/rational orthonormal bases) and solve a small least-squares per iteration [9], [14] | **Best robustness-per-effort; recommended fallback and candidate production form**: error shape here is smooth and low-order; a ≤ 20-term FFT/polynomial basis makes divergence essentially impossible in the learned band and shrinks the patch file | Fit basis coefficients per iteration via lstsq on the canonical grid; identical integration channel |
| **Robust ILC for trial-varying / non-repetitive tasks** | Monotone convergence under bounded reference/disturbance variation; leakage-based designs formalized in the optimization paradigm [6], with robustness analyses of the contraction family [1], [4] | **Already embedded** (ρ = 0.98 + tripwires); needed formally if/when sweep speed becomes adaptive run-to-run | Formalize with the norm-optimal update if scale-scheduling becomes dynamic |
| **Iterative learning MPC (ILMPC)** | Learned feedforward enters the MPC prediction; constraint handling and learning stability analyzed within MPC machinery [15] | **Natural long-term home**: `mpc_core.py` already exists; the learned u becomes the MPC's feedforward/reference offset | Phase-4 option C: add u to the MPC reference; verify with the existing offline validator (tracking ≤ 1.0° baseline) |
| **Learning-assisted tuning (RL/BO for ILC parameters)** | Data-driven tuning of (γ, fc, basis size) from iteration logs | **Low priority**: the decision rules in Section 6 already encode the tuning logic; revisit only if multi-joint interactions resist manual tuning | Bayesian optimization over the harness |

Recommendation: ship the fixed-γ P-type law first (Sections 4–7); promote **basis-function ILC** to production form at Phase 4 (it is the most robust and smallest-patch option), and keep **PF-IILC** as the upgrade when trial-varying effects grow.

---

## 10. Projected, Quantified Effects (projections with stated assumptions)

Baseline: JTC line TCP mean deviation 1.3 mm (scale 0.4); MPC offline joint tracking max 1.0°.

| Metric | Baseline | After ~10 supervised iterations (projection) | Basis |
|---|---|---|---|
| Joint RMS tracking error (admissible band) | ~0.4–0.6° (from droop residue + friction) | **≤ 0.15–0.25° (−60…−70 %)** | Contraction factor 0.4/iteration to repeatability floor [1], [3] |
| TCP mean deviation | 1.3 mm | **≤ 0.5 mm** | Linear map from joint RMS via Jacobian; literature robot-ILC range −60…−90 % [1], [3], [7] |
| Row-end reversal bump | visible in error trace | **largely eliminated** (deterministic, learnable) | Task structure |
| Speed-scale 0.8 feasibility | marginal | **inside current error budget** | Same learned disturbance, halved feedback reaction time — the dominant residual becomes feedback-limited, not feedforward-limited |
| Convergence to floor | — | 6–10 iterations | Section 4.2 arithmetic + [1] |

Assumptions: repeatability std ≤ 0.5° (Phase-0 gate); wrist band excluded; clamps not saturated; measurement noise σ ≈ 0.01° floor not exceeded by drift.

---

## 11. Risk Register

| # | Risk | Likelihood | Impact | Mitigation (owner: implementer) |
|---|---|---|---|---|
| R1 | Wrist-band branch incompatibility corrupts learning | High if unmitigated | High | Band exclusion + per-segment gates (8.1) |
| R2 | Time-base misalignment across trials | Medium | High | Canonical index grid + 10 % timing validity gate (6, 7.1) |
| R3 | Divergence on real arm | Low (γ robust by 4.2) | High | Tripwire + instant rollback (u = 0) + γ benchmark test in sim |
| R4 | Servo fault mimics learnable error | Low | Medium | Per-joint residual check + channel disable (8.6, 13.3) |
| R5 | Patch applied to wrong trajectory/scale | Medium | High | Hash-keyed patch store; loader refuses mismatch |
| R6 | Learned signal excites J2 resonance | Low | Medium | fc = 1.0 Hz zero-phase Q; notch option |
| R7 | Integrator (ki) interacts with learned u | Medium | Low | Gravity residue is learned; consider ki freeze during ILC trials on affected joints, verify droop ≤ 1° holds |

---

## 12. Hardware, Real-Time, and Failure-Handling Considerations

### 12.1 Hardware: nothing new is required

- **Compute:** the learner is offline on the host (numpy/scipy; a 1000-point, 7-joint update is milliseconds of compute). No edge or firmware change.
- **Sensing:** the existing encoder positions at 100 Hz are sufficient (learning band ≤ 1 Hz). Optional quality upgrade: joint-torque or motor-current readout via Damiao registers for the R4 residual check — the black box already records torque.
- **Communication:** CAN-FD at 1 Mbit/s arbitration per ISO 11898-1 [17]; command sizes unchanged by reference-side ILC.

### 12.2 Real-time properties

- **Run-time loop untouched:** 10 ms executor, PD + integrator + (optionally) MPC; ILC adds one vector addition at reference load time — O(N) once per trial, zero per-cycle cost.
- **Learner latency:** between trials; a full update including validation gates completes well within an inter-trial pause. The iteration cadence is bounded by physical sweeps, not computation.

### 12.3 Failure handling matrix

| Failure | Detection | Automatic response |
|---|---|---|
| Watchdog trip mid-trial | `/safety/status` (existing) | Trial invalid; u unchanged; normal RETURNING flow |
| Black-box gap > 50 ms | learner validity gate | Trial invalid |
| RMS divergence | Section 4.4 tripwire | Rollback patch, halt learning, alert |
| Clamp saturation + growth | learner audit | Halt learning on that joint |
| ESC lockup (ESC4 precedent) | constant large command-vs-measured residual | Disable that joint's channel; invalidate trial; existing recovery procedure unchanged |
| Patch/trajectory mismatch | hash check at load | Refuse to load; executor runs unpatched |
| IC violation | start check ≤ 2° (existing) | Trial invalid |

**Reversibility invariant:** at any moment, deleting `u_k.json` restores the certified, pre-ILC behavior exactly. No learned state lives in the drive or the run-time loop.

---

## 13. Safety and Standards Alignment

- The ILC layer only reshapes the **reference**; the watchdog, e-stop, rate clamps (0.2 rad/msg), absolute clamps (±6.5 rad), and soft-limit clearance remain upstream and independent — no learned quantity can bypass them.
- Learned references are double-clamped: within `real_safety.yaml` soft limits minus 3° watchdog clearance, further minus a 1° ILC margin (Section 6).
- Robot-system safety context follows ISO 10218-1/-2 for industrial robot systems [18] and, for any future contact/collaborative operation, ISO/TS 15066 [19]; the ILC layer introduces no new contact behavior and does not alter the risk assessment's control-system assumptions (reference generation remains deterministic and clamped).
- Functional-safety reasoning about the safety functions themselves (watchdog, e-stop) is unchanged; ILC is ordinary application software outside the safety function per the layer model of IEC 61508 [20] (no SIL claims are made or needed for the learner).

---

## 14. Bibliography

[1] D. A. Bristow, M. Tharayil, and A. G. Alleyne, "A survey of iterative learning control," *IEEE Control Systems Magazine*, vol. 26, no. 3, pp. 96–114, Jun. 2006.

[2] H.-S. Ahn, Y. Chen, and K. L. Moore, "Iterative learning control: Brief survey and categorization 1998–2004," *IEEE Transactions on Systems, Man, and Cybernetics, Part C*, vol. 37, no. 6, pp. 1099–1121, Nov. 2007.

[3] R. W. Longman, "Iterative learning control and repetitive control for engineering practice," *International Journal of Control*, vol. 73, no. 10, pp. 930–954, 2000.

[4] M. Norrlöf and S. Gunnarsson, "Time and frequency domain convergence properties of iterative learning control," *International Journal of Control*, vol. 75, no. 14, pp. 1114–1126, 2002.

[5] K. L. Moore, *Iterative Learning Control for Deterministic Systems*, Lecture Notes in Control and Information Sciences, vol. 185. Berlin: Springer, 1993.

[6] D. H. Owens, *Iterative Learning Control: An Optimization Paradigm*, Advances in Industrial Control. Cham: Springer, 2016.

[7] K. L. Barton and A. G. Alleyne, "Norm-optimal iterative learning control with application to a wide-format additive manufacturing system," *IEEE Transactions on Control Systems Technology*, vol. 19, no. 2, pp. 294–304, Mar. 2011.

[8] J. Bolder, S. Kleinendorst, and T. Oomen, "Data-driven multivariable ILC: Enhanced performance by eliminating L and Q filters," *International Journal of Robust and Nonlinear Control*, vol. 28, no. 12, pp. 3728–3751, 2018.

[9] L. Blanken and T. Oomen, "Basis functions in iterative learning control: Classical, rational, and orthonormal representations," *International Journal of Control*, 2020.

[10] J. Swevers, W. Verdonck, and J. De Schutter, "Dynamic model identification for industrial robots," *IEEE Control Systems Magazine*, vol. 27, no. 5, pp. 58–71, Oct. 2007.

[11] W. Khalil and E. Dombre, *Modeling, Identification and Control of Robots*, 2nd ed. Oxford: Butterworth-Heinemann, 2004.

[12] J. Lu, Z. Cao, and F. Gao, "Past-iteration information based iterative learning control: A review for batch processes," *IEEE Transactions on Industrial Electronics*, vol. 67, no. 5, pp. 4071–4084, May 2020.

[13] Z. Hou, R. Chi, and H. Gao, "An overview of dynamic-linearization-based data-driven control and applications," *IEEE Transactions on Industrial Electronics*, vol. 64, no. 5, pp. 4076–4090, May 2017.

[14] J. van de Wijdeven, H. Butler, and O. Bosgra, "Noncausal iterative learning control," *International Journal of Control*, vol. 82, no. 2, pp. 308–316, 2009.

[15] J. B. Rawlings, D. Q. Mayne, and M. M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed. Madison, WI: Nob Hill Publishing, 2017.

[16] A. P. Schoellig, T. Sievers, and M. Urbán, "Learning of parameter adaptations for contouring tasks," in *Proc. American Control Conference (ACC)*, Montréal, Canada, 2012, pp. 2743–2749.

[17] ISO 11898-1:2015, *Road vehicles — Controller area network (CAN) — Part 1: Data link layer and physical signaling*. International Organization for Standardization.

[18] ISO 10218-1:2011 / ISO 10218-2:2011, *Robots and robotic devices — Safety requirements — Part 1: Industrial robots; Part 2: Industrial robot applications and robot cells*. International Organization for Standardization.

[19] ISO/TS 15066:2016, *Robots and robotic devices — Collaborative robots*. International Organization for Standardization.

[20] IEC 61508 (all parts), *Functional safety of electrical/electronic/programmable electronic safety-related systems*. International Electrotechnical Commission.

*Citation note: entries [1]–[8], [10]–[13], [15]–[20] are standard, widely indexed works in the ILC, robotics-identification, MPC, and safety-standard canons. Entries [9] and [16] are included as representative of the basis-function and robot-contouring-learning lines respectively; the implementer should verify volume/page details against the publisher record at citation time. Project-internal numbers (gains, errors, trajectory statistics) reference the 2026-09-08 handover records of this repository and are reproducible from the files cited in Section 2.*

---

## Appendix A — Minimal Update Law (reference pseudocode)

```text
Inputs : qd[N][7]  canonical dense trajectory (10 ms grid via retiming)
         q_log     black-box positions of trial k, resampled to canonical grid
         u_k[N][7] current patch (zeros on first trial)
         gamma=0.6, m=1, rho=0.98, fc=1.0 Hz, clamps per Section 6

e = qd - q_log                                   # per joint, canonical grid
rms_new = rms(e, admissible band only)
if not valid(trial):      keep u_k; exit
if rms_new > 1.2 * rms_prev (twice):      rollback u_k; halt; exit

for j in joints:                                  # decoupled per joint
    e_lead        = shift(e[:, j], m)             # time advance
    u_new[:, j]   = rho * zerophase_lowpass(u_k[:, j] + gamma * e_lead, fc)
    u_new[:, j]  = clamp(u_new[:, j], -clamp_j, +clamp_j)
write u_{k+1} = u_new  (hash-keyed by trajectory+scale)
```

## Appendix B — Phase-Gate Checklist (one page)

- [ ] Phase 0: repeatability std ≤ 0.5° documented; wrist band boundary drawn
- [ ] Phase 1: sim matrix {main, no-Q, γ=2.0} outcomes match Section 4 expectations
- [ ] Phase 2: shadow correlation ≥ 0.8; |u| within clamps
- [ ] Phase 3: one-row 3 trials → full sweep ≤ 12 trials → scale 0.8; zero watchdog trips
- [ ] Phase 4: hash-keyed patch store; rollback drill executed; handover updated
