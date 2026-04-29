# Theory

This page states the optimisation model that `evcharge` solves, what is known about the simple
deadline rules on it, why the LP relaxation is a valid lower bound, what separates an online
controller from the offline optimum, and the physics behind the phase and V2G models. Every
claim is either proved here (in sketch) or cited; where a claim is only empirical, that is said.

## Notation and the scheduling problem

Time is divided into steps $t = 0, \dots, T-1$ of length $\Delta t$ hours. Session $s$ is
connected in the window $\mathcal{T}_s = \lbrace a_s, \dots, d_s - 1 \rbrace$ and is commanded
by a setpoint $x_{s,t}$ in its control unit: kW on aggregate sites, amperes per phase on
phase-aware sites. One unit of setpoint is $k_s$ kW of grid power ($k_s = 1$ for kW setpoints,
$0.23 \cdot n$ for $n$ phases on TN, see [the phase model](#the-phase-model)). A charging
setpoint is $0$ or in $[\underline{x}_s, \overline{x}_s]$, where $\underline{x}_s$ is the
IEC 61851 minimum of 6 A; bidirectional sessions may also discharge with a magnitude $0$ or in
$[\underline{x}^-_s, \overline{x}^-_s]$.

**Capacity rows.** Every constraint of the site is a row $r$ of the form

$$
\sum_s \left( a_{r,s}\, x^+_{s,t} + a^-_{r,s}\, x^-_{s,t} \right) \le h_{r,t} \qquad \forall t, r,
$$

with charging and discharging magnitudes $x^+, x^- \ge 0$ and $x = x^+ - x^-$. The aggregate
model has a single row ($a = 1$, $h_t = L - B_t + V_t$ for grid limit $L$, base load $B_t$ and
PV $V_t$). A phase-aware site adds one row per line in amperes, and a bidirectional site adds
export rows (see below). `Scenario.rows` builds them; the simulator, the heuristics and the LP
all use the same rows.

**The mixed-integer program** (costs in EUR; $\pi_t \ge \pi^{\mathrm{exp}}_t$ are the import
and export prices, $D$ the demand charge, $\rho$ the unmet-energy penalty, $\eta_s$ the
charging efficiency, $E_s$ the request):

$$
\begin{aligned}
\min\quad
  & \sum_t \Delta t \left( \pi_t\, g^+_t - \pi^{\mathrm{exp}}_t\, g^-_t \right) + D\, P
    + \rho \sum_s u_s + \sum_s \kappa_s\, o_s + \sum_{s,t} c^{\mathrm{wear}}_s \Delta t\, k_s
      \left( \eta_s x^+_{s,t} + x^-_{s,t} / \eta^-_s \right) \\
\text{s.t.}\quad
  & g^+_t - g^-_t = B_t - V_t + \sum_s k_s \left( x^+_{s,t} - x^-_{s,t} \right)
    && \text{(power balance)} \\
  & \text{capacity rows as above}, \qquad g^+_t \le P, \quad P \ge P_0 && \text{(peak)} \\
  & \textstyle\sum_{t \in \mathcal{T}_s} \eta_s k_s x^+_{s,t} \Delta t + u_s - o_s = E_s
    && \text{(charge-only sessions)} \\
  & b_{s,t} = b_{s,t-1} + \eta_s k_s x^+_{s,t} \Delta t - k_s x^-_{s,t} \Delta t / \eta^-_s - w_{s,t},
    \quad \underline{b}_s \le b_{s,t} \le \overline{b}_s, \quad b_{s,d_s-1} + u_s \ge b^{\star}_s
    && \text{(V2G sessions)} \\
  & \underline{x}_s\, y^+_{s,t} \le x^+_{s,t} \le \overline{x}_s\, y^+_{s,t}, \quad
    \underline{x}^-_s\, y^-_{s,t} \le x^-_{s,t} \le \overline{x}^-_s\, y^-_{s,t}, \quad
    y^+_{s,t} + y^-_{s,t} \le 1, \quad y \in \lbrace 0, 1 \rbrace && \text{(6 A minimum)} \\
  & 0 \le u_s,\quad 0 \le o_s \le \eta_s k_s \underline{x}_s \Delta t, \quad
    0 \le w_{s,t} \le \eta_s k_s x^+_{s,t} \Delta t, \quad g^\pm_t \ge 0
\end{aligned}
$$

Setpoints exist only inside the window. $u_s$ is unmet energy, $o_s$ the *overshoot* of a
charge-only EV that finishes during a minimum-current step (commanded but never drawn, since
the EV stops when full) and $w_{s,t}$ the same for a V2G battery that reaches its ceiling. Both
are priced at $\kappa_s = \max(0, -\min_{t \in \mathcal{T}_s} \pi^{\mathrm{exp}}_t) / \eta_s$
so that undrawn energy can never look like revenue; $w$ is allowed only where the minimum is
enforced. $\rho = 100$ EUR/kWh exceeds the marginal cost of delivering a kWh
($\pi_t + D/(\eta \Delta t)$ for realistic tariffs), so the optimum delivers everything it can
before it minimises cost. MPC adds a small early-charging term
$q \sum_{s,t} (t-k)\, \Delta t \cdot k_s x^+_{s,t} \Delta t$.

Dropping the binaries (and the grid resolution) gives the **LP relaxation**. The offline
optimum (`OptimalSchedule`) solves the MILP by relax-and-fix: the LP is solved, entries that
the LP left at 0 or at least at the minimum are fixed, and a small MILP decides the rest; a
final rounding MILP puts every setpoint on the charger's resolution. `solve_schedule(...,
strategy="exact")` solves the full MILP.

## When EDF and LLF are optimal, and when not

Charging EVs by deadline is real-time scheduling with a divisible resource: an EV is a job with
release time $a_s$, deadline $d_s$ and work $W_s = E_s / \eta_s$ (grid-side energy), and the
site's capacity is the processor.

**One shared resource, no per-EV limit, no minimum.** Suppose the only constraint is the kW
row, capacity $C_t = h_t$ per step, and any split of $C_t$ among connected EVs is allowed.
Then *earliest deadline first* (EDF: give the whole step capacity to the connected EV with the
earliest deadline, then the next) delivers every request whenever any schedule does. This is
the discrete-time form of the classical result that EDF is optimal for preemptive scheduling
of independent jobs on one processor (Liu and Layland 1973 for periodic tasks; Dertouzos 1974
for arbitrary job sets). *Proof sketch (exchange argument).* Take a feasible schedule and
the first step $t$ where it differs from EDF: it gives energy to an EV $j$ while an EV $i$ with
$d_i \le d_j$ still has work left. Since $i$ finishes by $d_i$, it receives energy in some later
step $t'$ with $t < t' < d_i \le d_j$. Move $\varepsilon$ kWh (the smaller of the two amounts)
so that $i$ gets $\varepsilon$ more at $t$ and less at $t'$, and $j$ the opposite. Every step's
total is unchanged, both EVs are connected at $t$ and $t'$, and both still finish by their
deadlines. (If step $t$ leaves capacity unused while $i$ has work left, moving energy of $i$
from $t'$ to $t$ works the same way.) Repeating the exchange turns the schedule into EDF's
without losing feasibility.
Least laxity first (LLF) is optimal in the same setting (Mok 1983).

**Per-EV power limits make it a parallel-machine problem.** With a cap $\overline{p}_s$ per EV
(charger, cable and on-board charger), the exchange above can exceed a cap, and EDF is no longer
optimal. Example: capacity 2 kW per step, every EV capped at 1 kW, $\Delta t = 1$ h. EV 1
needs 3 kWh by the end of step 2; EVs 2 and 3 need 1 kWh each by the end of step 1. EDF serves
EVs 2 and 3 in step 0, so EV 1 can get only 2 kWh; the schedule {1, 2}, {1, 3}, {1} delivers
everything (so does LLF here: EV 1 has zero laxity). Offline, feasibility with caps, windows
and per-step capacities is still easy: it is a maximum-flow problem (Horn 1974), sessions to
steps with arc capacities $\overline{p}_s \Delta t$, and the LP solves it. Online it is not:
for $m \ge 2$ identical processors, no online algorithm is optimal without knowledge of future
arrivals (Dertouzos and Mok 1989), so no online rule, EDF and LLF included, can guarantee to
deliver whenever the offline optimum does. The experiments show this happening: on the tight
workplace days LLF leaves 14.1 kWh per day undelivered against the optimum's 12.6
([experiments.md](experiments.md#2-does-an-arrival-forecast-make-mpc-robust)).

**The minimum current makes the problem hard.** The feasible set of a setpoint,
$\lbrace 0 \rbrace \cup [\underline{x}_s, \overline{x}_s]$, is not convex. Deciding whether all
requests can be met is then NP-hard in the strong sense, even on the aggregate model: take EVs
whose minimum equals their maximum ($\underline{p}_s = \overline{p}_s = p_s$, an on/off load),
request $E_s = \eta_s p_s \Delta t$ and a window covering all $T$ steps. Each must be on in at
least one step, and being on in more only uses capacity, so all requests can be met exactly
when the items $p_s$ can be packed into $T$ bins of capacity $C$: bin packing (Garey and
Johnson 1979). Unless P = NP, no polynomial-time rule is optimal with the minimum current, and
the MILP is the honest tool. The LP relaxation stays polynomial and gives a bound (next
section).

## The LP relaxation is a lower bound

**Claim.** For every policy, the penalised cost the simulator measures,
$\text{energy} + \text{demand charge} + \text{wear} + \rho \cdot \text{unmet}$, is at least the
optimum of the LP relaxation (`relaxation_bound`).

*Proof sketch.* Map the executed trajectory to a point of the LP: let $x^+_{s,t}$ be the power
an EV actually **drew** (not the command) divided by $k_s$, $x^-_{s,t}$ its discharge, $g^+_t,
g^-_t$ the positive and negative parts of the measured site import, $P$ the measured peak,
$u_s$ the undelivered energy (or battery shortfall) and $o = w = 0$.

- *Rows.* The simulator checks every row on the commands. Charging coefficients are
  non-negative on import rows and zero on export rows, and an EV draws at most its command
  (discharge draws exactly the command), so every row also holds for the drawn values. On line
  rows whose sessions all have a resolution, the LP uses the right-hand side rounded down to
  that grid. That is still valid: the commanded left-hand side is a multiple of the grid (the
  coefficients are 0 or $\pm 1$) and at most $h$, hence at most $h$ rounded down, and the drawn
  left-hand side is not larger.
- *Energy and batteries.* The simulator adds $\eta_s k_s x^+ \Delta t$ to the battery and takes
  $k_s x^- \Delta t / \eta^-_s$ out, exactly the LP's balance, and it enforces the battery
  bounds.
- *Cost.* With $\pi_t \ge \pi^{\mathrm{exp}}_t$, the LP cost of this point is exactly the
  simulated cost, and the minimum-current constraints, which the executed schedule satisfies
  anyway, are not in the LP.

The point is feasible, so the LP optimum is at most its cost. MPC's regulariser is not part of
the measured cost and is set to 0 in the bound. $\square$

**The sandwich.** For the offline optimum, the MILP objective is an upper bound on the cost of
replaying its plan: the replay commands exactly the plan, the EVs draw at most the command, and
the undrawn energy is at most $o_s$ (or $w_{s,t}$). The cost of a step is convex and
piecewise linear in its consumption with slopes between $\pi^{\mathrm{exp}}_t$ and $\pi_t$, so
drawing $\delta$ kWh less than planned raises the energy cost by at most
$\max(0, -\pi^{\mathrm{exp}}_t)\, \delta$, which the $\kappa$ term has already paid; the peak
can only fall. Hence

$$
\text{LP bound} \;\le\; \text{executed cost of any policy}, \qquad
\text{executed cost of }\texttt{optimal} \;\le\; \text{its MILP objective},
$$

and both are asserted by hypothesis property tests on random aggregate, phase-aware and V2G
sites (`tests/test_properties.py`). The `gap %` column of `evcharge compare` is the executed
penalised cost above the LP bound. It is a certificate: a policy at 0.05 % cannot be improved by
more than 0.05 %.

## The value of information and forecast-aware MPC

Let $\xi$ be the random day (arrivals, departures, energies) and $J(\pi, \xi)$ the penalised
cost of a policy $\pi$ on it. A *non-anticipative* policy decides step $k$ from what is known at
$k$ only; the offline optimum $J^\ast(\xi)$ sees all of $\xi$. For every non-anticipative
policy and every day, $J(\pi, \xi) \ge J^\ast(\xi)$, because its schedule is feasible for the
offline problem. So

$$
\mathbb{E}\, J(\pi, \xi) - \mathbb{E}\, J^\ast(\xi)
  = \underbrace{\left( \mathbb{E}\, J(\pi, \xi) - \min_{\pi'} \mathbb{E}\, J(\pi', \xi) \right)}_{\text{suboptimality of } \pi}
  + \underbrace{\left( \min_{\pi'} \mathbb{E}\, J(\pi', \xi) - \mathbb{E}\, J^\ast(\xi) \right)}_{\text{expected value of perfect information}}
$$

with both terms non-negative (the minimum is over non-anticipative policies; in stochastic
programming the second term is the EVPI, Birge and Louveaux 2011). The gap of an online
controller to the clairvoyant optimum is therefore not all "controller error": part of it is the
price of not knowing the future, which no online controller can avoid. The LP bound is a bound
for the online problem too, but not a tight one.

Plain MPC re-solves the offline problem over the *connected* sessions at every step and applies
the first step. With perfect information (all sessions connected from the start), no minimum
currents and $q = 0$, the tail of an optimal plan is optimal for the tail problem (Bellman's
principle of optimality), so MPC reproduces the offline optimum; a property test checks this on
LP instances (with ties the optimal plan need not be unique, so costs are compared, not
schedules). Without that information MPC treats the future as empty.

The forecast-aware variants replace the empty future by one learned from training days:

- **Certainty equivalence (`mpc-ev`)**: plan for the expected future fleet, as continuous ghost
  loads. It ignores the spread around the mean.
- **Sample average approximation (`mpc-saa`)**: sample $K$ future days
  $\xi^1, \dots, \xi^K$ and solve
  $\min \frac{1}{K} \sum_k J(x_0, x^k_{1:}, \xi^k)$, where the first-step decision $x_0$ of the
  connected EVs is shared by all scenarios (non-anticipativity) and the later decisions
  $x^k_{1:}$ are scenario-specific (`optim.solve_scenarios`). It is a two-stage approximation of
  a multi-stage problem: within a scenario the later steps see that scenario's whole future, so
  the recourse is optimistic. As $K \to \infty$ the optimal value and solutions of an SAA
  problem converge to those of the true two-stage problem under standard conditions (Shapiro,
  Dentcheva and Ruszczyński 2009); with $K = 10$ it is a heuristic.
- **Capacity reserve (`mpc-reserve`)**: keep plain MPC and subtract the expected load of future
  arrivals, each spread uniformly over its window, from the capacity of later steps. A
  heuristic without an optimality claim.

How much these recover of the value of information is an empirical question; the answer on the
built-in scenarios is in [experiments.md](experiments.md#2-does-an-arrival-forecast-make-mpc-robust).

## The phase model

**Currents.** On a TN supply (230 V line to neutral, 400 V line to line) a single-phase EV at
current $I$ draws $I$ on one line and returns it on the neutral; a three-phase EV draws $I$ on
each line. Its power is $n \cdot 230\ \mathrm{V} \cdot I$ for $n$ phases. On an IT supply
(230 V line to line, no neutral) a single-phase EV between L1 and L2 draws $I$ out of L1 and
back through L2, so it loads **two** lines with the same current, at $230\ \mathrm{V} \cdot I$;
a three-phase EV on 230 V draws $I$ on each line at $\sqrt{3} \cdot 230\ \mathrm{V} \cdot I$. A
charger's rotation maps its conductors to site lines. The model's row for line $\ell$ is
$\sum_s \mathbb{1}[\ell \in \text{lines}(s)]\, x_s \le \text{fuse}_\ell - B_\ell + \kappa V_\ell$
with $\kappa = 1$ on TN and $0$ on IT, plus the site's kW row.

The true current of a line is the magnitude of the **phasor** sum of the device currents. What
the linear model guarantees:

1. **Loads only: conservative for any phase angles.** By the triangle inequality
   $\lvert \sum_i I_i \rvert \le \sum_i \lvert I_i \rvert$, and the model uses the right-hand
   side. It is exact when all currents on a line are in phase, as for unity-power-factor EVs
   and loads on TN.
2. **TN with injection (PV, V2G discharge): exact for collinear currents.** A unity-power-factor
   load on a TN line draws current in phase with that line's voltage, and an injection is in
   anti-phase, so the signed sum $\lvert B - V + c - d \rvert$ is the true magnitude and
   crediting injection against load is correct.
3. **IT with injection: no credit is needed to be safe, and none is given.** On line L1, a
   unity-power-factor load on the pair L1-L2 draws current in phase with $V_{12}$, which is
   $30°$ ahead of the L1 phase reference; a load on L1-L3 is $30°$ behind; a three-phase load is
   in phase. So all load currents on L1 lie in a cone of half-angle $30°$, and their sum $L$
   does too (the cone is convex); injections lie in the opposite cone, with sum $-G$ for some
   $G$ in the load cone. The angle $\theta$ between $L$ and $G$ is at most $60°$, so
   $\lvert L - G \rvert^2 = \lvert L \rvert^2 + \lvert G \rvert^2 - 2 \lvert L \rvert \lvert G \rvert \cos\theta
   \le \lvert L \rvert^2 + \lvert G \rvert^2 - \lvert L \rvert \lvert G \rvert
   \le \max(\lvert L \rvert, \lvert G \rvert)^2$.
   Hence the line current is at most $\max(\sum \text{load currents}, \sum \text{injected
   currents})$, and the model's two rows per line (load current within the fuse, injected
   current within the fuse) bound it. Crediting injection against load would not be safe: two
   currents $60°$ apart do not cancel.

**Reactive base load on TN.** Item 2 assumes the base load is at unity power factor. If its
current $B$ lags the voltage by $\varphi$, and the EVs' net current $x = c - V$ on that line is
real, then $\lvert B e^{-j\varphi} + x \rvert^2 = (B + x)^2 - 2Bx(1 - \cos\varphi)$. For
$x \ge 0$ (EVs draw at least the PV current) the model $\lvert B + x \rvert$ is conservative.
For $x < 0$ it can under-estimate, by at most
$\sqrt{(B + x)^2 + 4 B \lvert x \rvert \sin^2(\varphi/2)} - \lvert B + x \rvert$. Example:
$B = 20$ A at power factor 0.9 and PV exceeding the EV current by $\lvert x \rvert = 10$ A: the
model says 10.0 A, the true current is 11.8 A. The error arises only when PV exceeds the EV
load on a line, which is when that line is far from its fuse unless the base load is large.

**Integrality.** Setpoints have a resolution (0.1 A by default, OCPP 1.6 limits carry one
decimal). A line row whose coefficients are all $\pm 1$ has an on-grid left-hand side, which is
why rounding its right-hand side down keeps the LP a valid bound (see the proof above).

## Bidirectional charging

A V2G session is modelled by its battery energy $b_{s,t}$ (kWh, end of step $t$):

- **Dynamics**: $b_t = b_{t-1} + \eta_s k_s x^+_t \Delta t - k_s x^-_t \Delta t / \eta^-_s$ from
  the arrival energy, within $[\underline{b}_s, \overline{b}_s]$ (driver's reserve and
  ceiling), and $b_{d_s - 1} \ge b^\star_s$ (arrival energy plus the request) up to unmet energy.
  Both efficiencies make a round trip lose energy, so cycling pays only if the price spread
  exceeds the losses plus wear.
- **Wear** is linear in battery throughput: $c^{\mathrm{wear}}$ per kWh charged into or taken
  out of the battery. Linear throughput cost is a first-order model; calendar ageing and the
  dependence on depth, rate and temperature are not modelled.
- **Rows.** Discharge enters import rows with coefficient $-k_s$ (kW row) and, on TN, $-1$ per
  line it uses; on IT its per-line coefficient in import rows is 0 (no credit, item 3 above).
  Export rows (export limit in kW, and per line the fuse against injected current, credited by
  the base load on TN only) contain discharge with coefficient $+k_s$ or $+1$ and charging with
  coefficient 0, so an EV that stops charging early cannot break an export row.
- **No simultaneous charge and discharge**: $y^+ + y^- \le 1$ in the MILP. The LP relaxation may
  do both at once, which wastes energy through the efficiencies; it can only lower the relaxed
  optimum, so the bound stays valid.
- **The simulator** tracks $b$, stops charging at the ceiling (the battery management's cut-off,
  which is why the MILP has the spill $w$), and clips a discharge command that would take the
  battery below its floor within the step (a `soc-limit` violation).

The value of V2G on the built-in scenarios is in [v2g.md](v2g.md).

## References

- Birge, J. R., and Louveaux, F. (2011). *Introduction to Stochastic Programming*, 2nd ed.
  Springer.
- Dertouzos, M. L. (1974). Control robotics: the procedural control of physical processes.
  *Proceedings of IFIP Congress 74*, 807-813.
- Dertouzos, M. L., and Mok, A. K. (1989). Multiprocessor on-line scheduling of hard-real-time
  tasks. *IEEE Transactions on Software Engineering*, 15(12), 1497-1506.
- Garey, M. R., and Johnson, D. S. (1979). *Computers and Intractability: A Guide to the Theory
  of NP-Completeness*. W. H. Freeman.
- Horn, W. A. (1974). Some simple scheduling algorithms. *Naval Research Logistics Quarterly*,
  21(1), 177-185.
- Lee, Z. J., Li, T., and Low, S. H. (2019). ACN-Data: analysis and applications of an open EV
  charging dataset. *Proceedings of the Tenth ACM International Conference on Future Energy
  Systems (e-Energy '19)*.
- Liu, C. L., and Layland, J. W. (1973). Scheduling algorithms for multiprogramming in a
  hard-real-time environment. *Journal of the ACM*, 20(1), 46-61.
- Mok, A. K. (1983). *Fundamental Design Problems of Distributed Systems for the Hard-Real-Time
  Environment*. PhD thesis, Massachusetts Institute of Technology.
- Shapiro, A., Dentcheva, D., and Ruszczyński, A. (2009). *Lectures on Stochastic Programming:
  Modeling and Theory*. SIAM.

The quick-charge regulariser of MPC follows the idea of the "quick charge" objective of the
Caltech Adaptive Charging Network scheduler (Lee, Li and Low 2019 describe the network and its
data set).
