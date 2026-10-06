# Variable compartments

Model `variable_compartment`, SBML Level 3 Version 2, read from variable\_compartment.xml, written by sbmlode VERSION.

Species in concentration in compartments whose size changes: Vc grows by a rate rule, Va follows an assignment rule of the time and of S1, the event grow doubles Ve. B is a boundary species which is diluted as Vc grows.

## Units

| Time | Substance | Extent | Volume |
| --- | --- | --- | --- |
| s | mole | mole | l |

## Compartments

| Symbol | Id | Name | Size | Unit | Constant |
| --- | --- | --- | --- | --- | :---: |
| $\mathrm{Vc}$ | `Vc` | growing cell | $1$ | l |  |
| $\mathrm{Va}$ | `Va` | assigned compartment |  | l |  |
| $\mathrm{Ve}$ | `Ve` | resized compartment | $1$ | l |  |

## Species

| Symbol | Id | Name | Compartment | Value | Unit | Properties |
| --- | --- | --- | --- | --- | --- | --- |
| $S_{1}$ | `S1` | substrate | $\mathrm{Vc}$ | $1$ | mole/l | concentration |
| $S_{2}$ | `S2` | intermediate | $\mathrm{Va}$ |  | mole/l | concentration |
| $S_{3}$ | `S3` | product | $\mathrm{Ve}$ | $1$ | mole/l | concentration |
| $B$ | `B` | buffer | $\mathrm{Vc}$ | $1$ | mole/l | concentration, boundary |

## Parameters

| Symbol | Id | Name | Value | Unit | Constant |
| --- | --- | --- | --- | --- | :---: |
| $k_{\mathrm{g}}$ | `k_g` | growth rate | $0.1$ | 1/s | ✓ |
| $k_{\mathrm{a}}$ | `k_a` | expansion rate | $0.2$ | 1/s | ✓ |
| $\mathrm{Va}_{0}$ | `Va0` | initial size of Va | $1$ | l |  |
| $c_{\mathrm{a}}$ | `c_a` | swelling by the substrate | $0.1$ | l<sup>2</sup>/mol | ✓ |
| $k_{1}$ | `k1` |  | $1$ | 1/s | ✓ |
| $k_{2}$ | `k2` |  | $0.5$ | 1/s | ✓ |

## Initial assignments and assignment rules

The initial assignments set the values at $t = 0$:

$$
\begin{aligned}
S_{2} &= \frac{1}{\mathrm{Va}}
\end{aligned}
$$

The assignment rules hold at every time $t$:

$$
\begin{aligned}
\mathrm{Va} &= \mathrm{Va}_{0} \cdot \mathopen{}\left(1 + k_{\mathrm{a}} \cdot t\right) + c_{\mathrm{a}} \cdot S_{1} \\[1ex]
\frac{\mathrm{d} \,\mathrm{Va}}{\mathrm{d} t} &= \mathrm{Va}_{0} \cdot k_{\mathrm{a}} + c_{\mathrm{a}} \cdot \mathopen{}\left(\frac{-v_{\mathrm{J1}}}{\mathrm{Vc}} - \frac{S_{1}}{\mathrm{Vc}} \cdot \frac{\mathrm{d} \,\mathrm{Vc}}{\mathrm{d} t}\right)
\end{aligned}
$$

The rate of the size of a compartment with an assignment rule is the derivative of the rule in time, with the rates of change of the states it depends on.

## Reactions

| Rate | Id | Name | Equation |
| --- | --- | --- | --- |
| $v_{\mathrm{J1}}$ | `J1` | transport | $S_{1} \longrightarrow S_{2}$ |
| $v_{\mathrm{J2}}$ | `J2` | conversion | $S_{2} \longrightarrow S_{3}$ |

The rates of the reactions are:

$$
\begin{aligned}
v_{\mathrm{J1}} &= k_{1} \cdot S_{1} \cdot \mathrm{Vc} \\[1ex]
v_{\mathrm{J2}} &= k_{2} \cdot S_{2} \cdot \mathrm{Va}
\end{aligned}
$$

## ODE system

The states change in time with the rates of the reactions and the rate rules:

$$
\begin{aligned}
\frac{\mathrm{d} \,\mathrm{Vc}}{\mathrm{d} t} &= k_{\mathrm{g}} \cdot \mathrm{Vc} \qquad \text{(rate rule)} \\[1ex]
\frac{\mathrm{d} S_{1}}{\mathrm{d} t} &= \frac{-v_{\mathrm{J1}}}{\mathrm{Vc}} - \frac{S_{1}}{\mathrm{Vc}} \cdot \frac{\mathrm{d} \,\mathrm{Vc}}{\mathrm{d} t} \\[1ex]
\frac{\mathrm{d} S_{2}}{\mathrm{d} t} &= \frac{v_{\mathrm{J1}} - v_{\mathrm{J2}}}{\mathrm{Va}} - \frac{S_{2}}{\mathrm{Va}} \cdot \frac{\mathrm{d} \,\mathrm{Va}}{\mathrm{d} t} \\[1ex]
\frac{\mathrm{d} S_{3}}{\mathrm{d} t} &= \frac{v_{\mathrm{J2}}}{\mathrm{Ve}} \\[1ex]
\frac{\mathrm{d} B}{\mathrm{d} t} &= -\frac{B}{\mathrm{Vc}} \cdot \frac{\mathrm{d} \,\mathrm{Vc}}{\mathrm{d} t}
\end{aligned}
$$

The concentration of a species in a compartment whose size changes is diluted by the rate of the size (SBML Level 3 Version 2, section 3.4.6), the amount it holds is kept.

## Events

**Event `grow`**

- Trigger: $t > 2$
- `initialValue` true, `persistent` true, `useValuesFromTriggerTime` true

$$
\begin{aligned}
\mathrm{Ve} &\mathrel{:=} 2 \\[1ex]
S_{3} &\mathrel{:=} \frac{S_{3} \cdot \mathrm{Ve}}{\mathrm{Ve}^{\mathrm{new}}}
\end{aligned}
$$

$S_{3}$ is converted from the size of $\mathrm{Ve}$ before the event to its size $\mathrm{Ve}^{\mathrm{new}}$ after the event, so that its amount is kept.
