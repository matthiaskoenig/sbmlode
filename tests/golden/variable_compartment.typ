#set document(title: [Variable compartments])
#set page(margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
// the equations of a section are one block, which breaks across pages
#show math.equation.where(block: true): set block(breakable: true)
#show math.equation.where(block: true): set par(leading: 0.9em)

#align(center, text(size: 16pt, weight: "bold")[Variable compartments])

Model `variable_`#sym.zws;`compartment`, SBML Level 3 Version 2, read from variable\_compartment.xml, written by sbmlode VERSION.

Species in concentration in compartments whose size changes: Vc grows by a rate rule, Va follows an assignment rule of the time and of S1, the event grow doubles Ve. B is a boundary species which is diluted as Vc grows.

= Units

#table(
  columns: 4,
  stroke: none,
  column-gutter: 1.5em,
  table.hline(stroke: 0.8pt),
  table.header([*Time*], [*Substance*], [*Extent*], [*Volume*]),
  table.hline(stroke: 0.4pt),
  [s], [mole], [mole], [l],
  table.hline(stroke: 0.8pt),
)

= Compartments

#table(
  columns: (auto, auto, 1fr, auto, auto, auto),
  stroke: none,
  align: (left, left, left, left, left, center),
  table.hline(stroke: 0.8pt),
  table.header([*Symbol*], [*Id*], [*Name*], [*Size*], [*Unit*], [*Constant*]),
  table.hline(stroke: 0.4pt),
  [$upright("Vc")$], [`Vc`], [growing cell], [$1$], [l], [],
  [$upright("Va")$], [`Va`], [assigned compartment], [], [l], [],
  [$upright("Ve")$], [`Ve`], [resized compartment], [$1$], [l], [],
  table.hline(stroke: 0.8pt),
)

= Species

#table(
  columns: (auto, auto, 1fr, auto, auto, auto, 1fr),
  stroke: none,
  align: (left, left, left, left, left, left, left),
  table.hline(stroke: 0.8pt),
  table.header([*Symbol*], [*Id*], [*Name*], [*Compartment*], [*Value*], [*Unit*], [*Properties*]),
  table.hline(stroke: 0.4pt),
  [$S_(1)$], [`S1`], [substrate], [$upright("Vc")$], [$1$], [mole/l], [concentration],
  [$S_(2)$], [`S2`], [intermediate], [$upright("Va")$], [], [mole/l], [concentration],
  [$S_(3)$], [`S3`], [product], [$upright("Ve")$], [$1$], [mole/l], [concentration],
  [$B$], [`B`], [buffer], [$upright("Vc")$], [$1$], [mole/l], [concentration, boundary],
  table.hline(stroke: 0.8pt),
)

= Parameters

#table(
  columns: (auto, auto, 1fr, auto, auto, auto),
  stroke: none,
  align: (left, left, left, left, left, center),
  table.hline(stroke: 0.8pt),
  table.header([*Symbol*], [*Id*], [*Name*], [*Value*], [*Unit*], [*Constant*]),
  table.hline(stroke: 0.4pt),
  [$k_("g")$], [`k_g`], [growth rate], [$0.1$], [1/s], [#sym.checkmark],
  [$k_("a")$], [`k_a`], [expansion rate], [$0.2$], [1/s], [#sym.checkmark],
  [$upright("Va")_(0)$], [`Va0`], [initial size of Va], [$1$], [l], [],
  [$c_("a")$], [`c_a`], [swelling by the substrate], [$0.1$], [l#super[2]\/mol], [#sym.checkmark],
  [$k_(1)$], [`k1`], [], [$1$], [1/s], [#sym.checkmark],
  [$k_(2)$], [`k2`], [], [$0.5$], [1/s], [#sym.checkmark],
  table.hline(stroke: 0.8pt),
)

= Initial assignments and assignment rules

The initial assignments set the values at $t = 0$:

$ S_(2) &= (1)/(upright("Va")) $

The assignment rules hold at every time $t$:

$ upright("Va") &= upright("Va")_(0) dot (1 + k_("a") dot t) + c_("a") dot S_(1) \
  (dif upright("Va"))/(dif t) &= upright("Va")_(0) dot k_("a") + c_("a") dot ((-v_("J1"))/(upright("Vc")) - (S_(1))/(upright("Vc")) dot (dif upright("Vc"))/(dif t)) $

The rate of the size of a compartment with an assignment rule is the derivative of the rule in time, with the rates of change of the states it depends on.

= Reactions

#table(
  columns: (auto, auto, 1fr, 1fr),
  stroke: none,
  align: (left, left, left, left),
  table.hline(stroke: 0.8pt),
  table.header([*Rate*], [*Id*], [*Name*], [*Equation*]),
  table.hline(stroke: 0.4pt),
  [$v_("J1")$], [`J1`], [transport], [$S_(1) --> S_(2)$],
  [$v_("J2")$], [`J2`], [conversion], [$S_(2) --> S_(3)$],
  table.hline(stroke: 0.8pt),
)

The rates of the reactions are:

$ v_("J1") &= k_(1) dot S_(1) dot upright("Vc") \
  v_("J2") &= k_(2) dot S_(2) dot upright("Va") $

= ODE system

The states change in time with the rates of the reactions and the rate rules:

$ (dif upright("Vc"))/(dif t) &= k_("g") dot upright("Vc") quad "(rate rule)" \
  (dif S_(1))/(dif t) &= (-v_("J1"))/(upright("Vc")) - (S_(1))/(upright("Vc")) dot (dif upright("Vc"))/(dif t) \
  (dif S_(2))/(dif t) &= (v_("J1") - v_("J2"))/(upright("Va")) - (S_(2))/(upright("Va")) dot (dif upright("Va"))/(dif t) \
  (dif S_(3))/(dif t) &= (v_("J2"))/(upright("Ve")) \
  (dif B)/(dif t) &= -(B)/(upright("Vc")) dot (dif upright("Vc"))/(dif t) $

The concentration of a species in a compartment whose size changes is diluted by the rate of the size (SBML Level 3 Version 2, section 3.4.6), the amount it holds is kept.

= Events

*Event `grow`*

- Trigger: $t > 2$
- `initialValue` true, `persistent` true, `useValuesFromTriggerTime` true

$ upright("Ve") &colon.eq 2 \
  S_(3) &colon.eq (S_(3) dot upright("Ve"))/(upright("Ve")^("new")) $

$S_(3)$ is converted from the size of $upright("Ve")$ before the event to its size $upright("Ve")^("new")$ after the event, so that its amount is kept.
