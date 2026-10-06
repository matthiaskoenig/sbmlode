#set document(title: [events\_model])
#set page(margin: 2cm)
#set text(size: 10pt)
#set par(justify: true)
#set heading(numbering: "1.")
// the equations of a section are one block, which breaks across pages
#show math.equation.where(block: true): set block(breakable: true)
#show math.equation.where(block: true): set par(leading: 0.9em)

#align(center, text(size: 16pt, weight: "bold")[events\_model])

Model `events_model`, SBML Level 3 Version 2, read from events.xml, written by sbmlode VERSION.

= Compartments

#table(
  columns: (auto, auto, auto, auto),
  stroke: none,
  align: (left, left, left, center),
  table.hline(stroke: 0.8pt),
  table.header([*Symbol*], [*Id*], [*Size*], [*Constant*]),
  table.hline(stroke: 0.4pt),
  [$c$], [`c`], [$1$], [#sym.checkmark],
  [$V$], [`V`], [$2$], [],
  table.hline(stroke: 0.8pt),
)

= Species

#table(
  columns: (auto, auto, 1fr, auto, auto, 1fr),
  stroke: none,
  align: (left, left, left, left, left, left),
  table.hline(stroke: 0.8pt),
  table.header([*Symbol*], [*Id*], [*Name*], [*Compartment*], [*Value*], [*Properties*]),
  table.hline(stroke: 0.4pt),
  [$S$], [`S`], [substrate], [$c$], [], [concentration],
  [$P$], [`P`], [], [$c$], [$0$], [concentration],
  [$A$], [`A`], [], [$V$], [$5$], [concentration],
  [$B$], [`B`], [], [$V$], [$1$], [concentration],
  table.hline(stroke: 0.8pt),
)

= Parameters

#table(
  columns: (auto, auto, auto, auto),
  stroke: none,
  align: (left, left, left, center),
  table.hline(stroke: 0.8pt),
  table.header([*Symbol*], [*Id*], [*Value*], [*Constant*]),
  table.hline(stroke: 0.4pt),
  [$upright("vmax")$], [`vmax`], [$2$], [#sym.checkmark],
  [$upright("km")$], [`km`], [$0.5$], [#sym.checkmark],
  [$k_(1)$], [`k1`], [$0.1$], [#sym.checkmark],
  [$k_(2)$], [`k2`], [], [],
  [$S_(0)$], [`S0`], [$10$], [#sym.checkmark],
  [$upright("total")$], [`total`], [$0$], [],
  table.hline(stroke: 0.8pt),
)

= Function definitions

$ upright("mm")(S, upright("km")) &= (S)/(upright("km") + S) $

= Initial assignments and assignment rules

The initial assignments set the values at $t = 0$:

$ S &= S_(0) $

The assignment rules hold at every time $t$:

$ k_(2) &= 2 dot k_(1) $

= Reactions

#table(
  columns: (auto, auto, 1fr),
  stroke: none,
  align: (left, left, left),
  table.hline(stroke: 0.8pt),
  table.header([*Rate*], [*Id*], [*Equation*]),
  table.hline(stroke: 0.4pt),
  [$v_("J0")$], [`J0`], [$S harpoons.rtlb P$],
  [$v_("J1")$], [`J1`], [$S + A harpoons.rtlb 2 thin P$],
  table.hline(stroke: 0.8pt),
)

The rates of the reactions are:

$ v_("J0") &= upright("vmax") dot upright("mm")(S, upright("km")) \
  v_("J1") &= k_(1) dot S dot A $

= ODE system

The states change in time with the rates of the reactions and the rate rules:

$ (dif V)/(dif t) &= 0.1 quad "(rate rule)" \
  (dif S)/(dif t) &= (-v_("J0") - v_("J1"))/(c) \
  (dif P)/(dif t) &= (v_("J0") + 2 dot v_("J1"))/(c) \
  (dif A)/(dif t) &= (-v_("J1"))/(V) - (A)/(V) dot (dif V)/(dif t) \
  (dif B)/(dif t) &= -0.1 dot B quad "(rate rule)" $

The concentration of a species in a compartment whose size changes is diluted by the rate of the size (SBML Level 3 Version 2, section 3.4.6), the amount it holds is kept.

= Events

*Event `E1`* (reset)

- Trigger: $t > 5$
- Priority: $1$
- `initialValue` true, `persistent` true, `useValuesFromTriggerTime` true

$ V &colon.eq 2 dot V \
  S &colon.eq 10 \
  A &colon.eq (A dot V)/(V^("new")) \
  B &colon.eq (B dot V)/(V^("new")) $

$A$ is converted from the size of $V$ before the event to its size $V^("new")$ after the event, so that its amount is kept.

$B$ is converted from the size of $V$ before the event to its size $V^("new")$ after the event, so that its amount is kept.

*Event `E2`*

- Trigger: $S < 2$
- Delay: $1$
- `initialValue` true, `persistent` true, `useValuesFromTriggerTime` true

$ A &colon.eq 1 \
  upright("total") &colon.eq upright("total") + 1 $
