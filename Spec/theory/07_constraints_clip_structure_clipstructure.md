# Clip, Structure, and Clipstructure

## Scenarios

Current scenario names are:

- `none`
- `clip`
- `structure`
- `clipstructure`

`clipstructure` is the canonical combined scenario.

## Structure

Structure assigns hidden units to groups. Feedforward masks connect source and target rows with matching group identity. Recurrent masks restrict recurrent weights to same-group hidden units.

The recurrent signal remains the previous output spike of the same layer. Structure masks the recurrent weight path; it does not introduce another recurrent source.

## Clip

Clip constrains cell parameters using bounded parameterization.

- LIF: membrane decay bounds.
- RF: resonant frequency and damping bounds.
- IF: threshold bounds.
- All supported cells: threshold policy where enabled.

## Clipstructure

Clipstructure combines group structure with per-group or per-feature parameter bounds.

## Output layer policy

Output-layer constraint application is intentionally unsupported in the current phase. Hidden-layer constraints are the implemented contract.
