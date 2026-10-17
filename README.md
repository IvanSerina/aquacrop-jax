# AquaCrop-JAX

JAX port of AquaCrop v7.3. The external compatibility target is the
standalone program: the same `LIST/ListProjects.txt` and legacy AquaCrop input
files must produce equivalent `.OUT` files.

## Status

This repository is being migrated from the progressive TensorFlow port. The
legacy compatibility layer is retained while kernels are rewritten directly
with JAX in 64-bit precision. It is **not yet a complete AquaCrop simulator**.
The authoritative parity fixture is `../aquacrop-fortran/testcase/OUTP_REF`.

### Backend migration

- Migrated: numerical kernels, climate/GDD, drainage, infiltration,
  two-stage evaporation, roots, transpiration, canopy, production and coupled
  `lax.scan` water trajectories.
- Pending: the full legacy runner, salinity and exact `.OUT` parity.
  Irrigation and groundwater kernels are implemented; legacy `.IRR`/`.GWT`
  parsing and full-run wiring remain.
- TensorFlow is not a dependency of this distribution.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
```

## Commands

```bash
aquacrop-jax inspect ../aquacrop-fortran/testcase/LIST/Ottawa.PRM
# Executable JAX diagnostic stages:
aquacrop-jax run ../aquacrop-fortran/testcase/LIST/ListProjects.txt \
  --stage climate --output OUTP_TF/climate.csv
aquacrop-jax run ../aquacrop-fortran/testcase/LIST/ListProjects.txt \
  --stage water --output OUTP_TF/water.csv
aquacrop-jax run ../aquacrop-fortran/testcase/LIST/ListProjects.txt \
  --stage water-evap --output OUTP_TF/water-evap.csv
# Compare only after a simulator has populated OUTP:
aquacrop-jax compare OUTP ../aquacrop-fortran/testcase/OUTP_REF
pytest
```

`compare` is a regression checker, not a simulation command. The full daily
simulator and legacy `.OUT` writers have not yet been reconstructed, so
this milestone cannot populate `OUTP` by itself.

The Ottawa project contains 892 simulated calendar days across its three runs
(164 + 358 + 370). This environment uses JAX on CPU; GPU execution requires a
matching JAX CUDA wheel and driver.

The soil milestone includes the AquaCrop v7.3 `.SOL` reader, gravel-volume
conversion, initial 12-compartment discretization, drainage-tau derivation and
profile-water storage kernel. Dynamic expansion reproduces the Fortran
`AdjustSizeCompartments` geometry (3.00 m crop root depth becomes a 3.05 m
profile), and the daily water-balance residual is available as a float64 JAX
diagnostic.

The homogeneous-profile drainage path follows Fortran's drainability checks,
stores incoming water in dry compartments, and redistributes `Ksat` excess
upward while correcting interface fluxes. Unrounded native fixtures cover
wet/dry fronts, adjusted FC, gravel and backflow, with daily mass conservation.
Multihorizon soil profiles remain unsupported by the integrated runner.

The first daily hydrology composition is available for the homogeneous Ottawa
path: pre-infiltration drainage, SCS runoff, top-down saturation filling,
`Ksat`-limited transfer, deep percolation and an end-of-step mass residual.

The water-only recurrence can now be unrolled for an arbitrary daily rainfall
tensor. It returns `theta[day, compartment]` and daily runoff, drainage,
infiltration and balance residual histories. Evaporation, transpiration and
cross-season state are the next integrations before Ottawa output parity.

The `water` stage now connects this recurrence to all 892 Ottawa days, writes
12 compartment states per day, and preserves the final state across `KeepSWC`
run boundaries. It remains a diagnostic water-only trajectory, not `day.OUT`.

Stage-1 soil evaporation now includes the exact AquaCrop top-down extraction
rule, fractional evaporation-layer depth, WP/2 air-dry limit, REW store and
unmet-demand state. Stage 2 and canopy shading are not integrated yet.

Stage 2 now implements the 20 daily substeps, exponential reduction
coefficient, AquaCrop water limits and 1 mm evaporation-depth expansion for a
homogeneous profile. Salt transport and canopy shading remain separate.

Hydrology and both evaporation stages are now coupled in one recurrent graph.
The state carries theta, REW surface storage, stage-2 start percentage and
evaporation depth; the daily diagnostic includes actual evaporation.

Root development now includes AquaCrop's GDD growth curve and crop-period
guards. Root-zone storage integrates fractional boundary compartments and
returns FC, WP, SAT, depletion, TAW and bounded relative depletion.

The water-stress transpiration kernel now reproduces `KsAny`, root fractions,
the depth-varying root sink profile, compartment extraction capacity and the
Fortran residue redistribution loop. It conserves profile water and keeps
salinity and delayed anaerobic stress as explicit future recurrent operators.

Hydrology, two-stage evaporation and root transpiration are now composed in a
single `jax.lax.scan`. Daily root depth and potential transpiration remain
explicit inputs, and the resulting trajectory reports the combined water
balance residual and recurrent compartment state.

The potential crop-demand block now includes calendar and GDD no-stress canopy
curves, micro-advection, crop-coefficient ageing and CO2 correction, cold GDD
stress, late-season evaporation bounds and dying-canopy transpiration decline.

Production now has explicit recurrent state for actual, unlimited and
fertility-constrained biomass, daily CO2-adjusted water productivity,
reproductive WP switching, reference harvest-index development and dry yield.

Manual schedules, depletion/RAW generated triggers, fixed/refill depths and
net-irrigation profile adjustment are available as pure JAX operators. Daily
irrigation is an explicit input to the coupled water-crop recurrence and is
included in infiltration and mass-balance diagnostics.

Groundwater support includes piecewise-linear depth/EC interpolation,
water-table saturation, AquaCrop's quadratic field-capacity adjustment,
anaerobic root-depth clipping and the bottom-up capillary-rise recurrence.
Capillary and water-table inflows are explicit daily balance terms.

## Compatibility policy

- Python handles legacy text parsing and exact output formatting.
- JAX `float64` performs numerical kernels.
- State is explicit and immutable at kernel boundaries.
- Every translated operator is checked against Fortran-derived golden values.
- Output comparison ignores only the creation date/time line.
