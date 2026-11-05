# Langmuir (Carter–Kibler) mass uptake: code for the numerical experiments

Code accompanying L. Marazzi, *Identifiability of diffusion and trapping rates from
mass-uptake curves* (preprint, 2026).

Model: `u_t = D Δu − γu + βN`, `N_t = γu − βN`, with `u = n_∞` on the boundary and
`u = N = 0` at `t = 0`; the measured quantity is the total mass `M(t) = ∫(u + N)`.

- `sim/`: series solution and finite-difference solvers (`langmuir.py`), parameter sweeps
  (`sweep.py`), the three-step reconstruction and Levenberg–Marquardt fits (`inverse.py`),
  local and error/stability checks, with their outputs (`*.out`).
- `sim2/`: experiments A–F of the paper's numerical section (no inverse crime, rectangle,
  estimated saturated mass, stability constant, figure data, fit to the Carter–Kibler 1978
  curve). `sim2/README.txt` lists the exact commands, run times, random seeds and the
  scope of each run.

Requirements: Python 3 (tested with 3.14.4), standard library only. Run each script from
its folder, e.g. `cd sim2 && python3 -B A_noinvcrime.py slab,disk,ball 20 > A_noinvcrime.out`.

The digitised points of Carter and Kibler's Fig. 1 in `sim2/F_kc78.py` were read by eye
(see the comments there); the scanned figure is not included.

A finite computation certifies only the cases it ran.

## License

MIT; see `LICENSE`.
