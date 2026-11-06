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
- `markers/`: code for L. Marazzi, *Markers and invisibility in mass-uptake identification of
  trapping materials* (preprint, 2026). `fast_exchange_check.py`, `lower_bound_check.py` and
  `slow_release_check.py` check the fast-exchange upper and lower bounds and the slow-release
  bound on the slab (Theorems 4.2, 4.4 and 5.2 of that paper). `fit_bristol.py` and
  `classify_bristol_v2.py` fit Fickian and Carter–Kibler slab models to open flax/epoxy
  immersion data and run the classification of its Section 7. Each script's output is stored
  next to it (`*.out`); run them from `markers/`, e.g. `python3 -B fit_bristol.py > fit_bristol.out`.
  `classify_bristol_v2.py` takes about 15 minutes (permutation tests).
- `markers/data-bristol/`: three CSV files copied unchanged from Li et al., Mendeley Data,
  doi:10.17632/t7fr46vjcs.2, licensed CC BY 4.0 (not MIT); see `markers/data-bristol/README.txt`
  for the source, hashes and what the files contain.

Requirements: Python 3 (tested with 3.14.4), standard library only. Run each script from
its folder, e.g. `cd sim2 && python3 -B A_noinvcrime.py slab,disk,ball 20 > A_noinvcrime.out`.

The digitised points of Carter and Kibler's Fig. 1 in `sim2/F_kc78.py` were read by eye
(see the comments there); the scanned figure is not included.

A finite computation certifies only the cases it ran.

## License

MIT for the code; see `LICENSE`. The data files in `markers/data-bristol/` are third-party data
under CC BY 4.0, credited to Li, Sun, Le Duigou, Guo, Rossiter, Liu, Leng and Scarpa (2022).
