# CMB Power Spectrum Emulator — Dataset Description

## Overview

The goal is to train a neural-network emulator that maps 6 ΛCDM cosmological parameters to four CMB angular power spectra (TT, TE, EE, φφ). The task and scoring rules are in `idea.md` (provided separately). This file describes only the data: what it contains, how to load it, and what each array means.

## Loading the data

The competition data ships with the `cmbemu` Python package (already installed). Always load via the API — do NOT read the .npz files directly:

```python
import cmbemu as cec

train = cec.load_train()   # 50,000 cosmologies
test  = cec.load_test()    # 5,000 cosmologies (held-out — do NOT train on this)

# Small variants for fast iteration
train_small = cec.load_train(size="small")   # 5,000 cosmologies
test_small  = cec.load_test(size="small")    # 500 cosmologies
```

Each returns a plain Python dict of NumPy arrays.

The underlying .npz files are cached at:
- `/home/node/.cache/cmbemu/datasets--borisbolliet--cmbemu-competition-v1/snapshots/e45fd9a3f451038e3ce677a56f7f6cb81c8c47c9/train.npz`
- `/home/node/.cache/cmbemu/datasets--borisbolliet--cmbemu-competition-v1/snapshots/e45fd9a3f451038e3ce677a56f7f6cb81c8c47c9/test.npz`

## Schema

Every dataset dict has these keys:

| Key           | Shape       | dtype   | Description |
|---------------|-------------|---------|-------------|
| `params`      | (N, 6)      | float32 | Cosmological parameter vectors |
| `param_names` | (6,)        | str     | Canonical parameter name ordering |
| `tt`          | (N, 6001)   | float32 | C_ℓ^TT for ℓ ∈ [0, 6000] |
| `te`          | (N, 6001)   | float32 | C_ℓ^TE for ℓ ∈ [0, 6000] |
| `ee`          | (N, 6001)   | float32 | C_ℓ^EE for ℓ ∈ [0, 6000] |
| `pp`          | (N, 3001)   | float32 | C_ℓ^φφ for ℓ ∈ [0, 3000] |
| `box_lo`      | (6,)        | float32 | Lower bounds of parameter prior |
| `box_hi`      | (6,)        | float32 | Upper bounds of parameter prior |
| `lmax_cmb`    | scalar      | int32   | 6000 |
| `lmax_pp`     | scalar      | int32   | 3000 |
| `seed`        | scalar      | int64   | RNG seed used for LHC sampling |

Index `i` is consistent across all arrays: `params[i]` is the parameter vector whose spectra are `tt[i]`, `te[i]`, `ee[i]`, `pp[i]`.

## Parameters

```python
cec.PARAM_NAMES  # ('omega_b', 'omega_cdm', 'H0', 'tau_reio', 'ln10^{10}A_s', 'n_s')
```

| Parameter       | Description              | Min   | Max  |
|-----------------|--------------------------|-------|------|
| omega_b         | Ω_b h² (baryon density)  | 0.020 | 0.025 |
| omega_cdm       | Ω_cdm h² (cold dark matter) | 0.09 | 0.15 |
| H0              | Hubble constant (km/s/Mpc) | 55  | 85   |
| tau_reio        | Reionization optical depth | 0.03 | 0.10 |
| ln10^{10}A_s    | Log primordial amplitude  | 2.7  | 3.3  |
| n_s             | Spectral tilt            | 0.92 | 1.02 |

Sampled on a Latin hypercube within this 6D box. The emulator only needs to be accurate inside these bounds.

## Spectra conventions

- All spectra are C_ℓ (not D_ℓ = ℓ(ℓ+1)C_ℓ/2π) — but the scoring is invariant to any consistent rescaling
- ℓ=0 and ℓ=1 entries are present but ignored by the scorer (scoring starts at ℓ=2)
- The 2×2 CMB covariance matrix at each ℓ is: [[C_ℓ^TT, C_ℓ^TE], [C_ℓ^TE, C_ℓ^EE]]
- This matrix must be positive definite: |C_ℓ^TE|² < C_ℓ^TT · C_ℓ^EE at every ℓ

## Dataset sizes

| Dataset     | N      | Memory  |
|-------------|--------|---------|
| train       | 50,000 | ~6.3 GB |
| test        | 5,000  | ~0.6 GB |
| train_small | 5,000  | ~0.6 GB |
| test_small  | 500    | ~60 MB  |

## Scoring API

Evaluate your emulator during development:

```python
import cmbemu as cec

# During training — fast, deterministic, no timing overhead
acc = cec.get_accuracy_score(emu)
mae = acc["mae_total"]["mae"]   # the number to minimize

# After model is frozen — measure CPU inference speed
tim = cec.get_time_score(emu)
t_ms = tim["t_cpu_ms_mean"]

# Final submission score (lower is better)
full = cec.get_score(emu)
print(full["combined_S"])
```

The emulator interface:
```python
class MyEmulator:
    def predict(self, params: dict) -> dict:
        # params: dict with keys from cec.PARAM_NAMES
        # returns: {"tt": array(6001,), "te": array(6001,), "ee": array(6001,), "pp": array(3001,)}
        ...
```

## Generating additional data

Additional training cosmologies can be generated from the same prior:

```python
extra = cec.generate_data(n=20_000, seed=42, save_to="/path/to/extra.npz")
```

Do NOT use seeds 202604 or 202605 (those reproduce the released train/test sets).
