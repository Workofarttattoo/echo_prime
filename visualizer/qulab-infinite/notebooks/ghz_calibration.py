# %%
# Qulab Infinite GHZ Calibration Notebook (Jupytext-compatible)
#
# 1. Set DATABASE_URL environment variable before launching Jupyter.
# 2. Ensure Poetry environment is activated (use `poetry run jupyter lab`).

# %%
import os
from pathlib import Path

import arviz as az
import numpy as np
import pymc as pm
import sqlalchemy as sa
from matplotlib import pyplot as plt

from qulab_infinite.units import get_unit_registry

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required.")

engine = sa.create_engine(DATABASE_URL)

# %%
# Pull GHZ fidelity observations from Postgres

query = sa.text(
    """
    SELECT value_si, sigma, context_tags
    FROM measurements
    WHERE quantity = :quantity
    ORDER BY date_ingested ASC
    """
)

records = []
with engine.connect() as conn:
    result = conn.execute(query, {"quantity": "ghz_state_fidelity"})
    for row in result.mappings():
        records.append(row)

if not records:
    raise RuntimeError("No GHZ records found. Run run-ghz-benchmark.py first.")

values = np.array([row["value_si"] for row in records], dtype=float)
sigmas = np.array([row["sigma"] or 0.01 for row in records], dtype=float)

# %%
# PyMC Model: simple normal around latent fidelity with measurement noise

with pm.Model() as model:
    fidelity_mu = pm.Beta("fidelity_mu", alpha=10, beta=2)
    fidelity_sigma = pm.HalfNormal("fidelity_sigma", sigma=0.05)
    discrepancy = pm.Normal("discrepancy", mu=0, sigma=fidelity_sigma, shape=len(values))
    obs = pm.Normal(
        "obs",
        mu=fidelity_mu + discrepancy,
        sigma=sigmas,
        observed=values,
    )
    idata = pm.sample(draws=2000, tune=1000, target_accept=0.9, chains=2)

# %%
# Posterior diagnostics

az.plot_trace(idata, var_names=["fidelity_mu", "fidelity_sigma"])
plt.tight_layout()

# %%
# Posterior predictive mean and 5–95% interval

posterior_samples = idata.posterior["fidelity_mu"].values.flatten()
mean = posterior_samples.mean()
ci_low, ci_high = np.percentile(posterior_samples, [5, 95])

print(f"Posterior mean fidelity: {mean:.4f}")
print(f"90% credible interval: [{ci_low:.4f}, {ci_high:.4f}]")

# %%
# Save summary to artifacts

summary_dir = Path("../../artifacts/calibration")
summary_dir.mkdir(parents=True, exist_ok=True)
az.to_netcdf(idata, summary_dir / "ghz_fidelity_posterior.nc")

with (summary_dir / "ghz_fidelity_summary.txt").open("w", encoding="utf-8") as handle:
    handle.write(f"Posterior mean: {mean:.6f}\n")
    handle.write(f"CI 5-95: {ci_low:.6f}, {ci_high:.6f}\n")
    handle.write(f"Samples: {len(posterior_samples)}\n")

print(f"[info] Wrote posterior bundle to {summary_dir}")
