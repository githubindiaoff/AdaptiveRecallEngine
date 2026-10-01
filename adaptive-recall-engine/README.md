# Adaptive Recall Engine

Return one trust weight for every dimension of a damaged query. A fixed recall
system uses those weights while attempting to recover the stored pattern from
which the query was created.

The controller does not return a class label or a repaired pattern. It returns
only the vector of per-dimension weights used by the frozen dynamics.

## Data used in one evaluation seed

For each seed, the evaluator creates:

- `K` stored patterns, collected in a matrix `X` with shape `(K, N)`;
- a symmetric model matrix `R` with shape `(N, N)`;
- fixed dynamics parameters;
- corrupted queries, each derived from one of the stored patterns; and
- the hidden source-pattern index for each query.

Stored patterns are unit-length vectors arranged in clusters. To create a
query, the generator sets a selected fraction of its source pattern's
dimensions to zero, adds Gaussian noise, and normalizes the result. The default
public configuration uses:

```text
K = 16 stored patterns
N = 64 dimensions
mask fractions = 0.60, 0.75, 0.85
250 queries per mask fraction and seed
```

New patterns, queries, and truth labels are generated independently for every
seed.

## What happens during evaluation

For each seed, the evaluator:

1. constructs the stored patterns and frozen recall model;
2. creates one fresh instance of your adapter;
3. calls `predict_precision(query)` once for every retrieval query;
4. runs the frozen dynamics using the returned weights;
5. classifies the final state against the stored patterns;
6. evaluates convergence balance on a sample of stored patterns; and
7. aggregates the results with the other seeds.

The adapter is recreated when the seed changes. It receives the stored patterns
and model parameters but never receives the source-pattern labels.

## Submission contract

Create `adapters/myteam.py` containing:

```python
from adapter import Adapter
import numpy as np


class Engine(Adapter):
    def __init__(self, stored_patterns, model_params):
        self.X = stored_patterns
        self.N = stored_patterns.shape[1]

    def predict_precision(self, corrupted_query):
        return np.ones(self.N)
```

The constructor arguments are:

| Argument | Meaning |
| --- | --- |
| `stored_patterns` | Float array of shape `(K, N)` |
| `model_params["R"]` | Frozen model matrix of shape `(N, N)` |
| `eta`, `beta` | Energy-model parameters |
| `dt` | Integration step size |
| `T_max` | Maximum number of integration steps |
| `tol` | Early-convergence tolerance |
| `T_in` | Number of initial steps for which the damaged query remains an input |
| `pi_min`, `pi_max` | Permitted bounds for each submitted weight |

For every call, `corrupted_query` is a float vector with shape `(N,)`.
`predict_precision` must return exactly `N` finite numeric values. Treat all
constructor inputs and queries as read-only.

Before use, the evaluator converts the returned vector to `float64`, clips it
to `[pi_min, pi_max]`, and rescales it toward mean `1`. A non-finite vector is
replaced with the all-ones baseline.

## Frozen recall dynamics

The recall state begins at the corrupted query:

```text
a(0) = corrupted_query
```

At each integration step, the model calculates:

```text
s(a)       = softmax(beta * X * a)
gradient   = R * a - eta * X^T * s(a)
external(t) = corrupted_query, if t < T_in
              0,               otherwise
update       = -precision * gradient + external(t)
```

The state is then updated by:

```text
a(next) = a + dt * update
```

Integration stops after `T_max` steps or when the change in state is below
`tol`. The final state is normalized and classified as the stored pattern with
the greatest cosine similarity. Retrieval is correct only when that index
equals the hidden source-pattern index.

## Automated metrics

### Retrieval accuracy — 70 points

The evaluator first measures retrieval with the all-ones precision vector. For
your adapter, it computes the accuracy difference for each seed:

```text
delta = adapter_accuracy - all_ones_accuracy
```

The mean delta is scaled linearly from zero points at `0.00` to all 70 points
at `+0.08`. Values above `+0.08` remain capped at 70 points. If mean delta is
not positive, this section scores zero. If any evaluated seed has negative
delta, the retrieval points are halved.

The report also shows direct cosine-classification accuracy for context. This
diagnostic does not itself change the score.

### Convergence balance — 20 points

For sampled stored patterns, the evaluator finds the equilibrium reached under
uniform precision and calculates the Hessian `H` at that point. Your adapter is
called on a lightly perturbed version of the pattern. With its normalized
precision vector `pi`, the evaluator measures the eigenvalue spread of:

```text
sqrt(pi) * H * sqrt(pi)
```

For each sampled pattern, reduction is:

```text
uniform_spread / adapter_spread
```

A reduction of `1` means no improvement in spread. The mean reduction is scored
logarithmically from zero points at `1x` to all 20 points at `5x`. If the mean
is not above `1x`, this section scores zero. If any evaluated seed has reduction
of `1x` or less, the balance points are halved.

### Code quality — 10 points

The remaining 10 points are reviewed manually for readable, reproducible, and
clearly explained code. The automated maximum is 90 points.

The public runner is a practice result. Official grading uses fresh unpublished
seeds and may use larger pattern sets.

## Submission integrity

AI coding tools are allowed, but your adapter must be a genuine precision
controller that generalizes across newly generated pattern collections,
queries, and seeds. Submit exactly one readable Python adapter file. The
organizers will review, hash, and rerun that exact file.

Treat `stored_patterns`, `model_params`, and every `corrupted_query` as
read-only. Do not inspect caller frames, truth labels, evaluator locals,
closures, globals, or other runtime internals. Do not modify model objects,
queries, NumPy state used by the evaluator, reports, clocks, limits, or scoring
state. Seed-specific lookup tables, reconstructed truth sequences,
monkeypatches, reflection, dynamic imports, `eval`/`exec`, subprocesses,
filesystem or network access, environment inspection, native-code loading, and
encoded or obfuscated payloads are prohibited.

Normal numerical code using NumPy is allowed. Embedded learned parameters are
allowed when they form a disclosed, seed-independent model; data keyed to the
published seeds or queries is not. Unreadable or unexplained generated code may
be rejected, and a rules violation may disqualify a submission regardless of
its reported score.

## Running locally

```bash
pip install -r requirements.txt
python self_check.py --adapter adapters.myteam:Engine --quick
python run.py --adapter adapters.myteam:Engine --out report.json
```

The quick check uses two seeds, two noise levels, fewer queries, and fewer
balance probes. The normal runner uses the full public configuration and writes
the complete per-seed report.

## Relevant files

```text
adapter.py         submission interface
recovery_model.py  frozen recall engine
data.py            stored-pattern and corrupted-query generation
metrics.py         retrieval and convergence-balance calculations
harness.py         multi-seed orchestration and scoring
adapters/          examples and your submitted adapter location
```
