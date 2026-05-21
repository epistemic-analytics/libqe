# pylibqe — Python bindings for libqe

nanobind bindings that expose the four libqe modules as a `pylibqe` Python
package, built with CMake + scikit-build-core.

## Submodules

| Module | Contents |
|---|---|
| `pylibqe.adjacency` | `choose_two`, `tri_indices`, `vector_to_upper_tri`, `directed_to_upper_tri`, `adjacency_matrix_to_vector`, `svector_to_upper_tri` |
| `pylibqe.normalization` | `sphere_norm`, `skip_sphere_norm` |
| `pylibqe.modeling` | `group_ci`, `outlier_ci`, `center_data`, `ena_correlation`, `lws_lsq_positions`, `directed_node_positions`, `directed_node_positions_ground_response`, `NodePositions` |
| `pylibqe.accumulation` | `calculate_adjacency_matrix`, `stanza_window`, `rows_to_co_occurrences`, `rolling_window_sum`, `calculate_1d_index` |

## Build dependencies

- Python ≥ 3.9
- numpy
- armadillo (system or conda-forge / homebrew)
- nanobind ≥ 1.9 (`pip install nanobind` or `brew install nanobind`)
- scikit-build-core ≥ 0.4 (`pip install scikit-build-core`)
- cmake ≥ 3.18

## Install (development)

```bash
cd libqe/python
pip install -e ".[dev]"
```

Or build a wheel:

```bash
pip wheel .
```

## Run tests

```bash
cd libqe/python
pytest tests/
```

## Usage example

```python
import numpy as np
from pylibqe import adjacency, normalization, modeling, accumulation

# Pair names for 3 codes
print(adjacency.svector_to_upper_tri(["X", "Y", "Z"]))
# ['X & Y', 'X & Z', 'Y & Z']

# Stanza-window accumulation
codes = np.array([[1,1,0],[1,0,1],[0,1,1]], dtype=float)
co = accumulation.stanza_window(codes, window_back=2, binary=True)
print(co)

# Sphere normalization
normed = normalization.sphere_norm(codes)
```
