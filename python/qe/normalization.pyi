"""Row-wise L2 normalization"""

from typing import Annotated

import numpy
from numpy.typing import NDArray


def normalize_networks(m: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Divide each row by its own L2 norm (project onto unit hypersphere). Zero rows are left unchanged.
    """

def scale_networks(m: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu')]) -> Annotated[NDArray[numpy.float64], dict(shape=(None, None))]:
    """
    Divide all entries by the *largest* row L2 norm. Preserves relative magnitudes across rows.
    """
