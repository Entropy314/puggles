# distutils: extra_compile_args = -O3 -fopenmp
# distutils: extra_link_args = -fopenmp
# distutils: libraries = gomp
"""OpenMP batch evaluator for the Puggles Python API.

Each OpenMP worker evaluates one candidate.  The module is intentionally
separate from the notebook so multiprocessing and notebook re-execution stay
reliable.
"""

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
cimport cython

cdef extern from "omp.h":
    int omp_get_max_threads()


cpdef int openmp_max_threads():
    """Return the OpenMP worker limit used by the compiled batch kernel."""
    return omp_get_max_threads()


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cnp.ndarray[cnp.float64_t, ndim=2] evaluate_population(
    cnp.ndarray[cnp.float64_t, ndim=2] population,
):
    cdef Py_ssize_t count = population.shape[0]
    cdef Py_ssize_t index
    cdef cnp.ndarray[cnp.float64_t, ndim=2] result = np.empty((count, 1), dtype=np.float64)
    cdef double[:, ::1] inputs = population
    cdef double[:, ::1] objectives = result

    with nogil:
        for index in prange(count, schedule="static"):
            objectives[index, 0] = (
                (inputs[index, 0] - 3.0) ** 2 + (inputs[index, 1] - 6.0) ** 2
            )

    return result
