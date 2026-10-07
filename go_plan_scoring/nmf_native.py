"""Optional sparse-support prediction kernel, compiled locally with C99.

No Python extension ABI or third-party build dependency. A C compiler is
optional: the caller can fall back to NumPy. Only the dot products at observed
edges are computed; the full KL zero-cell contribution is unchanged.
"""
import ctypes
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np

_SOURCE = r'''
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
static double dot(int64_t k, const double *ui, const double *vj) {
    double s0=0.,s1=0.,s2=0.,s3=0.;
    int64_t j=0;
    for (; j+3<k; j+=4) {
        s0+=ui[j]*vj[j]; s1+=ui[j+1]*vj[j+1];
        s2+=ui[j+2]*vj[j+2]; s3+=ui[j+3]*vj[j+3];
    }
    double sum=(s0+s1)+(s2+s3);
    for (; j<k; ++j) sum += ui[j]*vj[j];
    return sum;
}
void predict_csr(int64_t nr, int64_t k, const int64_t *ptr,
                 const int64_t *col, const double *u,
                 const double *v, double *out) {
    #pragma omp parallel for schedule(dynamic, 64)
    for (int64_t i=0; i<nr; ++i) {
        const double *ui = u+i*k;
        for (int64_t e=ptr[i]; e<ptr[i+1]; ++e) {
            const double *vj = v+col[e]*k;
            out[e]=dot(k,ui,vj);
        }
    }
}
/* Rows are independent and each row's arithmetic order is unchanged, so the
   OpenMP result is bitwise identical to the serial one for any thread count. */
int update_csr(int64_t nr, int64_t k, const int64_t *ptr,
               const int64_t *col, const double *data, double *restrict u,
               const double *restrict v, const double *total, const double *beta) {
    int failed = 0;
    #pragma omp parallel
    {
        double *num=malloc(k*sizeof(double));
        if (!num) {
            #pragma omp atomic write
            failed = 1;
        }
        #pragma omp for schedule(dynamic, 64)
        for (int64_t i=0; i<nr; ++i) {
            if (!num) continue;
            double *ui=u+i*k;
            memset(num,0,k*sizeof(double));
            for (int64_t e=ptr[i]; e<ptr[i+1]; ++e) {
                const double *vj=v+col[e]*k;
                double pred=dot(k,ui,vj);
                double ratio=data[e]/(pred>DBL_MIN ? pred:DBL_MIN);
                for (int64_t j=0; j<k; ++j) num[j]+=ratio*vj[j];
            }
            for (int64_t j=0; j<k; ++j) {
                double den=total[j]+beta[j]*ui[j];
                ui[j]*=num[j]/(den>DBL_MIN ? den:DBL_MIN);
            }
        }
        free(num);
    }
    return failed;
}
'''


@lru_cache(maxsize=1)
def _library():
    if sys.platform not in {'linux', 'darwin'}:
        raise RuntimeError('Native NMF currently supports Linux and macOS; use backend="numpy"')
    compiler = next((shutil.which(c) for c in ['cc', 'clang', 'gcc'] if shutil.which(c)), None)
    if compiler is None:
        raise RuntimeError('No C compiler found; use backend="numpy" or install a C compiler')
    temporary = tempfile.TemporaryDirectory(prefix='go-nmf-native-')
    folder = Path(temporary.name)
    source = folder/'predict.c'; source.write_text(_SOURCE)
    library = folder/('predict.dylib' if sys.platform=='darwin' else 'predict.so')
    base = [compiler, '-O3', '-std=c99', '-ffp-contract=off', '-fPIC',
            '-dynamiclib' if sys.platform=='darwin' else '-shared']
    loaded = error = None
    # Try OpenMP first (multi-threaded rows; OMP_NUM_THREADS controls threads),
    # then the plain serial build when the toolchain lacks OpenMP.
    for extra in (['-fopenmp'], []):
        try:
            subprocess.run(base + extra + [str(source), '-o', str(library)],
                           check=True, capture_output=True, text=True, timeout=60)
            loaded = ctypes.CDLL(str(library))
            break
        except Exception as exc:
            error = exc
    if loaded is None:
        temporary.cleanup()
        raise RuntimeError('Native NMF compilation failed; use backend="numpy"') from error
    function = loaded.predict_csr
    integer = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS')
    matrix = np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
    vector = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    function.argtypes = [ctypes.c_int64,ctypes.c_int64,integer,integer,matrix,matrix,vector]
    function.restype = None
    update = loaded.update_csr
    update.argtypes = [ctypes.c_int64,ctypes.c_int64,integer,integer,vector,matrix,matrix,vector,vector]
    update.restype = ctypes.c_int
    # Keep both objects alive as long as ctypes uses their code.
    return function, update, loaded, temporary


class SparsePredictor:
    def __init__(self, x):
        self.function, self.update_function, self.library, self.temporary = _library()
        self.shape = x.shape
        self.indptr = np.ascontiguousarray(x.indptr, dtype=np.int64)
        self.indices = np.ascontiguousarray(x.indices, dtype=np.int64)
        self.data = np.ascontiguousarray(x.data, dtype=np.float64)
        transpose = x.T.tocsr()
        self.t_indptr = np.ascontiguousarray(transpose.indptr, dtype=np.int64)
        self.t_indices = np.ascontiguousarray(transpose.indices, dtype=np.int64)
        self.t_data = np.ascontiguousarray(transpose.data, dtype=np.float64)

    def update(self, u, v, beta):
        penalty = np.zeros(u.shape[1]) if beta is None else beta
        status = self.update_function(self.shape[1],u.shape[1],self.t_indptr,self.t_indices,self.t_data,
                                      v,u,np.ascontiguousarray(u.sum(0)),penalty)
        if status:
            raise MemoryError('Native NMF numerator allocation failed')
        status = self.update_function(self.shape[0],u.shape[1],self.indptr,self.indices,self.data,
                                      u,v,np.ascontiguousarray(v.sum(0)),penalty)
        if status:
            raise MemoryError('Native NMF numerator allocation failed')

    def __call__(self, x, u, v):
        if x.shape != self.shape or len(x.data) != len(self.indices):
            raise ValueError('SparsePredictor must be used with its original support')
        u = np.ascontiguousarray(u, dtype=np.float64)
        v = np.ascontiguousarray(v, dtype=np.float64)
        if u.shape[0] != self.shape[0] or v.shape[0] != self.shape[1] or u.shape[1] != v.shape[1]:
            raise ValueError('Factors do not match sparse support')
        result = np.empty(len(self.indices), dtype=np.float64)
        self.function(self.shape[0],u.shape[1],self.indptr,self.indices,u,v,result)
        return result
