# Import libraries
import astra
import numpy as np
import scipy
import scipy.signal
from numpy.fft import fft2, fftshift, ifft2


class Operator:
    r"""
    The main class of the library. It defines the abstract Operator that will be subclassed for any specific case.
    """

    def __call__(self, x):
        return self._matvec(x)

    def __matmul__(self, x):
        return self._matvec(x)

    def T(self, x):
        return self._adjoint(x)


class Identity(Operator):
    r"""
    Defines the Identity operator (i.e. an operator that does not affect the input data).
    """

    def __init__(self, lmbda, img_shape):
        super().__init__()
        self.lmbda = lmbda
        self.n = img_shape[0]

    def _matvec(self, x):
        return (self.lmbda * x).flatten()

    def _adjoint(self, x):
        return (self.lmbda * x).flatten()


class TikhonovOperator(Operator):
    """
    Given matrices A and L, returns the operator that acts like [A; L], concatenated vertically.
    """

    def __init__(self, A, L):
        super().__init__()
        self.A = A
        self.L = L

        self.n = A.n
        self.shape = (self.n**2, self.n**2)

    def _matvec(self, x):
        x = x.reshape((self.n, self.n))

        Ax = self.A @ x
        Lx = self.L @ x

        Ax = Ax.flatten()
        Lx = Lx.flatten()

        return np.concatenate([Ax, Lx], axis=0)

    def _adjoint(self, x):
        x1 = x[: self.n**2]
        x2 = x[self.n**2 :]

        x1 = x1.reshape((self.n, self.n))
        x2 = x2.reshape((self.n, self.n))

        ATx1 = self.A.T(x1)
        LTx2 = self.L.T(x2)

        return ATx1 + LTx2


class GradientOperator(Operator):
    def __init__(self, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.shape = (img_shape[0] * img_shape[1], img_shape[0] * img_shape[1])

    def _matvec(self, x):
        D = np.zeros((2,) + self.img_shape)
        D[0] = np.diff(x.reshape(self.img_shape), n=1, axis=1, prepend=0)
        D[1] = np.diff(x.reshape(self.img_shape), n=1, axis=0, prepend=0)
        return D

    def _adjoint(self, y):
        D_h = y[0]
        D_v = y[1]

        D_h_T = np.fliplr(np.diff(np.fliplr(D_h), n=1, axis=1, prepend=0))
        D_v_T = np.flipud(np.diff(np.flipud(D_v), n=1, axis=0, prepend=0))
        return D_h_T + D_v_T

    def magnitude(self, D):
        D_h = D[0]
        D_v = D[1]

        return np.sqrt(np.square(D_h) + np.square(D_v))


class CTProjector(Operator):
    def __init__(self, m, n, angles, det_size=None, geometry="parallel"):
        super().__init__()
        # Input setup
        self.m = m
        self.n = n

        # Geometry
        self.geometry = geometry

        # Projector setup
        if det_size is None:
            self.det_size = max(self.n, self.m) * np.sqrt(2)
        else:
            self.det_size = det_size
        self.angles = angles
        self.n_angles = len(angles)

        # Define projector
        self.proj = self.get_astra_projection_operator()
        self.shape = self.proj.shape

    # ASTRA Projector
    def get_astra_projection_operator(self):
        # create geometries and projector
        if self.geometry == "parallel":
            proj_geom = astra.create_proj_geom(
                "parallel", 1.0, self.det_size, self.angles
            )
            vol_geom = astra.create_vol_geom(self.m, self.n)
            proj_id = astra.create_projector("linear", proj_geom, vol_geom)

        elif self.geometry == "fanflat":
            proj_geom = astra.create_proj_geom(
                "fanflat", 1.0, self.det_size, self.angles, 1800, 500
            )
            vol_geom = astra.create_vol_geom(self.m, self.n)
            proj_id = astra.create_projector("cuda", proj_geom, vol_geom)

        else:
            print("Geometry (still) undefined.")
            return None

        return astra.OpTomo(proj_id)

    # On call, project
    def _matvec(self, x):
        y = self.proj @ x.flatten()
        return y

    def _adjoint(self, y):
        x = self.proj.T @ y.flatten()
        return x.reshape((self.m, self.n))

    # FBP
    def FBP(self, y):
        x = self.proj.reconstruct("FBP", y.flatten())
        return x.reshape((self.m, self.n))


class ConcatenateOperator(Operator):
    def __init__(self, A, B):
        super().__init__()
        self.A = A
        self.B = B

        self.mA, self.nA = A.shape
        self.mB, self.nB = B.shape

        self.shape = (self.mA + self.mB, self.nA)

    def _matvec(self, x):
        y1 = self.A(x)
        y2 = self.B(x)
        return np.concatenate((y1, y2), axis=0)

    def _adjoint(self, y):
        y1 = y[: self.mA]
        y2 = y[self.mA :]

        x1 = self.A.T(y1)
        x2 = self.B.T(y2)
        return x1 + x2


class MatrixOperator(Operator):
    def __init__(self, A):
        super().__init__()
        self.A = A
        self.shape = self.A.shape

    def _matvec(self, x):
        return self.A @ x.flatten()

    def _adjoint(self, y):
        return self.A.T @ y.flatten()


class myGradient(Operator):
    def __init__(self, lmbda, img_shape):
        super().__init__()
        self.img_shape = img_shape
        self.lmbda = lmbda
        self.shape = (img_shape[0] * img_shape[1], img_shape[0] * img_shape[1])

    def _matvec(self, x):
        D_h = np.diff(x.reshape(self.img_shape), n=1, axis=1, prepend=0).flatten()
        D_v = np.diff(x.reshape(self.img_shape), n=1, axis=0, prepend=0).flatten()
        return np.concatenate((D_h, D_v), axis=0)

    def _adjoint(self, y):
        y = y.flatten()
        D_h = y[: len(y) // 2].reshape(self.img_shape)
        D_v = y[len(y) // 2 :].reshape(self.img_shape)

        D_h_T = np.fliplr(np.diff(np.fliplr(D_h), n=1, axis=1, prepend=0)).flatten()
        D_v_T = np.flipud(np.diff(np.flipud(D_v), n=1, axis=0, prepend=0)).flatten()
        return D_h_T + D_v_T
