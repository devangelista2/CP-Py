import numpy as np


def power_method(K, gradient, maxit=10):
    """
    Calculates the norm of operator M = [grad, K],
    i.e the sqrt of the largest eigenvalue of M^T*M = -div(grad) + M^T*M :
        ||M|| = sqrt(lambda_max(M^T*M))

    K : forward projection
    y : acquired data
    maxit : number of iterations to perform (default: 10)
    """
    mx, nx = K.m, K.n

    x = np.random.rand(mx, nx)
    for _ in range(0, maxit):
        x = K.T(K(x)) + gradient.T(gradient(x))
        x_norm = np.linalg.norm(x.flatten(), 2)
        x /= x_norm
    return np.sqrt(x_norm)


def proj_l2(D):
    """
    Proximal operator of the L2,1 norm :
        L2,1(u) = sum_i ||u_i||_2
    i.e pointwise projection onto the L2 unit ball

    D : gradient-like numpy array
    """
    n = np.maximum(np.sqrt(np.sum(D**2, 0)), 1.0)
    D[0] = D[0] / n
    D[1] = D[1] / n
    return D


def gradient_magnitude(z):
    """
    Given a gradient-shaped array z (shape: 2 x m x n), returns an array of shape m x n implementing the |z| = sqrt(z[0]^2 + z[1]^2).
    """
    return np.square(z[0]) + np.square(z[1])


def prox_l1_reweighted(z, W, lmbda):
    """
    Proximal of the operator || w |z| ||_1, where z is a gradient-shaped array (shape: 2 x m x n),
    |z| = sqrt(z[0]^2 + z[1]^2), w weighting factor, and lmbda is the regularization parameter.
    """
    abs_z = np.repeat(np.expand_dims(gradient_magnitude(z), 0), 2, axis=0)
    lmbda_W = lmbda * W
    z = lmbda_W * z / np.maximum(lmbda_W, abs_z)
    return z
