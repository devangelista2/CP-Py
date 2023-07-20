import numpy as np

import operators
import solvers
import utils

# Params
sigma = 1
eta = 2e-3
p = 1

# Load data
x_true = np.load("gt.npy")[0, 0]
mx, nx = x_true.shape

D1 = operators.GradientOperator((mx, nx))
D2 = operators.myGradient(1, (mx, nx))

dx1 = D1(x_true)
dx2 = D2(x_true)

grad_mag1 = utils.gradient_magnitude(dx1)
grad_mag2 = np.square(dx2[: len(dx2) // 2]) + np.square(dx2[len(dx2) // 2 :])

yy1 = np.zeros((2, mx, nx)) + sigma * dx1
yy2 = np.zeros((2 * mx * nx, 1)) + sigma * np.expand_dims(dx2, 1)

W1 = np.expand_dims(np.power(np.sqrt(eta**2 + grad_mag1) / eta, p - 1), 0)
W1 = np.repeat(W1, 2, axis=0)

W2 = np.expand_dims(np.power(np.sqrt(eta**2 + grad_mag2) / eta, p - 1), -1)
W2 = np.concatenate((W2, W2), axis=0)

y1 = utils.prox_l1_reweighted(yy1, W1, lmbda=1)

abs_ww = np.zeros((mx * nx, 1))
abs_ww = np.square(yy2[: mx * nx]) + np.square(yy2[mx * nx :])
abs_ww = np.concatenate((abs_ww, abs_ww), axis=0)

lmbda_vec_over_nu = 1 * W2
y2 = lmbda_vec_over_nu * yy2 / np.maximum(lmbda_vec_over_nu, abs_ww)


diff = y1.flatten() - y2.flatten()
print(np.linalg.norm(diff.flatten(), 2))
