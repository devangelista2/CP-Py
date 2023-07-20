import numpy as np

import operators
import utils


class ChambollePockTV:
    def __init__(self, K):
        self.K = K

        # Get the shape of x and the shape of y from K
        self.mx, self.nx = self.K.m, self.K.n

        # Initialization
        self.gradient = operators.GradientOperator((self.mx, self.nx))

        # Compute the 2-norm of M = [grad; K]
        self.L = utils.power_method(self.K, self.gradient, maxit=10)

        self.sigma = 1.0 / self.L
        self.tau = 1.0 / self.L
        self.theta = 1.0

    def __call__(
        self, y_delta, lmbda, x0=None, maxit=200, tolf=5e-4, tolx=1e-5, verbose=False
    ):
        """
        Chambolle-Pock algorithm for the minimization of the objective function
            ||K*x - d||_2^2 + Lambda*TV(x)

        K : projection operator
        Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
        L : norm of the operator [P, Lambda*grad] (see power_method)
        maxit : number of iterations
        """
        # Initialization
        if x0 is None:
            x = np.zeros((self.mx, self.nx))
        else:
            x = x0.reshape((self.mx, self.nx))
        s = np.zeros((2, self.mx, self.nx))
        q = np.zeros_like(y_delta)
        x_tilde = x

        k = 0
        stopping = False
        while not stopping:
            # Update dual variables
            s = utils.proj_l2(s + self.sigma * self.gradient(x_tilde))
            q = (q + self.sigma * self.K(x_tilde) - self.sigma * y_delta) / (
                1.0 + lmbda * self.sigma
            )

            # Update primal variables
            x_old = x
            x = x - self.tau * self.gradient.T(s) - self.tau * self.K.T(q)
            x_tilde = x + self.theta * (x - x_old)

            # Update k
            k = k + 1

            # Compute residual and iterates distance for stopping conditions
            res = np.linalg.norm(self.K(x) - y_delta) / np.max(y_delta)
            dist = np.linalg.norm(x_old.flatten() - x.flatten()) / np.linalg.norm(
                x.flatten()
            )

            # Update stopping condition
            stopping = (
                (dist < tolx) or ((res / np.sqrt(len(y_delta))) < tolf) or (k > maxit)
            )

        if verbose:
            print("\n---------------------------")
            if k > maxit:
                print("Algorithm didn't converged.")
            elif dist < tolx:
                print("Algorithm converged with x-condition.")
            elif (res / np.sqrt(len(y_delta))) < tolf:
                print("Algorithm converged with f-condition.")
            print(f"Iterations: \t   {k}.")
            print(f"Relative Distance: {dist:0.5f}.")
            print(f"Residual: \t   {res / np.sqrt(len(y_delta)):0.5f}.")
        return x


class ChambollePockTpV:
    def __init__(self, K):
        self.K = K

        # Get the shape of x and the shape of y from K
        self.mx, self.nx = self.K.m, self.K.n

        # Initialization
        self.gradient = operators.GradientOperator((self.mx, self.nx))

        # Compute the 2-norm of M = [grad; K]
        self.L = utils.power_method(self.K, self.gradient, maxit=10)

        self.sigma = 1.0 / self.L
        self.tau = 1.0 / self.L
        self.theta = 1.0
        self.eta = 2e-3

    def __call__(
        self,
        y_delta,
        lmbda,
        x0=None,
        x_approx=None,
        x_true=None,
        p=1,
        maxit=200,
        tolf=5e-4,
        tolx=1e-5,
        verbose=False,
        return_obj=False,
    ):
        """
        Chambolle-Pock algorithm for the minimization of the objective function
            ||K*x - d||_2^2 + Lambda*TpV(x)
        by reweighting

        K : projection operator
        Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
        L : norm of the operator [P, Lambda*grad] (see power_method)
        maxit : number of iterations
        """
        # Initialization
        if return_obj:
            obj = np.zeros((maxit + 1,))

        if x0 is None:
            x0 = np.zeros((self.mx, self.nx))
        if x_approx is None:
            x_approx = x0

        x = x0
        s = np.zeros((2, self.mx, self.nx))
        q = np.zeros_like(y_delta)
        x_tilde = x

        # Compute the gradient magnitude of x_approx, for the reweighting
        grad = self.gradient(x_approx)
        grad_mag = np.sqrt(np.square(grad[0]) + np.square(grad[1]))

        # Compute the reweighting factor
        self.W = np.expand_dims(
            np.power(np.sqrt(self.eta**2 + grad_mag) / self.eta, p - 1), 0
        )
        self.W = np.repeat(self.W, 2, axis=0)

        # Compute the first value of the objective function
        if return_obj:
            obj[0] = self.obj_function(x0, y_delta, lmbda)

        k = 0
        stopping = False
        while not stopping:
            # Update dual variables
            s = utils.prox_l1_reweighted(
                s + self.sigma * self.gradient(x_tilde), self.W, lmbda
            )
            q = (q + self.sigma * (self.K(x_tilde) - y_delta)) / (
                1.0 + lmbda * self.sigma
            )

            # Update primal variables
            x_old = x
            x = x - self.tau * self.gradient.T(s) - self.tau * self.K.T(q)
            x_tilde = x + self.theta * (x - x_old)

            # Update k
            k = k + 1

            # Update objective function (if required)
            if return_obj:
                obj[k] = self.obj_function(x, y_delta, lmbda)
            # Print rel.err.
            if verbose and (x_true is not None):
                rel_err = np.linalg.norm(
                    x.flatten() - x_true.flatten()
                ) / np.linalg.norm(x_true.flatten())
                print(f"{k=}: Rel. Err. = {rel_err:0.4f}.")

            # Compute residual and iterates distance for stopping conditions
            res = np.linalg.norm(self.K(x) - y_delta) / np.max(y_delta)
            dist = np.linalg.norm(x_old.flatten() - x.flatten()) / np.linalg.norm(
                x.flatten()
            )

            # Update stopping condition
            stopping = (
                (dist < tolx) or ((res / np.sqrt(len(y_delta))) < tolf) or (k >= maxit)
            )

        if verbose:
            print("\n---------------------------")
            if k >= maxit:
                print("Algorithm didn't converged.")
            elif dist < tolx:
                print("Algorithm converged with x-condition.")
            elif (res / np.sqrt(len(y_delta))) < tolf:
                print("Algorithm converged with f-condition.")
            print(f"Iterations: \t   {k}.")
            print(f"Relative Distance: {dist:0.5f}.")
            print(f"Residual: \t   {res / np.sqrt(len(y_delta)):0.5f}.")

        if return_obj:
            return x, obj[:k]
        return x

    def obj_function(self, x, y_delta, lmbda):
        # Compute reweighted gradient of x
        grad_x = self.gradient(x)
        grad_mag = utils.gradient_magnitude(grad_x)

        # Compute the residual and the regularization term
        res = np.linalg.norm(self.K(x).flatten() - y_delta.flatten(), 2)
        tpv = np.sum(np.abs(self.W[0] * grad_mag))

        return res + lmbda * tpv


class IRChambollePockTpV:
    def __init__(self, K):
        self.K = K

        # Get the shape of x and the shape of y from K
        self.mx, self.nx = self.K.m, self.K.n

        # Initialization
        self.gradient = operators.GradientOperator((self.mx, self.nx))

        # Compute the 2-norm of M = [grad; K]
        self.L = utils.power_method(self.K, self.gradient, maxit=10)

        self.sigma = 1.0 / self.L
        self.tau = 1.0 / self.L
        self.theta = 1.0
        self.eta = 2e-3

    def __call__(
        self,
        y_delta,
        lmbda,
        x0=None,
        p=1,
        maxit=200,
        tolf=5e-4,
        tolx=1e-5,
        verbose=False,
    ):
        """
        Chambolle-Pock algorithm for the minimization of the objective function
            ||K*x - d||_2^2 + Lambda*TpV(x)
        by reweighting

        K : projection operator
        Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
        L : norm of the operator [P, Lambda*grad] (see power_method)
        maxit : number of iterations
        """
        # Initialization
        if x0 is None:
            x = np.zeros((self.mx, self.nx))
        else:
            x = x0.reshape((self.mx, self.nx))
        s = np.zeros((2, self.mx, self.nx))
        q = np.zeros_like(y_delta)
        x_tilde = x

        k = 0
        stopping = False
        while not stopping:
            # Compute the gradient magnitude of x_approx, for the reweighting
            grad = self.gradient(x_tilde)
            grad_mag = utils.gradient_magnitude(grad)

            # Compute the reweighting factor
            W = np.expand_dims(
                np.power(np.sqrt(self.eta**2 + grad_mag) / self.eta, p - 1), 0
            )
            W = np.repeat(W, 2, axis=0)

            # Update dual variables
            s = utils.prox_l1_reweighted(
                s + self.sigma * self.gradient(x_tilde), W, lmbda
            )
            q = (q + self.sigma * self.K(x_tilde) - self.sigma * y_delta) / (
                1.0 + lmbda * self.sigma
            )

            # Update primal variables
            x_old = x
            x = x - self.tau * self.gradient.T(s) - self.tau * self.K.T(q)
            x_tilde = x + self.theta * (x - x_old)

            # Update k
            k = k + 1

            # Compute residual and iterates distance for stopping conditions
            res = np.linalg.norm(self.K(x) - y_delta) / np.max(y_delta)
            dist = np.linalg.norm(x_old.flatten() - x.flatten()) / np.linalg.norm(
                x.flatten()
            )

            # Update stopping condition
            stopping = (
                (dist < tolx) or ((res / np.sqrt(len(y_delta))) < tolf) or (k > maxit)
            )

        if verbose:
            print("\n---------------------------")
            if k > maxit:
                print("Algorithm didn't converged.")
            elif dist < tolx:
                print("Algorithm converged with x-condition.")
            elif (res / np.sqrt(len(y_delta))) < tolf:
                print("Algorithm converged with f-condition.")
            print(f"Iterations: \t   {k}.")
            print(f"Relative Distance: {dist:0.5f}.")
            print(f"Residual: \t   {res / np.sqrt(len(y_delta)):0.5f}.")
        return x


class IterativeChambollePockTpV:
    def __init__(self, K, alpha=0.8):
        self.K = K
        self.alpha = alpha

        self.CP = ChambollePockTpV(K)

        # Get the shape of x and the shape of y from K
        self.mx, self.nx = self.K.m, self.K.n

    def __call__(
        self,
        y_delta,
        lmbda,
        H=10,
        x0=None,
        x_approx=None,
        p=1,
        maxit=200,
        tolf=5e-4,
        tolx=1e-5,
        verbose=False,
        return_obj=False,
    ):
        # Initialization
        x_H = np.zeros((H + 1, self.mx, self.nx))
        if x0 is not None:
            x_H[0] = x0

        if type(maxit) is int:
            maxit = np.repeat(maxit, H)

        outer_obj = np.zeros((H + 1,))  # Required to update lambda

        for h in range(H):
            # Compute next iterate
            x_H[h + 1], obj = self.CP(
                y_delta,
                lmbda,
                x0=x_H[h],
                x_approx=x_approx,
                p=p,
                maxit=maxit[h],
                tolf=tolf,
                tolx=tolx,
                verbose=verbose,
                return_obj=True,
            )
            outer_obj[h] = obj[-1]

            # Update parameters
            print(f"{h=} -> {lmbda=:0.4f}, {p=:0.4f}.")
            if h == 0:
                lmbda = lmbda / 2
            else:
                lmbda = lmbda * outer_obj[h] / outer_obj[h - 1]

            p = p * self.alpha

        if return_obj:
            return x_H, outer_obj
        return x_H


class ConstrainedChambollePockTpV:
    def __init__(self, A):
        self.A = A

        self.m, self.n = A.shape

        # Generate Gradient operators
        self.grad = operators.myGradient(
            1, (int(np.sqrt(self.n)), int(np.sqrt(self.n)))
        )

        self.m, self.n = A.shape

    def __call__(
        self,
        b,
        epsilon,
        lmbda,
        x_true=None,
        starting_point=None,
        eta=2e-3,
        maxiter=100,
        p=1,
    ):
        # Compute the approximation to || A ||_2
        nu = np.sqrt(
            self.power_method(self.A, num_iterations=10)
            / self.power_method(self.grad, num_iterations=10)
        )

        # Generate concatenate operator
        K = operators.ConcatenateOperator(self.A, self.grad)

        Gamma = np.sqrt(self.power_method(K, num_iterations=10))

        # Compute the parameters given Gamma
        tau = 1 / Gamma
        sigma = 1 / Gamma
        theta = 1

        # Iteration counter
        k = 0

        # Initialization
        if starting_point is None:
            x = np.zeros((self.n, 1))
        else:
            x = starting_point
        y = np.zeros((self.m, 1))
        w = np.zeros((2 * self.n, 1))

        xx = x

        # Initialize errors
        rel_err = np.zeros((maxiter + 1, 1))
        residues = np.zeros((maxiter + 1, 1))

        # Stopping conditions
        con = True
        while con and (k < maxiter):
            # Update y
            yy = y + sigma * np.expand_dims(self.A(xx) - b, -1)
            y = yy / (1 + lmbda * sigma)
            # y = max(np.linalg.norm(yy) - (sigma * epsilon), 0) * yy / np.linalg.norm(yy)

            # Compute the magnitude of the gradient
            grad_x = self.grad(xx)
            grad_mag = np.square(grad_x[: len(grad_x) // 2]) + np.square(
                grad_x[len(grad_x) // 2 :]
            )

            # Compute the reweighting factor
            W = np.expand_dims(np.power(np.sqrt(eta**2 + grad_mag) / eta, p - 1), -1)
            WW = np.concatenate((W, W), axis=0)

            # Update w
            x_grad = np.expand_dims(self.grad(xx), -1)
            ww = w + sigma * x_grad

            abs_ww = np.zeros((self.n, 1))
            abs_ww = np.square(ww[: self.n]) + np.square(ww[self.n :])
            abs_ww = np.concatenate((abs_ww, abs_ww), axis=0)

            lmbda_vec_over_nu = lmbda * WW / nu
            w = lmbda_vec_over_nu * ww / np.maximum(lmbda_vec_over_nu, abs_ww)

            # Save the value of x
            xtmp = x

            # Update x
            x = xtmp - tau * (
                np.expand_dims(self.A.T(y), -1)
                + nu * np.expand_dims(self.grad.T(w), -1)
            )

            # Project x to (x>0)
            x[x < 0] = 0

            # Compte signed x
            xx = x + theta * (x - xtmp)

            # Compute relative error
            if x_true is not None:
                rel_err[k] = np.linalg.norm(
                    xx.flatten() - x_true.flatten()
                ) / np.linalg.norm(x_true.flatten())

            # Compute the magnitude of the gradient of the actual iterate
            grad_x = self.grad(xx)
            grad_mag = np.expand_dims(
                np.sqrt(
                    np.square(grad_x[: len(grad_x) // 2])
                    + np.square(grad_x[len(grad_x) // 2 :])
                ),
                -1,
            )

            # Compute the value of TpV by reweighting
            ftpv = np.sum(np.abs(W * grad_mag))
            res = np.linalg.norm(self.A(xx) - b, 2) ** 2
            residues[k] = 0.5 * res + lmbda * ftpv

            # Stopping criteria
            c = np.sqrt(res) / (np.max(b) * np.sqrt(self.m))
            d_abs = np.linalg.norm(x.flatten() - xtmp.flatten(), 2)

            if (c >= 9e-6) and (c <= 1.1e-5):
                con = False

            if d_abs < 1e-4 * (1 + np.linalg.norm(xtmp.flatten(), 2)):
                con = False

            # Update k
            k = k + 1

        return x

    def power_method(self, A, num_iterations: int):
        b_k = np.random.rand(A.shape[1])

        for _ in range(num_iterations):
            # calculate the matrix-by-vector product Ab
            b_k1 = A.T(A(b_k))

            # calculate the norm
            b_k1_norm = np.linalg.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k1_norm

    def compute_obj_value(self, x, y, lmbda, p, eta):
        # Compute the value of the objective function TpV by reweighting
        grad_x = np.expand_dims(self.grad(x), -1)
        grad_mag = np.square(grad_x[: len(grad_x) // 2]) + np.square(
            grad_x[len(grad_x) // 2 :]
        )
        W = np.power(np.sqrt(eta**2 + grad_mag) / eta, p - 1)

        ftpv = np.sum(np.abs(W * np.sqrt(grad_mag)))
        return 0.5 * np.linalg.norm(self.A(x) - y, 2) ** 2 + lmbda * ftpv
