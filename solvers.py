import numpy as np

import operators
import utils


class ChambollePockTpV:
    def __init__(self, K):
        self.K = K

        # Get the shape of x and the shape of y from K
        self.mx, self.nx = self.K.m, self.K.n

        # Initialization
        self.gradient = operators.GradientOperator((self.mx, self.nx))

        # Compute the 2-norm of M = [grad; K]
        self.L = np.sqrt(utils.power_method(self.K, self.gradient, maxit=10))

        # Compute nu
        self.nu = np.sqrt(
            utils.power_method(self.K, maxit=10)
            / utils.power_method(self.gradient, maxit=10)
        )

        self.sigma = 1.0 / self.L
        self.tau = 1.0 / self.L
        self.theta = 1.0
        self.eta = 2e-3

    def __call__(
        self,
        y_delta,
        lmbda,
        x0=None,
        x_true=None,
        x_approx=None,
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
            x = np.zeros((self.mx, self.nx))
            x0 = x
        else:
            x = x0.reshape((self.mx, self.nx))

        if x_approx is None:
            x_approx = x0

        s = np.zeros((2, self.mx, self.nx))
        q = np.zeros_like(y_delta)
        x_tilde = x

        # Compute the reweighting factor
        self.W = np.expand_dims(
            np.power(
                np.sqrt(
                    self.eta**2 + utils.gradient_magnitude(self.gradient(x_approx))
                )
                / self.eta,
                p - 1,
            ),
            0,
        )
        self.W = np.repeat(self.W, 2, axis=0)

        # Compute the first value of the objective function
        if return_obj:
            obj[0] = self.obj_function(x0, y_delta, lmbda)

        k = 0
        stopping = False
        while not stopping:
            # Update dual variables
            q = (q + self.sigma * (self.K(x_tilde) - y_delta)) / (
                1 + lmbda * self.sigma
            )
            s = utils.prox_l1_reweighted(
                s + self.sigma * self.gradient(x_tilde), self.W, lmbda / self.nu
            )

            # Update primal variables
            x_old = x
            x = x - self.tau * (self.nu * self.gradient.T(s) + self.K.T(q))

            # Project x to (x>0)
            x[x < 0] = 0

            # Update x_tilde
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
            dist = np.linalg.norm(x_old.flatten() - x.flatten()) / (
                np.linalg.norm(x.flatten()) + 1e-6
            )

            # Update stopping condition
            stopping = (
                (dist < tolx) or ((res / np.sqrt(len(y_delta))) < tolf) or (k >= maxit)
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

        return 0.5 * res**2 + lmbda * tpv


class IRChambollePockTpV:
    def __init__(self, K):
        self.K = K

        # Get the shape of x and the shape of y from K
        self.mx, self.nx = self.K.m, self.K.n

        # Initialization
        self.gradient = operators.GradientOperator((self.mx, self.nx))

        # Compute the 2-norm of M = [grad; K]
        self.L = np.sqrt(utils.power_method(self.K, self.gradient, maxit=10))

        # Compute nu
        self.nu = np.sqrt(
            utils.power_method(self.K, maxit=10)
            / utils.power_method(self.gradient, maxit=10)
        )

        self.sigma = 1.0 / self.L
        self.tau = 1.0 / self.L
        self.theta = 1.0
        self.eta = 2e-3

    def __call__(
        self,
        y_delta,
        lmbda,
        x0=None,
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
            x = np.zeros((self.mx, self.nx))
        else:
            x = x0.reshape((self.mx, self.nx))
        s = np.zeros((2, self.mx, self.nx))
        q = np.zeros_like(y_delta)
        x_tilde = x

        # Compute the first value of the objective function
        if return_obj:
            obj[0] = self.obj_function(x0, y_delta, lmbda)

        k = 0
        stopping = False
        while not stopping:
            # Compute the gradient magnitude of x_approx, for the reweighting
            grad = self.gradient(x_tilde)
            grad_mag = utils.gradient_magnitude(grad)

            # Compute the reweighting factor
            self.W = np.expand_dims(
                np.power(np.sqrt(self.eta**2 + grad_mag) / self.eta, p - 1), 0
            )
            self.W = np.repeat(self.W, 2, axis=0)

            # Update dual variables
            q = (q + self.sigma * (self.K(x_tilde) - y_delta)) / (
                1 + lmbda * self.sigma
            )
            s = utils.prox_l1_reweighted(
                s + self.sigma * self.gradient(x_tilde), self.W, lmbda / self.nu
            )

            # Update primal variables
            x_old = x
            x = x - self.tau * (self.nu * self.gradient.T(s) + self.K.T(q))

            # Project x to (x>0)
            x[x < 0] = 0

            # Update x_tilde
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
            dist = np.linalg.norm(x_old.flatten() - x.flatten()) / (
                np.linalg.norm(x.flatten()) + 1e-6
            )

            # Update stopping condition
            stopping = (
                (dist < tolx) or ((res / np.sqrt(len(y_delta))) < tolf) or (k >= maxit)
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

        return 0.5 * res**2 + lmbda * tpv


class ConstrainedChambollePockTpV:
    def __init__(self, K):
        self.K = K

        # Get the shape of x and the shape of y from K
        self.mx, self.nx = self.K.m, self.K.n

        # Initialization
        self.gradient = operators.GradientOperator((self.mx, self.nx))

        # Compute the 2-norm of M = [grad; K]
        self.L = np.sqrt(utils.power_method(self.K, self.gradient, maxit=10))

        # Compute nu
        self.nu = np.sqrt(
            utils.power_method(self.K, maxit=10)
            / utils.power_method(self.gradient, maxit=10)
        )

        self.sigma = 1.0 / self.L
        self.tau = 1.0 / self.L
        self.theta = 1.0
        self.eta = 2e-3

    def __call__(
        self,
        y_delta,
        lmbda,
        epsilon,
        x0=None,
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
            delta_epsilon(||K*x - d||_2^2) + Lambda*TpV(x)
        by reweighting

        K : projection operator
        Lambda : weight of the TV penalization (the higher Lambda, the more sparse is the solution)
        epsilon : the radius of the ball in which the residual must be
        L : norm of the operator [P, Lambda*grad] (see power_method)
        maxit : number of iterations
        """
        # Initialization
        if return_obj:
            obj = np.zeros((maxit + 1,))

        if x0 is None:
            x = np.zeros((self.mx, self.nx))
        else:
            x = x0.reshape((self.mx, self.nx))
        s = np.zeros((2, self.mx, self.nx))
        q = np.zeros_like(y_delta)
        x_tilde = x

        # Compute the first value of the objective function
        if return_obj:
            obj[0] = self.obj_function(x0, y_delta, lmbda)

        k = 0
        stopping = False
        while not stopping:
            # Compute the gradient magnitude of x_approx, for the reweighting
            grad = self.gradient(x_tilde)
            grad_mag = utils.gradient_magnitude(grad)

            # Compute the reweighting factor
            self.W = np.expand_dims(
                np.power(np.sqrt(self.eta**2 + grad_mag) / self.eta, p - 1), 0
            )
            self.W = np.repeat(self.W, 2, axis=0)

            # Update dual variables
            q_tilde = q + self.sigma * (self.K(x_tilde) - y_delta)
            q = (
                max(np.linalg.norm(q_tilde.flatten()) - (self.sigma * epsilon), 0)
                * q_tilde
                / np.linalg.norm(q_tilde.flatten())
            )
            s = utils.prox_l1_reweighted(
                s + self.sigma * self.gradient(x_tilde), self.W, lmbda / self.nu
            )

            # Update primal variables
            x_old = x
            x = x - self.tau * (self.nu * self.gradient.T(s) + self.K.T(q))

            # Project x to (x>0)
            x[x < 0] = 0

            # Update x_tilde
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
            dist = np.linalg.norm(x_old.flatten() - x.flatten()) / (
                np.linalg.norm(x.flatten()) + 1e-6
            )

            # Update stopping condition
            stopping = (
                (dist < tolx) or ((res / np.sqrt(len(y_delta))) < tolf) or (k >= maxit)
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

        return 0.5 * res**2 + lmbda * tpv


class IteratedChambollePockTpV:
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
        x_true=None,
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
            if verbose and x_true is not None:
                rel_err = np.linalg.norm(
                    x_true.flatten() - x_H[h + 1].flatten(), 2
                ) / np.linalg.norm(x_true.flatten())
                print(f"{h=} -> Rel. Err.: {rel_err:0.4f}.")
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
