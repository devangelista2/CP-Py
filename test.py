import matplotlib.pyplot as plt
import numpy as np

import operators
import solvers

if __name__ == "__main__":
    # Load data
    x_true = np.load("gt.npy")[0, 0]
    mx, nx = x_true.shape

    # Config operator
    det_size = int(mx * np.sqrt(2))
    n_angles = 180
    angles = np.linspace(np.deg2rad(0), np.deg2rad(180), n_angles, endpoint=False)

    # Define the operators
    K = operators.CTProjector(mx, nx, angles, det_size=det_size, geometry="fanflat")

    # Compute sinogram
    y = K(x_true)
    e = np.random.normal(0, 1, y.shape)
    e /= np.linalg.norm(e.flatten(), 2)
    y_delta = y + e * 0.01 * np.linalg.norm(y.flatten(), 2)
    my, ny = n_angles, det_size

    # Choose tests
    cp_tpv = False
    ircp_tpv = False
    constrained_cp_tpv = False
    icp_tpv = True

    if cp_tpv:
        # Solve
        CP_TpV = solvers.ChambollePockTpV(K)
        x_sol, obj_f = CP_TpV(
            y_delta, lmbda=1, p=1, maxit=500, verbose=False, return_obj=True
        )

        print(
            np.linalg.norm(x_sol.flatten() - x_true.flatten(), 2)
            / np.linalg.norm(x_true.flatten())
        )

        plt.imshow(x_sol, cmap="gray")
        plt.show()

        plt.semilogy(obj_f)
        plt.show()

    if ircp_tpv:
        # Solve
        CP_TpV = solvers.IRChambollePockTpV(K)
        x_sol = CP_TpV(
            y_delta,
            lmbda=1,
            p=1,
            maxit=500,
            verbose=False,
        )

        print(
            np.linalg.norm(x_sol.flatten() - x_true.flatten(), 2)
            / np.linalg.norm(x_true.flatten())
        )

        plt.imshow(x_sol, cmap="gray")
        plt.show()

    if constrained_cp_tpv:
        # Solve
        constrained_CP_TpV = solvers.ConstrainedChambollePockTpV(K)
        x_sol = constrained_CP_TpV(
            y_delta,
            lmbda=1,
            epsilon=np.linalg.norm(e.flatten(), 2),
            p=1,
            maxit=500,
            verbose=False,
        )

        print(
            np.linalg.norm(x_sol.flatten() - x_true.flatten(), 2)
            / np.linalg.norm(x_true.flatten())
        )

        plt.imshow(x_sol.reshape((256, 256)), cmap="gray")
        plt.show()

    if icp_tpv:
        # Solve
        iCP_TpV = solvers.IteratedChambollePockTpV(K, alpha=0.8)
        x_sol, obj = iCP_TpV(
            y_delta,
            x_true=x_true,
            lmbda=1,
            H=10,
            p=1,
            maxit=50,
            return_obj=True,
            verbose=True,
            tolx=1e-8,
        )
