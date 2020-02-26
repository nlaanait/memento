from optimizers import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    standard_gd = True
    standard_gd_plot = True
    explicit = True
    explicit_plot = True
    jacobi = True
    jacobi_plot = True
    n_dims = 1
    nsteps = 500
    lr = 0.1

    x0 = torch.randn((n_dims,), requires_grad=True)
    y0 = torch.randn((n_dims,), requires_grad=True)

    def f(x,y):
        # return x @ y #+  0.5 * x**2 - 1/3*x**3
        # return x @ y #+ 1e-6 * x**2 @ y**2  + 1e-8 * x **3 @ y **3
        return x @ y #- torch.exp(x**2) + torch.exp(y**2)

    def g(x,y):
        # return -x @ y #- 0.5 * y**2 + 1/3*y**3
        return -f(x,y)


    if standard_gd:

        x_gd = x0
        y_gd = y0

        conjugate_gd_solver = ConjugateGradient(nsteps=nsteps, residual_tol=1e-18, lr=lr)
        [f_history_gd, g_history_gd, x_history_gd, y_history_gd] = conjugate_gd_solver.solve(f, g, x_gd, y_gd)

    if explicit:

        x_cgd = x0
        y_cgd = y0

        conjugate_gd_solver = CompetitiveGradient(nsteps=nsteps, residual_tol=1e-18, lr=lr)
        [f_history_cgd, g_history_cgd, x_history_cgd, y_history_cgd] = conjugate_gd_solver.solve(f, g, x_cgd, y_cgd)

    if jacobi:

        x_cgd_jacobi = x0
        y_cgd_jacobi = y0

        competitive_cgd_jacobi_solver = CompetitiveGradientJacobi(nsteps=nsteps, residual_tol=1e-18, lr=lr)
        [f_history_cgd_jacobi, g_history_cgd_jacobi, x_history_cgd_jacobi, y_history_cgd_jacobi] = competitive_cgd_jacobi_solver.solve(f, g, x_cgd_jacobi, y_cgd_jacobi)

    if standard_gd and standard_gd_plot:
        plt.figure()
        plt.plot([x for x in range(0, len(f_history_gd))], f_history_gd)
        plt.xlabel('Number of iterations')
        plt.ylabel('Objective function value')
        plt.title('Gradient Descent')
        plt.savefig('gd.png', dpi=300)

    if explicit and explicit_plot:

        plt.figure()
        plt.plot([x for x in range(0, len(f_history_cgd))], f_history_cgd)
        plt.xlabel('Number of iterations')
        plt.ylabel('Objective function value')
        plt.title('Competitive Gradient Descent')
        plt.savefig('cgd.png', dpi=300)

    if jacobi and jacobi_plot:

        plt.figure()
        plt.plot([x for x in range(0, len(f_history_cgd_jacobi))], f_history_cgd_jacobi, label='jacobi')
        plt.xlabel('Number of iterations')
        plt.ylabel('Objective function value')
        plt.title('Competitive Gradient Descent Jacobi')
        plt.legend()
        plt.savefig('cgd_explicit_jac.png', dpi=300)


    # Plot of the trajectories
    if n_dims == 1:
        plt.figure()
        if standard_gd_plot:
            gd_line = plt.plot(x_history_gd, y_history_gd, label='Standard GD', marker='o')
        if explicit_plot:
            cgd_line = plt.plot(x_history_cgd, y_history_cgd, label='CGD explicit')
        if jacobi_plot:
            cgd_line_jacobi = plt.plot(x_history_cgd_jacobi, y_history_cgd_jacobi, label='CGD Jacobi')
        plt.xlabel('State of agent x')
        plt.ylabel('State of agent y')
        plt.title('Trajectory of states attained by agents')
        plt.legend()
        plt.savefig('traj.png', dpi=300)

        plt.figure()
        plt.plot(range(nsteps + 1), x_history_gd, label='Standard_GD_x')
        plt.plot(range(nsteps + 1), y_history_gd, label='Standard_GD_y')
        plt.plot(range(nsteps + 1), x_history_cgd, label='Competitive_GD_x')
        plt.plot(range(nsteps + 1), y_history_cgd, label='Competitive_GD_y')
        plt.plot(range(nsteps + 1), x_history_cgd_jacobi, label='Competitive_GD_Jacobi_x')
        plt.plot(range(nsteps + 1), y_history_cgd_jacobi, label='Competitive_GD_Jacobi_y')
        plt.xlabel('Iterations')
        plt.ylabel('States')
        plt.legend()
        plt.savefig('dynamics.png', dpi=300)



