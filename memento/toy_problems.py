from optimizers import *

if __name__ == "__main__":
    explicit = True
    explicit_plot = True
    jacobi = False
    jacobi_plot = False
    n_dims = 1
    nsteps = 500
    lr = 0.01

    x = torch.randn((n_dims,), requires_grad=True)
    y = torch.randn((n_dims,), requires_grad=True)

    def f(x,y):
        # return x @ y #+  0.5 * x**2 - 1/3*x**3
        # return x @ y #+ 1e-6 * x**2 @ y**2  + 1e-8 * x **3 @ y **3
        return x @ y #- torch.exp(x**2) + torch.exp(y**2)

    def g(x,y):
        # return -x @ y #- 0.5 * y**2 + 1/3*y**3
        return -f(x,y)


    competitive_cgd_solver = CompetitiveGradient(nsteps=500, residual_tol=1e-18, lr=0.1)
    competitive_cgd_solver.solve(f, g, x, y)