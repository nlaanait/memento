# This is a preliminary version of the code
from typing import Any
import time
import torch
import numpy
from torch import Tensor
from torch import autograd
from torch.autograd import Variable
from torch.autograd import grad


def hessian_vec(grad_vec, var, retain_graph=False):
    v = torch.ones_like(var)
    vec, = autograd.grad(grad_vec, var, grad_outputs=v, allow_unused=True, retain_graph=retain_graph)
    return vec

def hessian(grad_vec, var, retain_graph=False):
    v = torch.eye(var.shape[0])
    matrix = torch.cat([autograd.grad(grad_vec, var, grad_outputs=v_row, allow_unused=True, retain_graph=retain_graph)[0]
                            for v_row in v])
    matrix = matrix.view(-1,var.shape[0])
    return matrix



class Richardson(object):

    def __init__(self, matrix, rhs, tol, maxiter, relaxation, verbose=False):

        """
        :param matrix: coefficient matrix
        :param rhs: right hand side
        :param tol: tolerance for stopping criterion based on the relative residual
        :param maxiter: maximum number of iterations
        :param relaxation: relaxation parameter for Richardson
        :param initial_guess: initial guess
        :return: matrix ** -1 * rhs
        """

        self.rhs = rhs
        self.matrix = matrix
        self.tol = tol
        self.maxiter = maxiter
        self.relaxation = relaxation
        self.rhs_norm = torch.norm(rhs, 2)
        self.iteration_count = 0
        self.verbose = verbose

    def print_verbose(self, *args, **kwargs):
        if self.verbose :
            print(*args, **kwargs)

    def solve(self, initial_guess):
        ## TODO: consider passing initial guess to solve()

        residual = self.rhs - self.matrix @ initial_guess
        residual_norm = residual.norm()
        relative_residual_norm = residual_norm / self.rhs_norm

        solution = initial_guess

        while relative_residual_norm > self.tol and self.iteration_count < self.maxiter:
            ## TODO: consider making all of these non-attributes and just return them
            solution = solution + self.relaxation * residual
            
            residual = self.rhs - torch.matmul(self.matrix, solution)
            residual_norm = residual.norm()
            relative_residual_norm = residual_norm / self.rhs_norm
            self.iteration_count += 1
            self.print_verbose("Richardson converged in ", str(self.iteration_count), " iteration with relative residual norm: ",
                                     str(relative_residual_norm), end='...')

        # Do not return because it's already an attribute
        return solution

"""
class Optimizer:

    def __init__(self, iteration, tolerance, device="cpu"):
        self.iter = iteration
        self.tol = tolerance
        self.dev = torch.device(device)

    def solve(self):
        raise NotImplementedError


class SpecialOptimizer(Optimizer):

    def __init__(self, *args, **kwargs):
        iteration, tolerance = args[:]
        device = kwargs.get("device", "cpu")
        super(SpecialOptimizer, self).__init__(iteration, tolerance, device=device)
        ## do something with args and kwargs ...

    def solve(self):
        pass
"""

class ConjugateGradient(object):

    def __init__(self, nsteps=10, residual_tol=1e-18, lr=1.0, verbose=True):

        self.nsteps = nsteps
        self.residual_tol = residual_tol
        self.lr = lr
        self.verbose = verbose
        self.iter_count = 0

    def print_verbose(self, *args, **kwargs):
        if self.verbose :
            print(*args, **kwargs)

    def solve(self, f, g, x, y):

        f_history = []
        g_history = []
        x_history = []
        y_history = []

        f_history.append(f(x, y))
        g_history.append(g(x, y))
        x_history.append(x)
        y_history.append(y)

        while self.iter_count < self.nsteps:
            self.iter_count += 1

            f_eval = f(x, y)
            g_eval = g(x, y)
            grad_f_x = autograd.grad(f_eval, x, create_graph=True, allow_unused=True)
            grad_g_y = autograd.grad(g_eval, y, create_graph=True, allow_unused=True)

            new_x = x - self.lr * grad_f_x[0]
            new_y = y - self.lr * grad_g_y[0]
            x = new_x.clone().detach().requires_grad_(True)
            y = new_y.clone().detach().requires_grad_(True)

            self.print_verbose("######################################################")
            self.print_verbose("Iteration: ", self.iter_count)
            self.print_verbose("x: ", x)
            self.print_verbose("y: ", y)
            self.print_verbose("f(x,y): ", f(x, y))
            self.print_verbose("g(x,y): ", g(x, y))
            self.print_verbose("######################################################")

            f_history.append(f(x, y))
            g_history.append(g(x, y))
            x_history.append(x)
            y_history.append(y)

        return f_history, g_history, x_history, y_history


class CompetitiveGradient(object):

    def __init__(self, nsteps=10, residual_tol=1e-10, lr=1e-3, verbose=True, full_hessian=False):
        self.nsteps = nsteps
        self.residual_tol = residual_tol
        self.lr = lr
        self.verbose = verbose
        self.full_hessian = full_hessian
        self.iter_count = 0

    def print_verbose(self, *args, **kwargs):
        if self.verbose :
            print(*args, **kwargs)

    def solve(self, f, g, x, y):

        f_history = []
        g_history = []
        x_history = []
        y_history = []

        x_history.append(x)
        y_history.append(y)

        while self.iter_count < self.nsteps:
            self.iter_count += 1
            f_val = f(x, y)
            g_val = g(x, y)
            grad_f_x, = autograd.grad(f_val, x, create_graph=True, allow_unused=True)
            grad_g_y, = autograd.grad(g_val, y, create_graph=True, allow_unused=True)
            if not self.full_hessian:
                hess_f_xy = hessian_vec(grad_f_x, y, retain_graph=False)
                hess_g_yx = hessian_vec(grad_g_y, x, retain_graph=False)
                x_rhs = grad_f_x - self.lr * torch.matmul(hess_f_xy,  grad_g_y)
                y_rhs = grad_g_y - self.lr * torch.matmul(hess_g_yx, grad_f_x)
                # The "*" multiplication operates elementwise
                # We have to use the "*" and not the matmul method because we do NOT extract the entire Hessian matrix, we just
                # extract the diagonal entries
                #__x_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.diag_embed(torch.matmul(__hess_f_xy, __hess_g_yx))
                x_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.diag_embed(hess_f_xy * hess_g_yx)
                #__y_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.diag_embed(torch.matmul(__hess_g_yx, __hess_f_xy))
                y_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.diag_embed(hess_g_yx * hess_f_xy)
            else:
                hess_f_xy = hessian(grad_f_x, y, retain_graph=False)
                hess_g_yx = hessian(grad_g_y, x, retain_graph=False)
                x_rhs = grad_f_x - self.lr * torch.matmul(hess_f_xy, grad_g_y)
                y_rhs = grad_g_y - self.lr * torch.matmul(hess_g_yx, grad_f_x)
                x_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.matmul(hess_f_xy, hess_g_yx)
                y_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.matmul(hess_g_yx, hess_f_xy)

            solver1 = Richardson(x_A, x_rhs, 1e-10, 1000, 1, verbose=False)
            initial_guess_x = torch.randn(x_rhs.shape)
            delta_x = solver1.solve(initial_guess_x)
            solver2 = Richardson(y_A, y_rhs, 1e-10, 1000, 1, verbose=False)
            initial_guess_y = torch.randn(y_rhs.shape)
            delta_y = solver2.solve(initial_guess_y)
            new_x = x - self.lr * delta_x
            new_y = y - self.lr * delta_y
            x = new_x.clone().detach().requires_grad_(True)
            y = new_y.clone().detach().requires_grad_(True)

            self.print_verbose("######################################################")
            self.print_verbose("Iteration: ", self.iter_count)
            self.print_verbose("x: ", x)
            self.print_verbose("y: ", y)
            self.print_verbose("f(x,y): ", f(x, y))
            self.print_verbose("g(x,y): ", g(x, y))
            self.print_verbose("hess_f_xy:", hess_f_xy)
            self.print_verbose("hess_g_yx:", hess_g_yx)
            self.print_verbose("######################################################")

            f_history.append(f(x, y))
            g_history.append(g(x, y))
            x_history.append(x)
            y_history.append(y)

        return f_history, g_history, x_history, y_history


class CompetitiveGradientJacobi(object):

    def __init__(self, nsteps=10, residual_tol=1e-10, lr=1e-3, verbose=True, full_hessian=False):
        self.nsteps = nsteps
        self.residual_tol = residual_tol
        self.lr = lr
        self.verbose = verbose
        self.full_hessian = full_hessian
        self.iter_count = 0

    def print_verbose(self, *args, **kwargs):
        if self.verbose :
            print(*args, **kwargs)

    def solve(self, f, g, x, y, delay=1):

        x_buffer = []
        y_buffer = []
        f_history = []
        g_history = []
        x_history = []
        y_history = []

        prev_y = y.clone().detach().requires_grad_(True)
        prev_x = x.clone().detach().requires_grad_(True)
        x_history.append(x)
        y_history.append(y)

        while self.iter_count < self.nsteps:

            self.iter_count += 1
            f_val_x = f(x, prev_y)
            f_val_y = f(prev_x, y)
            g_val_x = g(x, prev_y)
            g_val_y = g(prev_x, y)
            grad_f_x_x, = autograd.grad(f_val_x, x, create_graph=True,
                                        allow_unused=True)  # terrible variable name, implies diagonal hessian!!
            grad_f_x_y, = autograd.grad(f_val_y, prev_x, create_graph=True,
                                        allow_unused=True)  # terrible variable name, implies diagonal hessian!!
            grad_g_y_x, = autograd.grad(g_val_x, prev_y, create_graph=True, allow_unused=True)
            grad_g_y_y, = autograd.grad(g_val_y, y, create_graph=True, allow_unused=True)

            if not self.full_hessian:

                hess_f_xy_x = hessian_vec(grad_f_x_x, prev_y, retain_graph=False)
                hess_f_xy_y = hessian_vec(grad_f_x_y, y, retain_graph=False)
                hess_g_yx_x = hessian_vec(grad_g_y_x, x, retain_graph=False)
                hess_g_yx_y = hessian_vec(grad_g_y_y, prev_x, retain_graph=False)
                delta_x = -self.lr * (grad_f_x_x + 2 * hess_f_xy_x * grad_g_y_x)
                delta_y = -self.lr * (grad_g_y_y + 2 * hess_g_yx_y * grad_f_x_y)
            else:

                hess_f_xy_x = hessian(grad_f_x_x, prev_y, retain_graph=False)
                hess_f_xy_y = hessian(grad_f_x_y, y, retain_graph=False)
                hess_g_yx_x = hessian(grad_g_y_x, x, retain_graph=False)
                hess_g_yx_y = hessian(grad_g_y_y, prev_x, retain_graph=False)
                delta_x = -self.lr * (grad_f_x_x + 2 * torch.matmul(hess_f_xy_x, grad_g_y_x))
                delta_y = -self.lr * (grad_g_y_y + 2 * torch.matmul(hess_g_yx_y, grad_f_x_y))

            new_x = x - self.lr * delta_x
            new_y = y - self.lr * delta_y

            x = new_x.clone().detach().requires_grad_(True)
            y = new_y.clone().detach().requires_grad_(True)

            x_buffer.append(x)
            y_buffer.append(y)

            self.print_verbose("######################################################")
            self.print_verbose("Iteration: ", self.iter_count)
            self.print_verbose("x: ", x)
            self.print_verbose("y: ", y)
            self.print_verbose("f(x,y): ", f(x, y))
            self.print_verbose("g(x,y): ", g(x, y))
            self.print_verbose("hess_f_xy_x:", hess_f_xy_x)
            self.print_verbose("hess_f_xy_y:", hess_f_xy_y)
            self.print_verbose("hess_g_yx_x:", hess_g_yx_x)
            self.print_verbose("hess_g_yx_y:", hess_g_yx_y)
            self.print_verbose("######################################################")

            f_history.append(f(x, y))
            g_history.append(g(x, y))
            x_history.append(x)
            y_history.append(y)

            if self.iter_count > delay:
                prev_y = y_buffer[self.iter_count - delay].clone().detach().requires_grad_(True)
                prev_x = x_buffer[self.iter_count - delay].clone().detach().requires_grad_(True)

        return f_history, g_history, x_history, y_history



