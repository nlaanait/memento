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

    @classmethod
    def build(cls):
        obj  = cls(*args)
        return obj

    def print_verbose(self, *args, **kwargs):
        if self.verbose and self.rank == 0:
            print(*args, **kwargs)

    def solve(self, initial_guess):
        ## TODO: consider passing initial guess to solve()

        __residual = self.rhs - self.matrix @ initial_guess
        __residual_norm = __residual.norm()
        __relative_residual_norm = __residual_norm / self.rhs_norm

        __solution = initial_guess

        while __relative_residual_norm > self.tol and self.iteration_count < self.maxiter:
            ## TODO: consider making all of these non-attributes and just return them
            __solution = __solution + self.relaxation * __residual
            
            __residual = self.rhs - torch.matmul(self.matrix, __solution)
            __residual_norm = __residual.norm()
            relative_residual_norm = __residual_norm / self.rhs_norm
            self.iteration_count += 1
            self.print_verbose("Richardson converged in ", str(self.iteration_count), " iteration with relative residual norm: ",
                                     str(__relative_residual_norm), end='...')

        # Do not return because it's already an attribute
        return __solution

class Optimizer:

    def __init__(self, iteration, tolerance, device="cpu"):
        self.iter = iteration
        self.tol = tolerance
        self.dev = torch.device(device)

    def solve(self):
        raise NotImplementedError

    @classmethod
    def build(cls):
        obj  = cls(*args)
        return obj

    @staticmethod
    def delete(something):
        del something


class SpecialOptimizer(Optimizer):

    def __init__(self, *args, **kwargs):
        iteration, tolerance = args[:]
        device = kwargs.get("device", "cpu")
        super(SpecialOptimizer, self).__init__(iteration, tolerance, device=device)
        ## do something with args and kwargs ...

    def solve(self):
        pass

class ConjugateGradient(object):

    #CAVEAT: functions f and g are used as external objects
    #CAVEAT: x, y

    def __init__(self, nsteps=10, residual_tol=1e-18, lr=1.0, verbose=0):

        self.nsteps = nsteps
        self.residual_tol = residual_tol
        self.lr = lr
        self.verbose = verbose
        self.iter_count = 0

    def print_verbose(self, *args, **kwargs):
        if self.verbose and self.rank == 0:
            print(*args, **kwargs)

    def solve(self, f, g, x, y):

        __f_history = []
        __g_history = []
        __x_history = []
        __y_history = []

        __f_history.append(f(x, y))
        __g_history.append(g(x, y))
        __x_history.append(x)
        __y_history.append(y)

        while self.iter_count < self.nsteps:

            __f_eval = f(x, y)
            __g_eval = g(x, y)
            __grad_f_x = autograd.grad(_f_eval, x, create_graph=True, allow_unused=True)
            __grad_g_y = autograd.grad(_g_eval, y, create_graph=True, allow_unused=True)

            __new_x = x - self.lr * __grad_f_x[0]
            __new_y = y - self.lr * __grad_g_y[0]
            x = __new_x.clone().detach().requires_grad_(True)
            y = __new_y.clone().detach().requires_grad_(True)

            print_verbose("######################################################")
            print_verbose("Iteration: ", self.iter_count)
            print_verbose("x: ", x)
            print_verbose("y: ", y)
            print_verbose("f(x,y): ", f(x, y))
            print_verbose("g(x,y): ", g(x, y))
            print_verbose("######################################################")

            __f_history.append(f(x, y))
            __g_history.append(g(x, y))
            __x_history.append(x)
            __y_history.append(y)

            self.iter_count += 1

        return __f_history, __g_history, __x_history, __y_history


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

        __f_history = []
        __g_history = []
        __x_history = []
        __y_history = []

        __x_history.append(x)
        __y_history.append(y)

        while self.iter_count < self.nsteps:
            self.iter_count += 1
            __f_val = f(x, y)
            __g_val = g(x, y)
            __grad_f_x, = autograd.grad(__f_val, x, create_graph=True, allow_unused=True)
            __grad_g_y, = autograd.grad(__g_val, y, create_graph=True, allow_unused=True)
            if not self.full_hessian:
                __hess_f_xy = hessian_vec(__grad_f_x, y, retain_graph=False)
                __hess_g_yx = hessian_vec(__grad_g_y, x, retain_graph=False)
                __x_rhs = __grad_f_x - self.lr * torch.matmul(__hess_f_xy,  __grad_g_y)
                __y_rhs = __grad_g_y - self.lr * torch.matmul(__hess_g_yx, __grad_f_x)
                __x_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.diag_embed(torch.matmul(__hess_f_xy, __hess_g_yx))
                #__x_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.diag_embed(__hess_f_xy * __hess_g_yx)
                __y_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.diag_embed(torch.matmul(__hess_g_yx, __hess_f_xy))
                #__y_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.diag_embed(__hess_g_yx * __hess_f_xy)
            else:
                __hess_f_xy = hessian(__grad_f_x, y, retain_graph=False)
                __hess_g_yx = hessian(__grad_g_y, x, retain_graph=False)
                __x_rhs = __grad_f_x - self.lr * torch.matmul(__hess_f_xy, __grad_g_y)
                __y_rhs = __grad_g_y - self.lr * torch.matmul(__hess_g_yx, __grad_f_x)
                __x_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.matmul(__hess_f_xy, __hess_g_yx)
                __y_A = torch.eye(x.shape[0]) - self.lr * self.lr * torch.matmul(__hess_g_yx, __hess_f_xy)

            __solver1 = Richardson(__x_A, __x_rhs, 1e-10, 1000, 1, verbose=False)
            __initial_guess_x = torch.randn(__x_rhs.shape)
            __delta_x = __solver1.solve(__initial_guess_x)
            __solver2 = Richardson(__y_A, __y_rhs, 1e-10, 1000, 1, verbose=False)
            __initial_guess_y = torch.randn(__y_rhs.shape)
            __delta_y = __solver2.solve(__initial_guess_y)
            __new_x = x - self.lr * __delta_x
            __new_y = y - self.lr * __delta_y
            x = __new_x.clone().detach().requires_grad_(True)
            y = __new_y.clone().detach().requires_grad_(True)

            self.print_verbose("######################################################")
            self.print_verbose("Iteration: ", self.iter_count)
            self.print_verbose("x: ", x)
            self.print_verbose("y: ", y)
            self.print_verbose("f(x,y): ", f(x, y))
            self.print_verbose("g(x,y): ", g(x, y))
            self.print_verbose("hess_f_xy:", __hess_f_xy)
            self.print_verbose("hess_g_yx:", __hess_g_yx)
            self.print_verbose("######################################################")

            __f_history.append(f(x, y))
            __g_history.append(g(x, y))
            __x_history.append(x)
            __y_history.append(y)

        return __f_history, __g_history, __x_history, __y_history


class CompetitiveGradientJacobi(object):

    def __init__(self, nsteps=10, residual_tol=1e-10, lr=1e-3, verbose=True, full_hessian=False):
        self.nsteps = nsteps
        self.residual_tol = residual_tol
        self.lr = lr
        self.verbose = verbose
        self.full_hessian = full_hessian
        self.iter_count = 0

    def print_verbose(self, *args, **kwargs):
        if self.verbose and self.rank == 0:
            print(*args, **kwargs)

    def solve(self, f, g, x, y, delay=1):

        __x_buffer = []
        __y_buffer = []
        __f_history = []
        __g_history = []
        __x_history = []
        __y_history = []

        __prev_y = y.clone().detach().requires_grad_(True)
        __prev_x = x.clone().detach().requires_grad_(True)
        __x_history.append(x0)
        __y_history.append(y0)

        while self.iter_count < self.nsteps:

            self.iter_count += 1
            __f_val_x = f(x, __prev_y)
            __f_val_y = f(__prev_x, y)
            __g_val_x = g(x, __prev_y)
            __g_val_y = g(__prev_x, y)
            __grad_f_x_x, = autograd.grad(__f_val_x, x, create_graph=True,
                                        allow_unused=True)  # terrible variable name, implies diagonal hessian!!
            __grad_f_x_y, = autograd.grad(__f_val_y, __prev_x, create_graph=True,
                                        allow_unused=True)  # terrible variable name, implies diagonal hessian!!
            __grad_g_y_x, = autograd.grad(__g_val_x, __prev_y, create_graph=True, allow_unused=True)
            __grad_g_y_y, = autograd.grad(__g_val_y, y, create_graph=True, allow_unused=True)

            if not self.full_hessian:

                __hess_f_xy_x = hessian_vec(__grad_f_x_x, __prev_y, retain_graph=False)
                __hess_f_xy_y = hessian_vec(__grad_f_x_y, y, retain_graph=False)
                __hess_g_yx_x = hessian_vec(__grad_g_y_x, x, retain_graph=False)
                __hess_g_yx_y = hessian_vec(__grad_g_y_y, prev_x, retain_graph=False)
                __delta_x = -lr * (__grad_f_x_x + 2 * __hess_f_xy_x * __grad_g_y_x)
                __delta_y = -lr * (__grad_g_y_y + 2 * __hess_g_yx_y * __grad_f_x_y)
            else:

                __hess_f_xy_x = hessian(__grad_f_x_x, __prev_y, retain_graph=False)
                __hess_f_xy_y = hessian(__grad_f_x_y, y, retain_graph=False)
                __hess_g_yx_x = hessian(__grad_g_y_x, x, retain_graph=False)
                __hess_g_yx_y = hessian(__grad_g_y_y, __prev_x, retain_graph=False)
                __delta_x = -self.lr * (__grad_f_x_x + 2 * torch.matmul(__hess_f_xy_x, __grad_g_y_x))
                __delta_y = -self.lr * (__grad_g_y_y + 2 * torch.matmul(__hess_g_yx_y, __grad_f_x_y))

            __new_x = x - self.lr * __delta_x
            __new_y = y - self.lr * __delta_y

            x = __new_x.clone().detach().requires_grad_(True)
            y = __new_y.clone().detach().requires_grad_(True)

            __x_buffer.append(x)
            __y_buffer.append(y)

            self.print_verbose("######################################################")
            self.print_verbose("Iteration: ", self.iter_count)
            self.print_verbose("x: ", x)
            self.print_verbose("y: ", y)
            self.print_verbose("f(x,y): ", f(x, y))
            self.print_verbose("g(x,y): ", g(x, y))
            self.print_verbose("hess_f_xy_x:", __hess_f_xy_x)
            self.print_verbose("hess_f_xy_y:", __hess_f_xy_y)
            self.print_verbose("hess_g_yx_x:", __hess_g_yx_x)
            self.print_verbose("hess_g_yx_y:", __hess_g_yx_y)
            self.print_verbose("######################################################")

            __f_history.append(f(x, y))
            __g_history.append(g(x, y))
            __x_history.append(x)
            __y_history.append(y)

            if self.iter_count > delay:
                __prev_y = __y_buffer[self.iter_count - delay].clone().detach().requires_grad_(True)
                __prev_x = __x_buffer[self.iter_count - delay].clone().detach().requires_grad_(True)

        return __f_history, __g_history, __x_history, __y_history



