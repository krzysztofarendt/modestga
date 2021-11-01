"""Optimization result containers."""


class OptRes:
    """
    Optimization result.

    Instance attributes:
    - x - numpy 1D array, optimized parameters
    - message - str, exit message
    - nfev - int, number of function evaluations
    - ng - int, number of generations
    - fx - float, final function value
    - constr - list of floats, constraint function values (if exist)
    """

    def __init__(self, x, message, ng, nfev, fx, constr=None):
        self.x = x
        self.message = message
        self.ng = ng
        self.nfev = nfev
        self.fx = fx
        self.constr = constr

    def __str__(self):
        s = "Optimization result:\n"
        s += "====================\n"
        s += f"x = {self.x}\n"
        s += f"message = {self.message}\n"
        s += f"ng = {self.ng}\n"
        s += f"nfev = {self.nfev}\n"
        s += f"fx = {self.fx}\n"
        if self.constr:
            s += f"constr = {self.constr}\n"
        return s
