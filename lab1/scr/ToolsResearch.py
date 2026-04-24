import math
import time
import numpy as np
import pandas as pd

from scr.Optimization import optimize
from scr.ConstructiveNumber import ConstructiveNumber

class CountingBox:
    def __init__(self, inner):
        self.inner = inner
        self.arity = inner.arity
        self.name = inner.name

        self.value_calls = 0
        self.grad_calls = 0
        self.hess_calls = 0

        self.value_points = []
        self.grad_points = []

    def __call__(self, x):
        self.value_calls += 1
        self.value_points.append([float(v.to_real()) if isinstance(v, ConstructiveNumber) else float(v) for v in x])
        return self.inner(x)

    def gradient(self, x):
        self.grad_calls += 1
        self.grad_points.append([float(v.to_real()) if isinstance(v, ConstructiveNumber) else float(v) for v in x])
        return self.inner.gradient(x)

    def hessian(self, x):
        self.hess_calls += 1
        return self.inner.hessian(x)


def reference_minimum(factory):
    box = factory()
    if box.name.startswith("quadratic"):
        A = np.array(box.A, dtype=float)
        b = np.array(box.b, dtype=float)
        x_star = -np.linalg.solve(A, b)
        f_star = float(box(x_star.tolist()))
        return x_star, f_star

    if box.name == "rosenbrock_3":
        x_star = np.ones(3, dtype=float)
        f_star = 0.0
        return x_star, f_star

    raise ValueError()


def tune_learning_rate(factory, x0, lr_grid, max_iter=3000, tol=1e-8):
    rows = []
    for lr in lr_grid:
        cb = CountingBox(factory())
        t0 = time.perf_counter()
        res = optimize(cb, x0, method="gradient_descent", learning_rate=lr, max_iter=max_iter, tol=tol)
        dt = time.perf_counter() - t0
        rows.append({
            "lr": lr,
            "converged": res.converged,
            "f_best": float(res.f_best),
            "iterations": res.iterations,
            "time_sec": dt,
            "f_calls": cb.value_calls,
            "g_calls": cb.grad_calls,
        })

    df = pd.DataFrame(rows).sort_values(["f_best", "iterations", "time_sec"], ascending=[True, True, True])
    return df, float(df.iloc[0]["lr"])


def run_method(factory, x0, method, x_star, f_star, **kwargs):
    cb = CountingBox(factory())
    t0 = time.perf_counter()
    res = optimize(cb, x0, method=method, **kwargs)
    dt = time.perf_counter() - t0

    return {
        "result": res,
        "history": res.history,
        "trajectory_gd": np.array(cb.grad_points, dtype=float) if cb.grad_points else None,
        "row": {
            "method": method,
            "converged": res.converged,
            "iterations": res.iterations,
            "time_sec": dt,
            "f_calls": cb.value_calls,
            "g_calls": cb.grad_calls,
            "f_best": float(res.f_best),
            "f_gap": float(res.f_best - f_star),
            "x_err_l2": float(np.linalg.norm(np.array(res.x_best, dtype=float) - x_star)),
        },
    }


def nelder_mead_trace(factory, x0, step=1.0, max_iter=1000, tol=1e-8):
    box = factory()
    x0 = np.array(x0, dtype=float)
    n = len(x0)

    simplex = [x0.copy()]
    for i in range(n):
        p = x0.copy()
        p[i] += step
        simplex.append(p)

    traj = []
    hist = []

    reflection = 1.0
    expansion = 2.0
    contraction = 0.5
    shrink = 0.5

    for _ in range(max_iter):
        simplex = sorted(simplex, key=lambda p: float(box(p.tolist())))
        values = [float(box(p.tolist())) for p in simplex]

        best = simplex[0]
        worst = simplex[-1]

        traj.append(best.copy())
        hist.append(values[0])

        if abs(values[-1] - values[0]) <= tol:
            break

        centroid = np.mean(simplex[:-1], axis=0)

        reflected = centroid + reflection * (centroid - worst)
        f_ref = float(box(reflected.tolist()))

        if values[0] <= f_ref < values[-2]:
            simplex[-1] = reflected
            continue

        if f_ref < values[0]:
            expanded = centroid + expansion * (reflected - centroid)
            f_exp = float(box(expanded.tolist()))
            simplex[-1] = expanded if f_exp < f_ref else reflected
            continue

        if f_ref < values[-1]:
            contracted = centroid + contraction * (reflected - centroid)
        else:
            contracted = centroid + contraction * (worst - centroid)

        f_con = float(box(contracted.tolist()))
        if f_con < min(values[-1], f_ref):
            simplex[-1] = contracted
            continue

        best = simplex[0]
        simplex = [best] + [best + shrink * (p - best) for p in simplex[1:]]

    return np.array(traj, dtype=float), hist

def f_slice_2d(factory, x_ref, i=0, j=1):
    box = factory()
    def _f(u, v):
        x = np.array(x_ref, dtype=float).copy()
        x[i] = u
        x[j] = v
        return float(box(x.tolist()))
    return _f

def to_cn_vector(x, eps):
    return [ConstructiveNumber.from_real(float(xi), eps) for xi in x]

def int_log10(n: int) -> float:
    if n <= 0:
        raise ValueError()
    bits = n.bit_length()
    shift = max(bits - 53, 0)
    mant = n >> shift
    return math.log10(float(mant)) + shift * math.log10(2.0)

def fraction_log10(fr):
    if fr == 0:
        return -np.inf
    num = abs(fr.numerator)
    den = fr.denominator
    return int_log10(num) - int_log10(den)

def cn_half_width(v: ConstructiveNumber):
    return (v.b - v.a) / 2

def epsilon_dynamics_on_trajectory(factory, trajectory, eps0):
    box = factory()
    logs = []
    for x in trajectory:
        x_cn = to_cn_vector(x, eps0)
        f_cn = box(x_cn)
        eps_f = cn_half_width(f_cn)
        logs.append(fraction_log10(eps_f))
    return np.array(logs, dtype=float)
