from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence, Union

try:
    from .ConstructiveNumber import ConstructiveNumber
except ImportError:
    from ConstructiveNumber import ConstructiveNumber

Scalar = Union[int, float, Fraction, ConstructiveNumber]
Vector = Sequence[Scalar]
Matrix = Sequence[Sequence[Scalar]]


class BlackBox:
    arity: int
    name: str

    def __call__(self, x: Vector) -> Scalar:
        return self.value(x)

    def value(self, x: Vector) -> Scalar:
        raise NotImplementedError

    def gradient(self, x: Vector) -> list[Scalar]:
        raise NotImplementedError

    def hessian(self, x: Vector) -> list[list[Scalar]]:
        raise NotImplementedError

    def _check_arity(self, x: Vector) -> None:
        if len(x) != self.arity:
            raise ValueError()


@dataclass(frozen=True)
class QuadraticBlackBox(BlackBox):
    A: Matrix
    b: Vector
    c: Scalar = 0
    name: str = "quadratic"

    def __post_init__(self) -> None:
        n = len(self.A)
        if n == 0:
            raise ValueError()
        if any(len(row) != n for row in self.A):
            raise ValueError()
        if len(self.b) != n:
            raise ValueError()
        object.__setattr__(self, "arity", n)

    def value(self, x: Vector) -> Scalar:
        self._check_arity(x)

        quad_term: Scalar = 0
        for i in range(self.arity):
            for j in range(self.arity):
                quad_term += self.A[i][j] * x[i] * x[j]

        linear_term: Scalar = 0
        for i in range(self.arity):
            linear_term += self.b[i] * x[i]

        return 0.5 * quad_term + linear_term + self.c

    def gradient(self, x: Vector) -> list[Scalar]:
        self._check_arity(x)

        grad: list[Scalar] = []
        for i in range(self.arity):
            s: Scalar = self.b[i]
            for j in range(self.arity):
                s += self.A[i][j] * x[j]
            grad.append(s)
        return grad

    def hessian(self, x: Vector) -> list[list[Scalar]]:
        self._check_arity(x)
        return [list(row) for row in self.A]


class Rosenbrock3BlackBox(BlackBox):
    arity = 3
    name = "rosenbrock_3"

    def value(self, x: Vector) -> Scalar:
        self._check_arity(x)
        x1, x2, x3 = x
        t1 = x2 - x1 * x1
        t2 = x3 - x2 * x2
        return 100 * t1 * t1 + (1 - x1) * (1 - x1) + 100 * t2 * t2 + (1 - x2) * (1 - x2)

    def gradient(self, x: Vector) -> list[Scalar]:
        self._check_arity(x)
        x1, x2, x3 = x

        g1 = -400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1)
        g2 = 200 * (x2 - x1 * x1) - 400 * x2 * (x3 - x2 * x2) - 2 * (1 - x2)
        g3 = 200 * (x3 - x2 * x2)

        return [g1, g2, g3]

    def hessian(self, x: Vector) -> list[list[Scalar]]:
        self._check_arity(x)
        x1, x2, x3 = x

        h11 = 1200 * x1 * x1 - 400 * x2 + 2
        h12 = -400 * x1
        h13 = 0

        h21 = -400 * x1
        h22 = 1200 * x2 * x2 - 400 * x3 + 202
        h23 = -400 * x2

        h31 = 0
        h32 = -400 * x2
        h33 = 200

        return [
            [h11, h12, h13],
            [h21, h22, h23],
            [h31, h32, h33],
        ]


def make_well_conditioned_quadratic_6() -> QuadraticBlackBox:
    A = [
        [1.00, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.05, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.95, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.10, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.90, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.00],
    ]
    b = [0.0, -1.0, 2.0, 0.5, -0.5, 1.5]
    return QuadraticBlackBox(A=A, b=b, c=0.0, name="quadratic_6_good_condition")


def make_bad_conditioned_quadratic_4() -> QuadraticBlackBox:
    A = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 50.0, 0.0],
        [0.0, 0.0, 0.0, 100.0],
    ]
    b = [1.0, -2.0, 0.5, 3.0]
    return QuadraticBlackBox(A=A, b=b, c=0.0, name="quadratic_4_bad_condition")


def make_rosenbrock_3() -> Rosenbrock3BlackBox:
    return Rosenbrock3BlackBox()