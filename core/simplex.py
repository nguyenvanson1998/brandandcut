import numpy as np
from fractions import Fraction


zero = Fraction(0)
one = Fraction(1)


class Simplex:
    def __init__(self):
        self.M = 0
        self.N = 0
        self.base = None
        self.a = None

    @classmethod
    def init_model(cls, tab, base):
        model = cls()
        model.M = tab.shape[0] - 1
        model.N = tab.shape[1] - 1
        model.base = base
        # self.c = np.array(tab[-1, :-1])
        model.a = tab
        model.normalize_objective()
        return model

    def normalize_objective(self):
        for r, j in enumerate(self.base):
            self.a[r, :] /= self.a[r, j]
            if self.a[-1, j] != zero:
                self.a[-1, :] -= self.a[-1, j] * self.a[r, :]

    def normalize(self, p=-1):
        p = self.M + p if p < 0 else p
        self.a[p, :] /= self.a[p, self.base[p]]
        for r, j in enumerate(self.base):
            if r != p and self.a[p, j] != zero:
                self.a[p, :] -= self.a[p, j] * self.a[r, :]

    def pivot(self, p: int, q: int):
        self.base[p] = q
        for i in range(self.M+1):
            for j in range(self.N+1):
                if i != p and j != q:
                    self.a[i][j] -= self.a[p][j] * self.a[i][q] / self.a[p][q]

        for i in range(0, self.M+1):
            if i != p:
                self.a[i][q] = zero
        for j in range(self.N+1):
            if j != q:
                self.a[p][j] /= self.a[p][q]
        self.a[p][q] = one

    def primal_pivot(self):
        while True:
            q = -1
            for j in range(self.N):
                if self.a[-1, j] < 0:
                    q = j
                    break
            if q == -1:
                return True
            min_ratio = float('inf')
            p = -1
            for i in range(self.M):
                if self.a[i, q] > 0 and self.a[i, -1] / self.a[i, q] < min_ratio:
                    min_ratio = self.a[i, -1] / self.a[i, q]
                    p = i
            if p == -1:
                return False
            self.pivot(p, q)

    def get_objective(self):
        return -self.a[-1, -1]

    def get_vars(self):
        x = np.full(self.N, zero, dtype=Fraction)
        for p, i in enumerate(self.base):
            x[i] = self.a[p, -1]
        return x


def print_table(a):
    s = a.shape
    for i in range(s[0]):
        print(' '.join(list(map(str, list(a[i, :])))))
