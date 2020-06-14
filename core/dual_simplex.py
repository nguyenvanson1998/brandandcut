import numpy as np
import math
from fractions import Fraction

from ortools.algorithms import pywrapknapsack_solver

from core.simplex import Simplex, zero, one
from core.utils import knapsack


class DualSimplex(Simplex):
    def duplicate(self):
        model = DualSimplex()
        model.M = self.M
        model.N = self.N
        model.base = np.copy(self.base)
        model.a = np.copy(self.a)
        return model

    def dual_pivot(self):
        while True:
            if np.min(self.a[:self.M, -1]) >= 0:
                return True
            for r in range(self.M):
                if self.a[r, -1] < 0 and np.min(self.a[r, :self.N]) >= 0:
                    return False
            r = np.argmin(self.a[:self.M, -1])

            max_ratio = -float('inf')
            s = -1
            for j in range(self.N):
                if self.a[r, j] < 0 and self.a[-1, j] / self.a[r, j] > max_ratio:
                    max_ratio = self.a[-1, j] / self.a[r, j]
                    s = j
            self.pivot(r, s)

    def remove_constraints(self, rows):
        self.a = np.delete(self.a, rows, 0)
        self.base = np.delete(self.base, rows)
        self.M = self.base.size

    def add_leq_constraint(self, g, b):
        if len(g) < self.N:
            g.extend([zero] * (self.N - len(g)))
        elif len(g) > self.N:
            print('Constraint length violated!')
            exit(-1)
        self.a = np.insert(self.a, -1, zero, axis=1)
        self.a = np.insert(self.a, -1, g + [one, b], axis=0)
        self.base = np.append(self.base, self.N)
        # self.c = np.append(self.c, zero)
        self.M += 1
        self.N += 1

        self.normalize(-1)

    def validate(self, int_list):
        for p in range(self.M):
            if self.base[p] in int_list \
                    and self.a[p, -1].denominator != 1:
                return p
        return -1

    def add_int_cut(self, j, v, leq=True):
        g = [zero] * self.N
        if leq:
            g[j] = one
            b = v
        else:
            g[j] = -one
            b = -v
        self.add_leq_constraint(g, b)

        # b = -1
        # for i, p in enumerate(self.base):
        #     if p == j:
        #         b = p
        #         break
        # if leq:
        #     self.a[-2, :] -= self.a[b, :]
        # else:
        #     self.a[-2, :] += self.a[b, :]
        #
        # return g, b
        # status = self.simplex.dual_pivot()
        # return status

    def add_gomory_cut(self, p):
        g = [zero] * self.N
        for q in range(self.N):
            a = self.a[p, q]
            g[q] = Fraction(math.floor(a)) - a
        b = Fraction(math.floor(self.a[p, -1])) - self.a[p, -1]
        self.add_leq_constraint(g, b)
        # return g, b
        # print('=== Add Gomory cut ===')
        # print_table(self.simplex.a)
        # status = self.simplex.dual_pivot()
        # print('=== Solve Gomory cut ===')
        # print_table(self.simplex.a)
        # return status

    def add_cover_cut(self, A, b):
        int_list = list(range(len(A[0])))
        x = self.get_vars()
        sep_c = [one - x[i] for i in int_list]

        for k in range(len(b)):
            # sep_c.extend([zero] * (self.N - len(sep_c)))
            sep_A = [A[k][i] for i in int_list]
            # sep_a.extend([zero] * (self.N - len(sep_a)))
            sep_b = b[k]

            # print('separation:')
            # print(sep_A)
            # print(sep_b)
            # print(sep_c)
            # print()

            offset = sum(sep_c)
            capacity = sum(sep_A) - sep_b - Fraction(1, 10000)
            if capacity < 0:
                continue

            obj, x_vals = knapsack(sep_c, sep_A, capacity)
            if offset - obj >= 1:
                continue

            cover_cut = [zero if val else one for val in x_vals]
            cover_cut.extend([zero] * (self.N - len(cover_cut)))
            C = sum(cover_cut) - one
            self.add_leq_constraint(cover_cut, C)
            # print('found cover cut:')
            # print(list(map(int, cover_cut)))
            # print(C)
            # print('---')
            return True
        return False
