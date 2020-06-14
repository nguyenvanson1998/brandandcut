import numpy as np
from fractions import Fraction

from core.simplex import one, zero, print_table
from core.dual_simplex import DualSimplex


class Model:
    OPTIMAL = 0
    UNBOUNDED = 1
    INFEASIBLE = -1
    
    def __init__(self, A, b, c):
        self.M = len(b)
        self.N = len(c)
        self.A = np.full([self.M, self.N + self.M], zero, dtype=Fraction)
        self.A[:self.M, :self.N] = np.array(A)
        for p in range(self.M):
            self.A[p, self.N + p] = one
        self.b = np.array(b)
        self.c = np.negative(c + [zero] * self.M)
        self.N += self.M
        self.simplex = None

    def first_phase(self, A, b):
        tab = np.full([self.M + 1, self.N + self.M + 1], zero, dtype=Fraction)
        for i in range(self.M):
            if b[i] >= 0:
                tab[i, :self.N] = A[i, :]
                tab[i, -1] = b[i]
            else:
                tab[i, :self.N] = np.negative(A[i, :])
                tab[i, -1] = -b[i]
            tab[i, self.N + i] = one
        base = np.array(range(self.N, self.N + self.M))
        tab[-1, self.N:self.N+self.M] = [one] * self.M
        simplex = DualSimplex.init_model(tab, base)

        # print('=== Phase 1 init ===')
        # print_table(simplex.a)
        simplex.primal_pivot()
        # print('=== Phase 1 end ===')
        # print_table(simplex.a)

        if simplex.get_objective() > zero:
            return None

        removed_rows = []
        move_out = []
        for p, i in enumerate(simplex.base):
            if self.N <= i < self.N + self.M:
                move_out.append(p)
        for p in move_out:
            moved = False
            for q in range(self.N):
                if q not in simplex.base and simplex.a[p, q] != zero:
                    simplex.pivot(p, q)
                    moved = True
                    break
            if not moved:
                removed_rows.append(p)
        simplex.remove_constraints(removed_rows)
        self.M -= len(removed_rows)

        return simplex

    def find_BFS(self):
        if min(self.b) >= 0:
            tab = np.empty([self.M+1, self.N+1], dtype=Fraction)
            tab[:-1, :-1] = self.A
            tab[:-1, -1] = self.b
            tab[-1, :-1] = self.c
            tab[-1, -1] = zero
            simplex = DualSimplex.init_model(tab, np.array(range(self.N-self.M, self.N)))
        else:
            BFS = self.first_phase(self.A, self.b)
            if BFS is None:
                return None
            tab = np.hstack((BFS.a[:, :self.N], BFS.a[:, -1].reshape(-1, 1)))
            tab[-1, :self.N] = self.c
            simplex = DualSimplex.init_model(tab, BFS.base)
        return simplex

    def lp_solve(self):
        self.simplex = self.find_BFS()
        if self.simplex is None:
            return Model.INFEASIBLE

        # print('=== Phase 2 init ===')
        # print_table(self.simplex.a)
        if not self.simplex.primal_pivot():
            return Model.UNBOUNDED
        # print('=== Phase 2 end ===')
        # print_table(self.simplex.a)
        return Model.OPTIMAL

    def mip_solve(self, int_list=None):
        if int_list is None:
            int_list = list(range(len(self.c) - len(self.b)))
        else:
            print('Not implemented yet!!!')
            exit(-1)

        status = self.lp_solve()
        if status != Model.OPTIMAL:
            return status

        while True:
            gomory_index = self.simplex.validate(int_list)
            if gomory_index == -1:
                return Model.OPTIMAL
            self.simplex.add_gomory_cut(gomory_index)
            if not self.simplex.dual_pivot():
                return Model.INFEASIBLE

    def print_solution(self, status):
        print('=== Result  ===')
        if status == Model.INFEASIBLE:
            print('NO SOLUTION')
        elif status == Model.UNBOUNDED:
            print('UNBOUNDED')
        else:
            print('OPTIMAL')
            print('Objective: f = ' + str(-self.simplex.get_objective()))
            print('Variables: x = (' + ', '.join(list(
                map(str, self.simplex.get_vars()[:len(self.c) - len(self.b)]))) + ')')
