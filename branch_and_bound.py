import math
import sys

from core.utils import read_data
from core.two_phase import Model
from core.dual_simplex import DualSimplex, zero


class BB:
    def __init__(self, A, b, c, int_list=None):
        self.A = A
        self.b = b
        self.c = c
        self.int_list = int_list
        if self.int_list is None:
            self.int_list = list(range(len(self.c)))
        self.best = float('inf')
        self.solution = None
        self.node_count = 0

    def find_frac(self, x):
        for j in self.int_list:
            if x[j].denominator != 1:
                return j, x[j]
        return -1, zero

    def branching(self, node: DualSimplex, cut):
        # print('Current node:')
        # print_table(node.a)
        # print(node.get_vars())
        if node.get_objective() >= self.best:
            return
        self.node_count += 1

        while True:
            x = node.get_vars()
            # print(x)
            j, v = self.find_frac(x)
            if j == -1:
                self.best = node.get_objective()
                self.solution = node
                return
            if not cut:
                break
            if not node.add_cover_cut(self.A, self.b):
                break
            if not node.dual_pivot() or node.get_objective() >= self.best:
                return
            # break

        # print('left')
        left = node.duplicate()
        left.add_int_cut(j, math.floor(v), True)
        if left.dual_pivot():
            self.branching(left, cut)
        # print_table(left.a)
        # left.dual_pivot()
        #
        # print(left.get_vars())
        # print_table(left.a)

        # print('right')
        right = node.duplicate()
        right.add_int_cut(j, math.ceil(v), False)
        if right.dual_pivot():
            self.branching(right, cut)
        # print_table(right.a)
        # print(right.get_vars())
        # right.dual_pivot()
        # print_table(right.a)

        # print(right.get_vars())

    def solve(self, cut=False):
        self.node_count = 0

        BFS = Model(self.A, self.b, self.c)
        status = BFS.lp_solve()
        if status != Model.OPTIMAL:
            BFS.print_solution(status)
            return

        start_node = BFS.simplex
        self.branching(start_node, cut)

        if self.solution is None:
            BFS.print_solution(Model.INFEASIBLE)
        else:
            BFS.simplex = self.solution
            BFS.print_solution(Model.OPTIMAL)
        # print('Node count = %i' % self.node_count)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Missing input file!!!')
    else:
        A, b, c = read_data(sys.argv[1].strip())
        model = BB(A, b, c)
        model.solve(cut=True)
