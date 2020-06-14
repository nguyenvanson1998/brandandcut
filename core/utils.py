from fractions import Fraction
import numpy as np

from ortools.linear_solver import pywraplp


def read_data(path):
    with open(path) as f:
        N, M = list(map(int, f.readline().split()))
        c = list(map(Fraction, f.readline().split()))
        A = []
        b = []
        for _ in range(M):
            line = list(map(Fraction, f.readline().split()))
            A.append(line[:-1])
            b.append(line[-1])

        return A, b, c


def knapsack(values, weights, capacity):
    # print('knapsack')
    # print(values)
    # print(weights)
    # print(capacity)
    # print('===')

    solver = pywraplp.Solver('knapsack_model',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    n = len(values)

    x = np.empty(n, dtype=object)
    for i in range(n):
        x[i] = solver.BoolVar('x%i' % i)

    bag = solver.Constraint(0, float(capacity))
    for i in range(n):
        bag.SetCoefficient(x[i], float(weights[i]))

    obj = solver.Objective()
    for i in range(n):
        obj.SetCoefficient(x[i], float(values[i]))
    obj.SetMaximization()

    status = solver.Solve()
    if not status == pywraplp.Solver.OPTIMAL:
        return None

    x_vals = [False] * n
    for i in range(n):
        x_vals[i] = False if int(x[i].solution_value()) == 0 else True

    # print(x_vals)
    # print('===')
    return obj.Value(), x_vals
