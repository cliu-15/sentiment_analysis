from __future__ import division
import numpy as np
import math
import itertools

def sign_test(data1, data2):
    """
    Performs sign test and returns p-value.
    """
    plus = 0
    minus = 0
    null = 0
    q = 0.5

    for i in range(len(data1)):
        if data1[i] > data2[i]:
            plus = plus + 1
        elif data1[i] < data2[i]:
            minus = minus + 1
        else:
            null = null + 1

    N = 2 * math.ceil(null / 2) + plus + minus
    k = math.ceil(null / 2) + min(plus, minus)
    p = 0.0
    for i in range(int(k)):
        p = p + 2 * (math.factorial(N) / (math.factorial(i) * math.factorial(N - i))) * (q ** i) * ((1 - q) ** (N- i))
    return p


def mean_diff(data_1, data_2, swaps):
    """
    Calculates the absolute value of the differences between the means of two lists.
    """
    mean_diff_1 = 0.0
    mean_diff_2 = 0.0
    length = len(data_1)
    for i in range(len(swaps)):
        if swaps[i] == 0:
            mean_diff_1 += data_1[i] / length
            mean_diff_2 += data_2[i] / length
        else:
            mean_diff_1 += data_2[i] / length
            mean_diff_2 += data_1[i] / length
    return abs(mean_diff_2 - mean_diff_1)


def create_permutation_list(length, R):
    """
    Returns a list of R possible permutations of given length, with each value being 0 or 1.
    """
    count = 0
    perm_list = set()

    for perm in itertools.product(range(2), repeat = length):
        count += 1
        if len(perm_list) == R:
            break
        else:
            perm_list.add(perm)
    return list(perm_list)


def monte_carlo(R, data_1, data_2):
    """
    Performs the Monte Carlo permutation test and returns the p-value.
    """
    permutation_list = create_permutation_list(len(data_1), R)
    orig_diff = mean_diff(data_1, data_2, [0]*len(data_1))
    s = 0.0
    for i in range(R):
        np.random.shuffle(permutation_list)
        if mean_diff(data_1, data_2, permutation_list[0]) >= orig_diff:
            s += 1
    p = (s + 1) / (R + 1)
    return p