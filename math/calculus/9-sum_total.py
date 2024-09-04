#!/usr/bin/env python3
"""Function for Sum Total"""


def summation_i_squared(n):
    """Calculating the sum"""
    # if n is less than 1 then true, return
    if n < 1:
        return None
    # if n is 1 or greater then go ahead and
    # apply the lambda function and square each ints
    # from 1 to n to calculate sum
    squares = list(map(lambda n: n**2, range(1, n + 1)))
    return sum(squares)
