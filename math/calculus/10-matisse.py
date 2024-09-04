#!/usr/bin/env python3
"""Function to calculate derivative of a poly"""


def poly_derivative(poly):
    """Calculate derivative of a poly"""
    # if the the poly list is empty, then return
    if not poly:
        return None
    # if the poly has one element then return as derivative
    elif len(poly) == 1:
        return [0]
    # calculates the derivative by multiplying, then skip constant x^0
    return [coef * i for i, coef in enumerate(poly) if i > 0]
