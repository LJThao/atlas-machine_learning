#!/usr/bin/env python3
"""Function that calculates the integral of a poly"""


def poly_integral(poly, C=0):
    """Calculate the integral of a poly"""
    # if poly is not empty (True) execute function
    # if poly is empty (False) return none
    if poly:
        # returns a list of coeff of the integral of a poly
        return [C] + [(coeff / (i + 1)) if (coeff / (i + 1))
                  % 1 != 0 else int(coeff / (i + 1)) for i,
                  coeff in enumerate(poly)]
    return None
