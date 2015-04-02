"""This module contains various interence algorithems for pgm's
"""
import numpy as np
from functools import reduce
from pgmlib import factor

def map_simple(factors):
    """Calculates the maximum a posteriori the simplest possible way
    """
    # idee: alle faktoren die die selbe variable beinhalten werden multiplizirt
    #       dann das maximum für diesen einen faktor bestimmt
    variables = {x for f in factors for x in f.var}
    results = dict()
    for var in variables:
        tomul = []
        for f in factors:
            if var in f.var:
                factors.remove(f)
                tomul.append(f)
        jointfactor = reduce(lambda x,y : factor.factor_product(x, y), tomul)
        factors.append(jointfactor)#factor.factor_marginalization(jointfactor, var))
        margfactor= jointfactor
        for ov in jointfactor.var:
            if not ov == var:
                margfactor = factor.factor_marginalization(margfactor, ov)
        results[var] = chr(ord("a") + np.argmax(margfactor._val) )
    return sorted(results.items())


def variable_elimination(factors):
    """Uses the variable elemination algorithem to compute the joint distribution
    """
    # Algorithem:
    # 1) Reduce all factors by evidence
    # 2) Eliminate non query variable
    # 3) Multiply all remaining factors
    raise Exception("Not yet implemented")


def main():
    pass

if __name__ == '__main__':
    main()
