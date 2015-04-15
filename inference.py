"""Specialisation of the inference module in the pgmlib for the ocr task
"""
import numpy as np
from operator import add

from pgmlib import factor
from pgmlib.factor import factor_product
from pgmlib.factor import factor_marginalization
from pgmlib import inference


def compute_exact_marginals_ocr_clique_tree(singelton_factors, picewise_factors):
    """Computes exact marginals using a clique tree
    Algorithem depends on order of singelton and piecwise factors
    """
    cliques = singelton_factors + picewise_factors
    edges = np.zeros((len(cliques), len(cliques)))
    for i in range(len(cliques)):
        for j in range(len(cliques)):
            if set(cliques[i].var).intersection(set(cliques[j].var)):
                edges[i, j] = 1

    # Calibrate Tree
    tree = inference.CliqueTree(cliques, edges)
    tree.calibrate(True)

    # Compute Marginals
    marginals = []
    for v in sorted({v for f in cliques for v in f.var}):
        for clique in (c for c in tree.cliqueList if v in c.var):
            other_vars =[va for va in clique.var if not va == v]
            marg = factor_marginalization(clique, other_vars[0] if other_vars else None, max)
            for other_var in other_vars[1:]:
                marg = factor_marginalization(marg, other_var, max)
            marginals.append(marg)
            break
    return marginals

def map_singelton_ocr(factors):
    """Takes a list of factors reduced to only one variable and returns the most
    probable assigments to these factors as a character.
    """
    return [chr(i - 1 + ord("a")) for i in inference.map_singelton(factors)]