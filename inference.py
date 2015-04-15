"""Specialisation of the inference module in the pgmlib for the ocr task
"""
import numpy as np
from operator import add
import copy

from pgmlib import factor
from pgmlib.factor import factor_product
from pgmlib.factor import factor_marginalization
from pgmlib import inference

def _compute_edges(cliques):
    edges = np.diag([1] *len(cliques))
    for i in range(len(cliques)):
        for j in range(len(cliques)):
            if set(cliques[i].var).intersection(set(cliques[j].var)) and not len(cliques[i].var) == len(cliques[j].var):
                edges[i, j] = 1
    return edges


def compute_exact_marginals_ocr_clique_tree(cliques, max_sum=True):
    """Computes exact marginals using a clique tree
    Algorithem depends on order of singelton and piecwise factors
    """
    #compute edges
    edges = _compute_edges(cliques)

    # Calibrate Tree
    tree = inference.CliqueTree(cliques, edges)
    tree.calibrate(max_sum)

    # Compute Marginals
    return _compute_marginals(tree.cliqueList)


def _compute_marginals(cliques):
    marginals = []
    for var in sorted({v for f in cliques for v in f.var}):
        for clique in (c for c in cliques if var in c.var):
            marg = copy.deepcopy(clique)
            for other_var in [v for v in clique.var if not v == var]:
                marg = factor_marginalization(marg, other_var, max)
            marginals.append(marg)
            break
    return marginals

def map_singelton_ocr(factors):
    """Takes a list of factors reduced to only one variable and returns the most
    probable assigments to these factors as a character.
    """
    return [chr(i - 1 + ord("a")) for i in inference.map_singelton(factors)]