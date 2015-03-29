"""Contains factor class and related functions"""

import numpy as np
from operator import mul
from functools import reduce

class Factor:
    """The nodes in a Bayesian/Markov network are represented by Factors.
    A Factor has a number of variables (there names are stored in the var
    property), and each assigment (a specified value for each variable)
    has a corresponding probability (of weight in case of Markov networks)
    """
    def __init__(self, var, card, val=None):
        """
        var = A list of names of the variables
        card = The cardinalities of the variables in a list
                (eg [2, 2] for two binary variables)
        val = The values of the probabilities for the assigment.
              The lenght of the  iterable must be reduce(mul, card)
        """
        self.var = var
        self.card = card
        self._val = val or np.zeros(reduce(mul, card), dtype=np.int16)


    def get_val_of_assigment(self, assigment):
        """Get the value of a assigment
        if assigment is a dictionary: key = name of variable, value = value
        (benefit of dictionary: there can be key value pairs in the assigment
        which are not relevant for this particular factor)
        """
        if isinstance(assigment, dict):
            index = self._assigment_to_index([assigment[name] for name in self.var])
            return self._val[index]
        else:
            raise Exception("Not implemented")

    def set_val_of_assigment(self, assigment, value):
        """Set the value of a assigment
        if assigment is a dictionary: key = name of variable, value = value
        """
        if isinstance(assigment, dict):
            index = self._assigment_to_index([assigment[name] for name in self.var])
            self._val[index] = value
        else:
            raise Exception("Not implemented")

    def get_all_assigments_d(self):
        """Returns all possible assigments which are possible for this factor
        as a list of dictionaries
        (convenience function)
        """
        assigmens =  [self._index_to_assigment(i) for i in range(len(self._val))]
        return [{name:ai for name, ai in  zip(self.var, a)} for a in assigmens]

    def _assigment_to_index(self, assigment):
        """Returns the index (of the _val field) of the corresponding
        assigment, the i th element of the assigment can have values from
        1 to self.card[i]"""
        # Idea: index = a1 * 1 + a2 * c1 + a3 * c1 * c2 + ...
        # where a = assigment, c = cardinality
        base = [1] + list(np.cumprod(self.card[:-1]))
        return sum((a-1)*b for a, b in zip(assigment, base))

    def _index_to_assigment(self, index):
        """Returns assigment to the corresponding index (of the _val field)"""
        # Idea: index = a1 * 1 + a2 * c1 + a3 * c1 * c2 + ...
        # where a = assigment, c = cardinality
        assigment = []
        for c in self.card:
            assigment.append(index % c + 1)
            index = index // c # alt: (index - (index % b)) / b
        return assigment

def factor_product(factorA, factorB):
    """ Computes the factor product of two factors
    """
    # CHeck if empty
    if len(factorA.var) == 0: return factorB
    if len(factorB.var) == 0: return factorA

    # Check that the commen variables in A and B have same cardinality
    sanity_check_coomen_var_same_card(factorA, factorB)

    # Create Empty result Factor
    Cvar, Ccard = zip(*sorted(list(
                    set(zip(factorA.var, factorA.card)).union(
                    set(zip( factorB.var, factorB.card))))))
    factorC = Factor(Cvar, Ccard)

    # Fill Factor
    assignments = factorC.get_all_assigments_d()
    for a in assignments:
        newval = factorA.get_val_of_assigment(a) * factorB.get_val_of_assigment(a)
        factorC.set_val_of_assigment(a, newval)
    return factorC

def factor_marginalization(factor, variable):
    """Sums a variable out of a factor
    """
    # Check if empty
    if not factor.var or not variable: return factor

    # Create result factor
    Cvar = factor.var[:]; Cvar.remove(variable)
    if not Cvar: raise Exception("Resultant factor has empty scope")
    Ccard = [factor.card[factor.var.index(v)] for v in Cvar]
    Cfactor = Factor(Cvar, Ccard)

    # Fill Factor
    for a in Cfactor.get_all_assigments_d():
        toset = 0  # Sum over all relevant assigments
        for x in range(1, factor.card[factor.var.index(variable)] + 1):
            newd = a
            newd[variable] = x
            toset += factor.get_val_of_assigment(newd)
        Cfactor.set_val_of_assigment(a, toset)
    return Cfactor

def observe_evidence(factor, evidence):
    """Modify a factor such that it is consistent with the observed evidence
    Evidence is a tubple (variable, value)
    """
    # Check if empty
    if not factor.var or not evidence[0] in factor.var: return factor

    # Create result factor
    Cvar = factor.var[:]; Cvar.remove(evidence[0])
    if not Cvar: raise Exception("Resultant factor has empty scope")
    Ccard = [factor.card[factor.var.index(v)] for v in Cvar]
    Cfactor = Factor(Cvar, Ccard)

    # Fill Factor
    for a in Cfactor.get_all_assigments_d():
        newd = a
        newd[evidence[0]] = evidence[1]
        Cfactor.set_val_of_assigment(a, factor.get_val_of_assigment(newd))
    return Cfactor

def sanity_check_coomen_var_same_card(factorA, factorB):
    """Checks if the commen variables have the same cardinality"""
    if not np.intersect1d(factorA.var, factorB.var):
        #they have commen elements
        if not all(factorA.card[np.in1d(factorA.var, factorB.var)] ==
                   factorB.card[np.in1d(factorB.var, factorA.var)]):
            raise Exception("Inconsistens variables",
            "Different cardinalitys for the same variable detected")


