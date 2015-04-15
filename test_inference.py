import unittest
import numpy as np
from operator import add
import copy


from inference import map_singelton_ocr
from inference import compute_exact_marginals_ocr_clique_tree
from inference import _compute_edges
from inference import _compute_marginals

from pgmlib.factor import Factor
from pgmlib import factor
from pgmlib.factor import factor_marginalization
from pgmlib.inference import CliqueTree
import data_processing
import ocr

class TestMapSingeltonOcr(unittest.TestCase):

    def test_map(self):
        a = Factor(["1"], [26], np.array(range(26)))
        b = Factor(["1"], [26], np.array(sorted(range(0, 26), reverse=True)))
        self.assertListEqual(["z", "a"], map_singelton_ocr([a, b]))

class TestExactMarginals(unittest.TestCase):
    def setUp(self):
        # two singelton and one pair
        self.factor1 = Factor(["1"], [2], np.array([0.2, 0.8]))
        self.factor2 = Factor(["2"], [2], np.array([0.1, 0.9]))
        self.pair1 = Factor(["1", "2"], [2, 2], np.array([0.99, 0.01, 0.01, 0.01]))
        self.factor3 = Factor(["3"], [2], np.array([0.91, 0.09]))
        self.factor4 = Factor(["4"], [2], np.array([0.81, 0.19]))
        self.pair2 = Factor(["2", "3"], [2, 2], np.array([0.5, 0.01, 0.01, 0.05]))
        self.pair3 = Factor(["3", "4"], [2, 2], np.array([0.01, 0.01, 0.01, 0.99]))

    def test_c_e_m_o_c_t(self):
        marginals = compute_exact_marginals_ocr_clique_tree(
                                    [self.factor1, self.factor2, self.pair1])
        self.factor1._val = np.log(self.factor1._val)
        self.factor2._val = np.log(self.factor2._val)
        self.pair1._val = np.log(self.pair1._val)
        comined = factor.factor_product(self.factor2,
                    factor.factor_product(self.factor1, self.pair1, add), add)
        one = factor.factor_marginalization(comined, "2", max)
        two = factor.factor_marginalization(comined, "1", max)
        for a, b in zip(marginals[0]._val.tolist(), one._val.tolist()):
            self.assertAlmostEqual(a,b)
        for a, b in zip(marginals[1]._val.tolist(), two._val.tolist()):
            self.assertAlmostEqual(a,b)

    def test_binary_factors_small(self):
        cliques = [self.factor1, self.factor2, self.pair1]
        edges = _compute_edges(cliques)
        tree = CliqueTree(cliques, edges)
        tree.calibrate()
        self.assertTrue(test_convergence(tree.cliqueList))

    def test_binary_factors_3(self):
        cliques = [self.factor1, self.factor2, self.factor3, self.pair1, self.pair2]
        edges = _compute_edges(cliques)
        tree = CliqueTree(cliques, edges)
        tree.calibrate()
        self.assertTrue(test_convergence(tree.cliqueList))

class IntegrationTest(unittest.TestCase):

    def setUp(self):
        words = data_processing.read_PA3Data()
        logistig_model = data_processing.train_logreg_model(words[1:])
        pairwise_model =  data_processing.read_PA3Models_pairwise()
        self.singelton_factors = ocr.compute_singleton_factors([l[0] for l in words[0]],
                                                                logistig_model)
        self.pairwise_factors = ocr.compute_pairwise_factors(len(words[0]),
                                                                pairwise_model)
        self.word = words[0]

    def test_integration_small(self):
        cliques = [self.singelton_factors[0], self.singelton_factors[1], self.pairwise_factors[0]]
        edges = _compute_edges(cliques)
        tree = CliqueTree(cliques, edges)
        tree.calibrate()
        self.assertTrue(test_convergence(tree.cliqueList))

    def test_integration_3(self):
        cliques = [self.singelton_factors[0], self.singelton_factors[1],
                   self.singelton_factors[2], self.pairwise_factors[0],
                   self.pairwise_factors[1]]
        edges = _compute_edges(cliques)
        tree = CliqueTree(cliques, edges)
        tree.calibrate()
        self.assertTrue(test_convergence(tree.cliqueList))

    def test_integration_full(self):
        cliques = self.singelton_factors
        cliques.extend(self.pairwise_factors)
        edges = _compute_edges(cliques)
        tree = CliqueTree(cliques, edges)
        tree.calibrate()
        self.assertTrue(test_convergence(tree.cliqueList))




def test_convergence(cliques):
    for var in sorted({v for f in cliques for v in f.var}):
        marginals = []
        for clique in (c for c in cliques if var in c.var):
            marg = copy.deepcopy(clique)
            for other_var in [v for v in clique.var if not v == var]:
                marg = factor_marginalization(marg, other_var)
            marginals.append(marg)
        for m1, m2 in zip(marginals[:-1], marginals[1:]):
            for a, b in zip(m1._val.flatten().tolist(), m2._val.flatten().tolist()):
                if not almost_equal(a,b):
                    return False
    return True

def almost_equal(a, b):
    return (b - b/10) < a < (b + b/10)

if __name__ == '__main__':
    unittest.main()
