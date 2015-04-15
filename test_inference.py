import unittest
import numpy as np
from operator import add



from inference import map_singelton_ocr
from inference import compute_exact_marginals_ocr_clique_tree

from pgmlib.factor import Factor
from pgmlib import factor


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

    def test_c_e_m_o_c_t(self):
        marginals = compute_exact_marginals_ocr_clique_tree(
                                    [self.factor1, self.factor2], [self.pair1])
        comined = factor.factor_product(self.factor2,
                    factor.factor_product(self.factor1, self.pair1, add), add)
        one = factor.factor_marginalization(comined, "2", max)
        two = factor.factor_marginalization(comined, "1", max)
        for a, b in zip(marginals[0]._val.tolist(), one._val.tolist()):
            self.assertAlmostEqual(a,b)
        for a, b in zip(marginals[1]._val.tolist(), two._val.tolist()):
            self.assertAlmostEqual(a,b)




if __name__ == '__main__':
    unittest.main()
