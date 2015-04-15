import unittest
import numpy as np
from operator import add


from inference import map_singelton
from inference import CliqueTree
import factor
from factor import Factor



class TestMap(unittest.TestCase):
    def setUp(self):
        self.factor1 = Factor(["1"], [10], np.array(range(10)))

    def test_map_singelton(self):
        self.assertEqual(map_singelton([self.factor1]), [10])


class TestCliqueTree(unittest.TestCase):
    def setUp(self):
        # two singelton and one pair
        self.factor1 = Factor(["1"], [2], np.array([0.2, 0.8]))
        self.factor2 = Factor(["2"], [2], np.array([0.1, 0.9]))
        self.pair1 = Factor(["1", "2"], [2, 2], np.array([0.99, 0.01, 0.01, 0.01]))
        edges = np.diag([1]*3)
        edges[0, 2] = 1; edges[2, 0] = 1; edges[1, 2] = 1; edges[2, 1] = 1;
        self.tree1 = CliqueTree([self.factor1, self.factor2, self.pair1], edges)

    def test_get_next_cliques(self):
        messages = [[None for _ in range(3)] for _ in range(3)]
        self.assertEqual([0, 2], self.tree1._get_netxt_cliques(messages))
        messages[0][2] = 1
        self.assertEqual([1, 2], self.tree1._get_netxt_cliques(messages))
        messages[1][2] = 1
        self.assertEqual([2, 0], self.tree1._get_netxt_cliques(messages))
        messages[2][0] = 1
        self.assertEqual([2, 1], self.tree1._get_netxt_cliques(messages))

    def test_calibrate(self):
        self.tree1.calibrate(False)
        comined = factor.factor_product(self.factor2,
                                factor.factor_product(self.factor1, self.pair1))
        one = factor.factor_marginalization(comined, "2")
        two = factor.factor_marginalization(comined, "1")
        for a, b in zip(self.tree1.cliqueList[0]._val.tolist(), one._val.tolist()):
            self.assertAlmostEqual(a,b)
        for a, b in zip(self.tree1.cliqueList[1]._val.tolist(), two._val.tolist()):
            self.assertAlmostEqual(a,b)
        for a, b in zip(self.tree1.cliqueList[2]._val.flatten().tolist(),
                        comined._val.flatten().tolist()):
            self.assertAlmostEqual(a,b)

    def test_calibrate_max_sum(self):
        self.tree1.calibrate(True)
        res = map_singelton(self.tree1.cliqueList[:-1])
        comined = factor.factor_product(self.factor2,
                                factor.factor_product(self.factor1, self.pair1, add), add)
        one = factor.factor_marginalization(comined, "2", max)
        two = factor.factor_marginalization(comined, "1", max)
        for a, b in zip(self.tree1.cliqueList[0]._val.tolist(), one._val.tolist()):
            self.assertAlmostEqual(a,b)
        for a, b in zip(self.tree1.cliqueList[1]._val.tolist(), two._val.tolist()):
            self.assertAlmostEqual(a,b)
        for a, b in zip(self.tree1.cliqueList[2]._val.flatten().tolist(),
                        comined._val.flatten().tolist()):
            self.assertAlmostEqual(a,b)

if __name__ == '__main__':
    unittest.main()
