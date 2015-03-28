import unittest

import factor
from factor import Factor

class TestFactorClass(unittest.TestCase):
    def setUp(self):
        self.factor1 = Factor(["1", "2", "3"], [1, 2, 3], list(range(6)))
        self.factor2 = Factor(["1", "2"], [3, 4], list(range(12,24)))
        self.factor3 = Factor(["1", "2", "3", "4", "5"], [7, 3, 4, 7, 2])

    def test_assigment_to_index(self):
        self.assertEqual(self.factor1._assigment_to_index([1, 1, 1]), 0)
        self.assertEqual(self.factor1._assigment_to_index([1, 2, 3]), 5)
        self.assertEqual(self.factor1._assigment_to_index([1, 2, 1]), 1)
        self.assertEqual(self.factor1._assigment_to_index([1, 2, 2]), 3)
        self.assertEqual(self.factor2._assigment_to_index([3, 3]), 8)
        self.assertEqual(self.factor2._assigment_to_index([3, 4]), 11)

    def test_index_to_assigment(self):
        index1 = self.factor1._assigment_to_index([1, 2, 3])
        self.assertEqual(self.factor1._index_to_assigment(index1), [1,2,3])
        index2 = self.factor1._assigment_to_index([1, 1, 2])
        self.assertEqual(self.factor1._index_to_assigment(index2), [1,1,2])
        index3 = self.factor2._assigment_to_index([3, 3])
        self.assertEqual(self.factor2._index_to_assigment(index3), [3,3])
        index4 = self.factor3._assigment_to_index([2,1,4,5,1])
        self.assertEqual(self.factor3._index_to_assigment(index4), [2,1,4,5,1])

    def test_get_val_of_assigment(self):
        self.assertEqual(
            self.factor1.get_val_of_assigment({"1":1, "2": 2, "3":2}), 3)
        self.assertEqual(
            self.factor1.get_val_of_assigment({"1":1, "2": 1, "3":3}), 4)

    def test_set_val_of_assigment(self):
        self.factor1.set_val_of_assigment({"1":1, "2":1, "3":2} , 10)
        self.factor1.set_val_of_assigment({"1":1, "2":2, "3":3} , 10)
        self.assertEqual(self.factor1._val,  [0, 1, 10, 3, 4, 10])

    def test_get_all_assigments_d(self):
        temp = self.factor1.get_all_assigments_d()
        self.assertEqual( len(temp), 6)

class TestFactorModule(unittest.TestCase):
    def setUp(self):
        self.factorA = Factor(["A", "B", "C"], [2, 3, 2], range(12))
        self.factorB = Factor(["D", "E", "B"], [2, 2, 3], range(12, 24))

    def test_factor_product(self):
        prod = factor.factor_product(self.factorA, self.factorB)
        self.assertEqual( (self.factorA.get_val_of_assigment({"A":1, "B":2, "C":2}) *
        self.factorB.get_val_of_assigment({"D":2, "E":1, "B":2})),
        prod.get_val_of_assigment({"A":1, "B":2, "C":2, "D":2, "E":1}))

if __name__ == '__main__':
    unittest.main()
