import unittest
import numpy as np

import data_processing
import ocr








class TestOCR(unittest.TestCase):
    def setUp(self):
        self.words = data_processing.read_PA3Data()
        self.model = data_processing.train_logreg_model(self.words)

    def test_compute_singleton_factors(self):
        singeltons = ocr.compute_singleton_factors(
                        [l[0] for l in self.words[0]], self.model)
        self.assertEqual(len(singeltons), 9)
        self.assertAlmostEqual(singeltons[0]._val[0], 0.0064798593031466208)


class IntegrationTest(unittest.TestCase):
    def setUp(self):
        words = data_processing.read_PA3Data()
        self.logistig_model = data_processing.train_logreg_model(words[1:])
        self.word = words[0]

    def test_integration_full(self):
        default = ocr.construct_network([l[0] for l in self.word],
                                                    self.logistig_model, None)
        pairwise = np.ones((26, 26))
        pairwise[tn("t"),tn("o")] = 20
        pairwise[tn("t"),tn("u")] = 10
        pairwise[tn("r"),tn("t")] = 2
        pairwise[tn("n"),tn("g")] = 10
        improved = ocr.construct_network([l[0] for l in self.word],
                                                    self.logistig_model, pairwise)
        self.assertListEqual(improved, ['t','o','r','t','u','r','i','n','g'])


def tn(char):
    return ord(char) - ord("a")

if __name__ == '__main__':
    unittest.main()
