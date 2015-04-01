import unittest
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

if __name__ == '__main__':
    unittest.main()
