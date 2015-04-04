"""Specalisation of the pgmlib specifig for the ocr task
"""
import numpy as np
from pgmlib.factor import Factor
from pgmlib.inference import map_simple


LETTERES_IN_ALPHABET = 26

def compute_singleton_factors(images, model):
    """Creats a factor for each image provided
    images: list of images
    model: model which applies each letter a probability given the image
    Returns a list of factors,
        naming converntion for variables: the ith letter gets the name "i"
    """
    factors = []
    for index, image in enumerate(images):
        # default value = 0.01, to guarantee that no value is zero!
        #(nothing is impossible)
        f = Factor([str(index + 1)], [LETTERES_IN_ALPHABET], np.repeat(0.01, 26))
        for let, prop in np.vstack((model.classes_, model.predict_proba(image.flat)[0])).T:
            f._val[ord(let) - ord("a")] = prop
        factors.append(f)
    return factors


def construct_network(images, model):
    """Constructs OCR network for a word, runs interference and returns result
    images: list of images
    model: model which applies each letter a probability given the image
    Returns Words as a list of chars
    """
    factors = compute_singleton_factors(images, model)
    # TODO: compute pairwise factores etc
    return [c[1] for c in  map_simple(factors)]


def main():
    import data_processing
    words = data_processing.read_PA3Data()
    model = data_processing.train_logreg_model(words[1:])
    result = construct_network([l[0] for l in words[0]], model)
    #singelton = compute_singleton_factors([l[0] for l in words[0]], model)
    pass



if __name__ == '__main__':
    main()
