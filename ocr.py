"""Specalisation of the pgmlib specifig for the ocr task
"""
import numpy as np
from pgmlib.factor import Factor
from inference import map_singelton_ocr
from inference import compute_exact_marginals_ocr_clique_tree


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

def compute_pairwise_factors(letters_in_word, pairwise_model):
    """Computes the n-1 pairwsie factors which assign a probability to the
    apearence of two subsequent letters.
    """
    factors = []
    for var_name in range(1, letters_in_word):
        f = Factor([str(var_name), str(var_name + 1)], [26, 26])
        for i in range(26):
            for j in range(26):
                # add small number such that we do not have zeros
                f.set_val_of_assigment({str(var_name): i + 1, str(var_name+1): j + 1}, pairwise_model[i, j] )
        factors.append(f)
    return factors

def construct_network(images, logistig_model, pairwise_model = None):
    """Constructs OCR network for a word, runs interference and returns result
    images: list of images
    logistig_model: model which applies each letter a probability given the image
    Returns Words as a list of chars
    """
    print('.', end="")
    factors = compute_singleton_factors(images, logistig_model)
    if not pairwise_model is None:
        pairwise = compute_pairwise_factors(len(images), pairwise_model)
        mar = compute_exact_marginals_ocr_clique_tree(factors + pairwise)
        return map_singelton_ocr(mar)
    return map_singelton_ocr(factors)


def main():
    import data_processing
    words = data_processing.read_PA3Data()
    model = data_processing.train_logreg_model(words[1:])
    result = construct_network([l[0] for l in words[0]], model)
    #singelton = compute_singleton_factors([l[0] for l in words[0]], model)
    pass



if __name__ == '__main__':
    main()
