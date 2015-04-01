"""Contains data processing function, e.g. converting from matlab file to
python object, learning a logistic regression for the image of the letters etc
"""

import os
import scipy.io
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

def read_PA3Data(file='PA3Data.mat'):
    """Reads matlab data file containing the image of letters and returns it
    as a list of list. The image of a letter and its value are stored in a tuble.
    A list of such tubles builds a word and a list of words are returned
    """
    # Set path to location of current python file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # load an preprocess data
    raw_data = scipy.io.loadmat(file)["allWords"]
    words = [[(let[0][0], chr(let[0][1][0][0] + ord("a") - 1))
                for let in word[0]]
                for word in raw_data]
    return words

def train_logreg_model(words):
    """Trains a logistig regression model to make the assigment between the
    image of letters to the actual character.
    Returns the model, model.predict_propab returns the probability for each
    letter
    """
    letters = [l for word in words for l in word]
    X, Y = zip(*[(np.array(x.flat), l) for x,l in letters])
    logreg = linear_model.LogisticRegression()
    logreg.fit(np.array(X), np.array(Y))
    return logreg

def show_letter(letter):
    """Displays the immage of a letter"""
    plt.imshow(letter[0].T, cmap="Greys") #, interpolation='nearest')
    plt.show()

def main():
    words = read_PA3Data()
    mod = train_logreg_model(words[:-10])
    pass

if __name__ == '__main__':
    main()
