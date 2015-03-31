# Converts the data stored in the matlab binary format to python list

import os
import scipy.io

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


def main():
    words = read_PA3Data()
    letters = [l for word in words for l in word]
    pass
    import matplotlib.pyplot as plt
    plt.imshow(letters[0][0].T, cmap="Greys") #, interpolation='nearest')
    plt.show()

if __name__ == '__main__':
    main()
