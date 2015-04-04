import data_processing
import ocr


def score(list_a, list_b):
    """Computes the score"""
    if not len(list_a) == len(list_b): raise Exception("Dimension mismatch")
    return sum(ea == eb for ea, eb in zip(list_a, list_b)) / len(list_a)

def main():
    # Load Data and split in test and train set
    words = data_processing.read_PA3Data()
    train_words = words[:85]
    test_words = words[85:]

    # Train logistig model
    model = data_processing.train_logreg_model(train_words)

    # Predict test words
    precition = [ocr.construct_network([l[0] for l in word], model) for word in test_words]

    # Calculate Scores
    print("Score for words: ",score(precition, [[l[1] for l in word] for word in test_words]))
    print("Score for letters: ", score([l for word in precition for l in word], [l[1] for word in test_words for l in word]))

if __name__ == '__main__':
    main()
