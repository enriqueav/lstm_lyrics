"""
Script to generate text from an already trained network (with lstm_train.py)
--By word--

It is necessary to at least provide the trained model and the vocabulary file
(generated also by lstm_train.py).
"""


import argparse
import numpy as np
import re
from keras.models import load_model


def validate_seed(vocabulary, seed):
    """Validate that all the words in the seed are part of the vocabulary"""
    print("\nValidating that all the words in the seed are part of the vocabulary: ")
    seed_words = seed.split(" ")
    valid = True
    for w in seed_words:
        print(w, end="")
        if w in vocabulary:
            print(" ✓ in vocabulary")
        else:
            print(" ✗ NOT in vocabulary")
            valid = False
    return valid


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_text(model, indices_word, word_indices, seed,
                  sequence_length, diversity, quantity):
    """
    Similar to lstm_train::on_epoch_end
    Used to generate text using a trained model

    :param model: the trained Keras model (with model.load)
    :param indices_word: a dictionary pointing to the words
    :param seed: a string to be used as seed (already validated and padded)
    :param sequence_length: how many words are given to the model to generate
    :param diversity: is the "temperature" of the sample function (usually between 0.1 and 2)
    :param quantity: quantity of words to generate
    :return: Nothing, for now only writes the text to console
    """
    sentence = seed.split(" ")
    print("----- Generating text")
    print('----- Diversity:' + str(diversity))
    print('----- Generating with seed:\n"' + seed)

    print(seed)
    for i in range(quantity):
        x_pred = np.zeros((1, sequence_length, len(vocabulary)))
        for t, word in enumerate(sentence):
            x_pred[0, t, word_indices[word]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_word = indices_word[next_index]

        sentence = sentence[1:]
        sentence.append(next_word)

        print(" "+next_word, end="")
    print("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate lyrics using the weights of a trained network.\n"
    )
    parser.add_argument("-v", "--vocabulary",
                        help="The path of vocabulary used by the network.",)
    parser.add_argument("-n", "--network",
                        help="The path of the trained network.",)
    parser.add_argument("-s", "--seed",
                        help="The seed used to generate the text. All the words should be part\n"
                             "of the vocabulary. Only the last SEQUENCE_LENGTH "
                             "words are considered",)
    parser.add_argument("-l", "--sequence_length",
                        help="The length of the sequence used for the training. Default is 10",
                        type=int,
                        default=10)
    parser.add_argument("-d", "--diversity",
                        help="The value of diversity. Usually a number between 0.1 and 2\n"
                             "Default is 0.5",
                        type=float,)
    parser.add_argument("-q", "--quantity",
                        help="Quantity of words to generate. Default is 50",
                        type=int,
                        default=50)
    args = parser.parse_args()

    vocabulary_file = args.vocabulary
    model_file = args.network
    seed = args.seed
    sequence_length = args.sequence_length
    diversity = args.diversity
    quantity = args.quantity

    if not vocabulary_file or not model_file:
        print('\033[91mERROR: At least --vocabulary and --network are needed\033[0m')
        exit(0)

    model = load_model(model_file)
    print("\nSummary of the Network: ")
    model.summary()

    vocabulary = open(vocabulary_file, "r").readlines()
    # remove the \n at the end of the word, except for the \n word itself
    vocabulary = [re.sub(r'(\S+)\s+', r'\1', w) for w in vocabulary]
    vocabulary = sorted(set(vocabulary))

    word_indices = dict((c, i) for i, c in enumerate(vocabulary))
    indices_word = dict((i, c) for i, c in enumerate(vocabulary))

    if validate_seed(vocabulary, seed):
        print("\nSeed is correct.\n")
        # repeat the seed in case is not long enough, and take only the last elements
        seed = " ".join((((seed+" ")*sequence_length)+seed).split(" ")[-sequence_length:])
        generate_text(
            model, indices_word, word_indices, seed, sequence_length, diversity, quantity
        )
    else:
        print('\033[91mERROR: Please fix the seed string\033[0m')
        exit(0)
