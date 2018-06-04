"""
Example script to generate text from a corpus of text
--By word--

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

Based on
https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py

20 epochs should be enough to get decent results.
Uses data generator to avoid loading all the test set into memory.
Saves the weights and model every epoch.
"""

from __future__ import print_function
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
import numpy as np
import random
import sys
import io
import os

# HARD-CODED parameters, change to experiment different configurations
SEQUENCE_LEN = 10
MIN_WORD_FREQUENCY = 10
STEP = 1
BATCH_SIZE = 32
SIMPLE_MODEL = True


def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []
    tmp_next_words = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_words.append(next_original[i])
    x = tmp_sentences
    y = tmp_next_words
    print('Shuffling finished')

    cut_index = int(len(sentences) * (1.-(percentage_test/100.)))
    x_train = x[:cut_index]
    y_train = y[:cut_index]
    x_test = x[cut_index:]
    y_test = y[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))

    return x_train, y_train, x_test, y_test


# Data generator for fit and evaluate
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN, len(words)), dtype=np.bool)
        y = np.zeros((batch_size, len(words)), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index]):
                x[i, t, word_indices[w]] = 1
            y[i, word_indices[next_word_list[index]]] = 1

            index = index + 1
            if index == len(sentence_list):
                index = 0
        yield x, y


def get_model(simple=True, dropout=0.2):
    print('Build model...')
    model = Sequential()

    if simple:
        model.add(Bidirectional(LSTM(128), input_shape=(SEQUENCE_LEN, len(words))))
        if dropout > 0:
            model.add(Dropout(dropout))
        model.add(Dense(len(words)))
        model.add(Activation('softmax'))
    else:
        model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(SEQUENCE_LEN, len(words))))
        if dropout > 0:
            model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(256, return_sequences=False), input_shape=(SEQUENCE_LEN, len(words))))
        if dropout > 0:
            model.add(Dropout(dropout))
        model.add(Dense(len(words)))
        model.add(Activation('softmax'))
    return model


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)

    # Randomly pick a seed sequence that does not contain words in ignored_words
    start_index = random.randint(0, len(text_in_words) - SEQUENCE_LEN - 1)
    while len(set(text_in_words[start_index: start_index + SEQUENCE_LEN]).intersection(ignored_words)) > 0:
        start_index = random.randint(0, len(text_in_words) - SEQUENCE_LEN - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        sentence = text_in_words[start_index: start_index + SEQUENCE_LEN]
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed: "' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))

        for i in range(50):
            x_pred = np.zeros((1, SEQUENCE_LEN, len(words)))
            for t, word in enumerate(sentence):
                x_pred[0, t, word_indices[word]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(" "+next_word)
        examples_file.write('\n')
    examples_file.write('='*80 + '\n')
    examples_file.flush()

    # print accuracy evaluation
    steps = int(len(sentences_test)/BATCH_SIZE)
    scores = model.evaluate_generator(
        generator(sentences_test, next_words_test, BATCH_SIZE),
        steps=steps,
        max_queue_size=1
    )
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":

    # Argument check
    if len(sys.argv) != 3:
        print('\033[91m' + 'Argument Error!\nUsage: python3 lstm_train.py <path_to_corpus> <examples_txt>' + '\033[0m')
        exit(1)
    if not os.path.isfile(sys.argv[1]):
        print('\033[91mERROR: ' + sys.argv[1] + ' is not a file!' + '\033[0m')
        exit(1)

    corpus = sys.argv[1]
    examples = sys.argv[2]

    if not os.path.isdir('./checkpoints'):
        os.makedirs('./checkpoints')

    with io.open(corpus, encoding='utf-8') as f:
        text = f.read().lower()
    print('Corpus length in characters:', len(text))

    text_in_words = [word for word in text.replace('\n', ' ').split(' ') if word.strip() != '']
    print('Corpus length in words:', len(text_in_words))

    # Calculate word frequency
    word_freq = {}
    for word in text_in_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] < MIN_WORD_FREQUENCY:
            ignored_words.add(k)

    words = set(text_in_words)
    print('Unique words before ignoring:', len(words))
    print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)
    words = sorted(set(words) - ignored_words)
    print('Unique words after ignoring:', len(words))

    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))

    # cut the text in semi-redundant sequences of SEQUENCE_LEN words
    sentences = []
    next_words = []
    ignored = 0
    for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP):
        # Only add the sequences where no word is in ignored_words
        if len(set(text_in_words[i: i+SEQUENCE_LEN+1]).intersection(ignored_words)) == 0:
            sentences.append(text_in_words[i: i + SEQUENCE_LEN])
            next_words.append(text_in_words[i + SEQUENCE_LEN])
        else:
            ignored = ignored + 1
    print('Ignored sequences:', ignored)
    print('Remaining sequences:', len(sentences))

    # x, y, x_test, y_test
    sentences, next_words, sentences_test, next_words_test = shuffle_and_split_training_set(sentences, next_words)

    model = get_model(SIMPLE_MODEL)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    file_path = "./checkpoints/LSTM_LYRICS_words%d_sequence%d_simple%r_minfreq%d_epoch{epoch:02d}_loss{loss:.4f}" % (
        len(words),
        SEQUENCE_LEN,
        SIMPLE_MODEL,
        MIN_WORD_FREQUENCY
    )
    checkpoint = ModelCheckpoint(file_path, save_best_only=False)
    print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
    early_stopping = EarlyStopping(monitor='loss', patience=5)
    callbacks_list = [checkpoint, print_callback, early_stopping]

    examples_file = open(examples, "w")
    model.fit_generator(generator(sentences, next_words, BATCH_SIZE),
                        steps_per_epoch=int(len(sentences)/BATCH_SIZE),
                        epochs=100,
                        max_queue_size=1,
                        callbacks=callbacks_list)
