"""
Train a Recurrent Neural Network RNN to classify between "positive" and "negative" examples
of text. What "positive" and "negative" means depends on the problem to solve.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

The training set can be created from a text corpus with the util script

$ python3 generate_random_lines.py ../corpora/corpus_banda.txt subcorpus.txt rnd.txt
$ wc subcorpus.txt rnd.txt
  126665  703435 3515098 subcorpus.txt
  126665  705279 3523796 rnd.txt
"""

import numpy as np
import sys
import os
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping

SEQ_LEN = 20
BATCH_SIZE = 32
PAD_WORD = '[PAD]'


def process_file(file_name):
    lines = open(file_name,'r').readlines()
    return [l.strip() for l in lines]


def shuffle_and_split_training_set(sentences_original, labels_original, percentage_test=10):
    # shuffle at unison
    print('Shuffling sentences')
    tmp_sentences = []
    tmp_next_char = []
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_char.append(labels_original[i])
    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_char[:cut_index], tmp_next_char[cut_index:]

    print("Training set = %d\nTest set = %d" % (len(x_train), len(y_test)))
    return x_train, y_train, x_test, y_test


def pad_and_split_sentences(sentence_list):
    sentences = []
    for s in sentence_list:
        in_words = s.split()
        in_words.extend([PAD_WORD] * (SEQ_LEN - len(in_words)))
        sentences.append(in_words[:SEQ_LEN])
    return sentences


# Data generator for fit and evaluate
def generator(sentence_list, labels_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQ_LEN), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.bool)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word_indices[w]
            y[i] = labels_list[index % len(sentence_list)]
            index = index + 1
        yield x, y


def vectorization(sentence_list, labels_list):
    x = np.zeros((len(sentence_list), SEQ_LEN), dtype=np.int32)
    y = np.zeros((len(sentence_list)), dtype=np.bool)
    for i in range(len(sentence_list)):
        for t, w in enumerate(sentence_list[i]):
            x[i, t] = word_indices[w]
        y[i] = labels_list[i]
    return x, y


def get_model(dropout=0.2):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(len(words), 32))
    model.add(Bidirectional(LSTM(64)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    return model


def confusion_matrix(sentence_list, labels_list, print_errors=False):
    examples, labels = vectorization(sentence_list, labels_list)
    predictions = model.predict(examples, verbose=0)
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i, p in enumerate(predictions):
        positive = p > 0.5
        if positive:
            if labels[i] == 1:
                true_positives = true_positives + 1
            else:
                false_positives = false_positives + 1
                if print_errors:
                    print("FP [%0.4f]: %s" % (p, [w for w in sentences_test[i] if w != PAD_WORD]))
        else:
            if labels[i] == 0:
                true_negatives = true_negatives + 1
            else:
                false_negatives = false_negatives + 1
                if print_errors:
                    print("FN [%0.4f]: %s" % (p, [w for w in sentences_test[i] if w != PAD_WORD]))

    print("TP=", true_positives)
    print("TN=", true_negatives)
    print("FP=", false_positives)
    print("FN=", false_negatives)


if __name__ == "__main__":
    # Argument check
    if len(sys.argv) != 3:
        print('\033[91m' + 'Argument Error!\nUsage: python3 classifier_train.py <positive_examples> <negative_examples>' + '\033[0m')
        exit(1)
    if not os.path.isfile(sys.argv[1]):
        print('\033[91mERROR: ' + sys.argv[1] + ' is not a file!' + '\033[0m')
        exit(1)
    if not os.path.isfile(sys.argv[2]):
        print('\033[91mERROR: ' + sys.argv[2] + ' is not a file!' + '\033[0m')
        exit(1)

    good_ones = process_file(sys.argv[1])
    bad_ones = process_file(sys.argv[2])

    x = pad_and_split_sentences(good_ones + bad_ones)
    y = [1]*len(good_ones) + [0]*len(bad_ones)

    print("Reading files and getting unique words")
    words = set([PAD_WORD])
    for l in x:
        words = words.union(set(l))
    words = sorted(words)
    print('Unique words:', len(words))

    word_indices = dict((c, i) for i, c in enumerate(words))
    indices_word = dict((i, c) for i, c in enumerate(words))

    sentences, labels, sentences_test, labels_test = shuffle_and_split_training_set(x, y)

    train_positive = labels.count(1)
    test_positive = labels_test.count(1)
    print("Training set. Positive %d, Negative %d" % (train_positive, len(labels) - train_positive))
    print("Test set. Positive %d, Negative %d" % (test_positive, len(labels_test)-test_positive))

    print('Build model...')
    model = get_model()

    file_path = "./checkpoints/CLASSIFIER_epoch{epoch:02d}-loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}"
    checkpoint = ModelCheckpoint(file_path, monitor="val_acc", save_best_only=True)
    early_stopping = EarlyStopping(monitor="val_acc", patience=10)
    callbacks_list = [checkpoint, early_stopping]

    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    print(model.summary())

    model.fit_generator(generator(sentences, labels, BATCH_SIZE),
                        steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                        epochs=10,
                        callbacks=callbacks_list,
                        validation_data=generator(sentences_test, labels_test, BATCH_SIZE),
                        validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)

    confusion_matrix(sentences_test, labels_test, True)
