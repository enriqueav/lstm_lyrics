"""
Utility script to generate the training set for the text classifier.

Usage: python3 generate_random_lines.py <path_to_corpus> <corpus_subset_txt> <random_txt>

Creates two files:
    -<corpus_subset_txt>: is a subset of the original corpus, containing only the lines without
                          any of the ignored words
    -<random_txt>: is a file containing random lines, choosing random words
                   from the vocabulary of the corpus,
                   and also using the same probabilities of word usage and
                   quantity of words in each line.

As a result, both files have exactly the same number of lines (126665 in this example),
and roughly the same number of total words (703435 vs 705279 in the example):

$ python3 generate_random_lines.py ../corpora/corpus_banda.txt banda_subset.txt random_banda.txt
$ wc banda_subset.txt random_banda.txt
  126665  703435 3515098 banda_subset.txt
  126665  705279 3523796 random_banda.txt
"""

import numpy as np
import sys
import os
from progressbar import ProgressBar, Percentage, Bar, ETA, FileTransferSpeed

# Parameter: change to experiment different configurations
MIN_WORD_FREQUENCY = 10


def get_count_and_freqs(lines):
    frequencies = {}
    count = 0
    for l in lines:
        in_words = l.split()
        count = count + len(in_words)
        for w in in_words:
            frequencies[w] = frequencies.get(w, 0) + 1

    return count, frequencies


def get_words_and_probabilities(word_freq):
    words = []
    probs = np.zeros(len(word_freq))
    index = 0
    for k, v in word_freq.items():
        words.append(k)
        probs[index] = v
        index = index + 1
    return words, probs


def get_sizes_and_probabilites(lines):
    frequencies = {}
    for l in lines:
        in_words = l.split()
        wc = len(in_words)
        frequencies[wc] = frequencies.get(wc, 0) + 1

    line_count = len(lines)
    values = []
    probabilities = []
    total_prob = 0
    for k, v in sorted(frequencies.items()):
        values.append(k)
        probabilities.append(v / (line_count+0.0))
        total_prob = total_prob + (v / (line_count+0.0))

    dif = np.float32(1 - total_prob)
    values.append(1)
    probabilities.append(dif)

    return values, probabilities


if __name__ == "__main__":
    # Argument check
    if len(sys.argv) != 4:
        print('\033[91m' + 'Argument Error!\nUsage: python3 generate_classifier_set.py '
                           '<path_to_corpus> <corpus_subset_txt> <random_txt>' + '\033[0m')
        exit(1)
    if not os.path.isfile(sys.argv[1]):
        print('\033[91mERROR: ' + sys.argv[1] + ' is not a file!' + '\033[0m')
        exit(1)

    corpus = sys.argv[1]
    subcorpus_txt = sys.argv[2]
    random_txt = sys.argv[3]

    lines = open(corpus, encoding='utf-8').readlines()
    lines = [l.strip() for l in lines if l.strip() != '']

    word_count, word_freq = get_count_and_freqs(lines)

    print('Corpus size in lines:', len(lines))
    # print('Corpus size in words:', word_count)
    print('Unique words before ignoring:', len(set(word_freq)))
    print('Ignoring words with frequency <', MIN_WORD_FREQUENCY)

    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] < MIN_WORD_FREQUENCY:
            ignored_words.add(k)

    for word in ignored_words:
        del(word_freq[word])
    print('Unique words after ignoring:', len(word_freq))

    lines = [l for l in lines if len(set(l.split()).intersection(ignored_words)) == 0]
    print('Size in lines after ignoring %d' % (len(lines)))

    word_count, word_freq = get_count_and_freqs(lines)

    for k, v in word_freq.items():
        word_freq[k] = word_freq[k] / (word_count + 0.0)

    words, word_probabilities = get_words_and_probabilities(word_freq)
    total_prob = np.sum(word_probabilities)
    # print("Total probability before adjust " + str(total_prob))
    # print("Going to adjust %0.20f" % np.float32((1.0 - total_prob)))
    # Adjusting probabilities to leave it at 1 (required by np.random.choice)
    if np.float32((1.0 - total_prob)) != 0:
        word_freq['<<ADJ>>'] = np.float32(1.0 - total_prob)

    words, word_probabilities = get_words_and_probabilities(word_freq)

    total_prob = np.sum(word_probabilities)
    # print("Total probability after adjust " + str(total_prob))
    assert total_prob == 1, "The total probability of Words is not 1. Something is wrong!"

    # get sizes probabilities
    sizes, sizes_probabilities = get_sizes_and_probabilites(lines)
    assert np.sum(sizes_probabilities) == 1, "The total probability of Sizes is not 1. " \
                                             "Something is wrong!"

    print("\nGoing to create subset of the corpus, %d lines to file '%s'" %
          (len(lines), subcorpus_txt))
    subcorpus_file = open(subcorpus_txt, 'w')
    for l in lines:
        subcorpus_file.write(l + '\n')
        subcorpus_file.flush()
    subcorpus_file.close()

    print("\nGoing to print %d random lines to file '%s'" % (len(lines), random_txt))
    widgets = ['Progress: ', Percentage(), ' ', Bar(marker='=', left='[', right=']'),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets, maxval=len(lines))
    pbar.start()

    result_file = open(random_txt, 'w')
    done = 0
    for i in range(len(lines)):
        length = np.random.choice(sizes, p=sizes_probabilities)
        chosen_words = np.random.choice(words, length, p=word_probabilities)
        result_file.write(' '.join(chosen_words) + '\n')
        result_file.flush()
        done = done+1
        pbar.update(done)
    result_file.close()
    pbar.finish()
    print("DONE!")
