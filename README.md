# lstm_lyrics
LSTM text generation by word. Used to generate lyrics from a corpus of a music genre.

[https://medium.com/@monocasero/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb](https://medium.com/@monocasero/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb)

And the update: working with word embeddings

[https://medium.com/@monocasero/update-automatic-song-lyrics-creator-with-word-embeddings-e30de94db8d1](https://medium.com/@monocasero/update-automatic-song-lyrics-creator-with-word-embeddings-e30de94db8d1)


### How to install

The first thing is to clone this same repo and `cd` to it:

```bash
git clone https://github.com/enriqueav/lstm_lyrics.git
cd lstm_lyrics
```

If you want to test an experimental branch:

```bash
git clone https://github.com/enriqueav/lstm_lyrics.git -b <experimental_branch>
```

### Install dependencies

If necessary, install virtualenv
```bash
pip install virtualenv
```

Then create an environment and install the dependencies

```bash
virtualenv env --python=python3.6
source env/bin/activate
pip install -r requirements.txt
```

### To start the training:

To train the one-hot encoded version:

```bash
python3 lstm_train.py corpora/corpus_banda.txt examples.txt
``` 

or the version using [Word Embedding (words to vectors)](https://medium.com/@monocasero/update-automatic-song-lyrics-creator-with-word-embeddings-e30de94db8d1):

```bash
python3 lstm_train_embedding.py corpora/corpus_reggeaton.txt examples_reggeaton.txt
``` 

### Text classifier 

The objective is to create a neural network to classify real text taken from a corpus vs randomly generated text.
The idea is to increase the quality of the generated lyrics pre-filtering the lines that look a lot like randomly chosen words.

#### To create the training set 

```bash
python3 utils/generate_classifier_set.py corpora/corpus_banda.txt banda_subset.txt random_banda.txt
```

### How to contribute 

Be sure to check that your changes did not include any [flake8](http://flake8.pycqa.org/en/latest/) error:

```bash
$ flake8
$ 
```
