# lstm_lyrics
LSTM text generation by word. Used to generate lyrics from a corpus of a music genre.

[https://medium.com/@monocasero/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb](https://medium.com/@monocasero/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb)

To start the training:

```bash
git clone https://github.com/enriqueav/lstm_lyrics.git
cd lstm_lyrics
```
And then, to train the one-hot encoded version:

```bash
python3 lstm_train.py corpora/corpus_banda.txt examples.txt
``` 

or the version using Word Embedding (words to vectors):

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