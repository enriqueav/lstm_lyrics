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
python3 lstm_train.py corpora/corpus_banda.txt examples.txt vocabulary.txt
``` 

Where 
- **corpora/corpus_banda.txt**: points to the corpus you want to train from
- **examples.txt**: is the file where the example text is going to be written after every epoch
- **vocabulary.txt**: is a file where all the words used by the network is written, one per line. 
It is used by generate.py.


Or the version using [Word Embedding (words to vectors)](https://medium.com/@monocasero/update-automatic-song-lyrics-creator-with-word-embeddings-e30de94db8d1):

```bash
python3 lstm_train_embedding.py corpora/corpus_reggeaton.txt examples_reggeaton.txt
``` 

Where 
- **corpora/corpus_reggeaton.txt**: points to the corpus you want to train from
- **examples_reggeaton.txt**: is the file where the example text is going to be written after every epoch

### To generate text from a trained model:

The model and weights will be saved in the directory `./checkpoints/`.
We can use these files to generate text from a given seed.

The script `generate.py` is used for this. It used argparse to manage the arguments.

```bash
$ python generate.py -h
Using TensorFlow backend.
usage: generate.py [-h] [-v VOCABULARY] [-n NETWORK] [-c CORPUS] [-s SEED]
                   [-l SEQUENCE_LENGTH] [-d DIVERSITY] [-q QUANTITY]

Generate lyrics using the weights of a trained network.

optional arguments:
  -h, --help            show this help message and exit
  -v VOCABULARY, --vocabulary VOCABULARY
                        The path of vocabulary used by the network.
  -n NETWORK, --network NETWORK
                        The path of the trained network.
  -c CORPUS, --corpus CORPUS
                        The path of the original corpus used to train the
                        network. Only necessary if the seed is set to random
  -s SEED, --seed SEED  The seed used to generate the text. All the words
                        should be part of the vocabulary. Only the last
                        SEQUENCE_LENGTH words are considered
  -l SEQUENCE_LENGTH, --sequence_length SEQUENCE_LENGTH
                        The length of the sequence used for the training.
                        Default is 10
  -d DIVERSITY, --diversity DIVERSITY
                        The value of diversity. Usually a number between 0.1
                        and 2 Default is 0.5
  -q QUANTITY, --quantity QUANTITY
                        Quantity of words to generate. Default is 50
```

At least -v and -n are needed to generate text.

For instance

```bash
$ python generate.py -v reggeaton_vocabulary.txt -n checkpoints/LSTM_LYRICS-epoch009-words12952-sequence10-minfreq10-loss2.1511-acc0.6280-val_loss2.9192-val_acc0.5508 -s "perrea mami perrea dale duro" -q 60 -d 0.7
Using TensorFlow backend.

Summary of the Network: 
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
bidirectional_1 (Bidirection (None, 256)               13394944  
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 12952)             3328664   
_________________________________________________________________
activation_1 (Activation)    (None, 12952)             0         
=================================================================
Total params: 16,723,608
Trainable params: 16,723,608
Non-trainable params: 0
_________________________________________________________________

Validating that all the words in the seed are part of the vocabulary: 
perrea ✓ in vocabulary
mami ✓ in vocabulary
perrea ✓ in vocabulary
dale ✓ in vocabulary
duro ✓ in vocabulary

Seed is correct.

----- Generating text
----- Diversity:0.7
----- Generating with seed:
"perrea mami perrea dale duro perrea mami perrea dale duro
perrea mami perrea dale duro perrea mami perrea dale duro
 
 uh 
 que no se siente bien 
 que tengo la pista en mi cuerpo 
 y no se porque tú eres la misma hecho 
 que no te tengo a mi a mi me encanta tu culpa 
 si tu me dices tu me llamas si te sientes solita 
 y es que vuelvo a ver 
 que
```

For now, the generated text is only printed to the console.

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
