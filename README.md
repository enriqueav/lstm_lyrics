# lstm_lyrics
LSTM text generation by word. Used to generate lyrics from a corpus of a music genre.

[https://medium.com/@monocasero/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb](https://medium.com/@monocasero/word-level-lstm-text-generator-creating-automatic-song-lyrics-with-neural-networks-b8a1617104fb)

To start the training:

```bash
git clone https://github.com/enriqueav/lstm_lyrics.git
cd lstm_lyrics
python3 lstm_train.py corpora/corpus_banda.txt nothing.txt
```