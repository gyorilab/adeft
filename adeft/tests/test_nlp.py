import pytest
from adeft.nlp import WatchfulStemmer, word_tokenize, word_detokenize


def test_tokenize_untokenize():
    """Test the tokenizer and untokenizer"""
    text1 = 'Test the tokenizer and untokenizer.'
    text2 = 'Test***the    tokenizer and---untokenizer..'
    text3 = 'Test'
    text4 = 'Test the'
    text5 = ''

    result1 = [('Test', (0, 3)), ('the', (5, 7)), ('tokenizer', (9, 17)),
               ('and', (19, 21)), ('untokenizer', (23, 33)),
               ('.', (34, 34))]
    result2 = [('Test', (0, 3)), ('*', (4, 4)), ('*', (5, 5)), ('*', (6, 6)),
               ('the', (7, 9)), ('tokenizer', (14, 22)), ('and', (24, 26)),
               ('-', (27, 27)), ('-', (28, 28)), ('-', (29, 29)),
               ('untokenizer', (30, 40)), ('.', (41, 41)), ('.', (42, 42))]
    result3 = [('Test', (0, 3))]
    result4 = [('Test', (0, 3)), ('the', (5, 7))]
    result5 = []

    for text, result in zip([text1, text2, text3, text4, text5],
                            [result1, result2, result3, result4, result5]):
        assert word_tokenize(text) == result

    for text, result in zip([text1, text2, text3, text4, text5],
                            [result1, result2, result3, result4, result5]):
        assert word_detokenize(result) == text


def test_watchful_stemmer():
    """Test the watchful stemmer"""
    stemmer = WatchfulStemmer()
    words = (['verb']*3 + ['verbs']*5 + ['verbed']*2 + ['verbing'] +
             ['verbings']*8 + ['verbification']*5 + ['verbifications']*3 +
             ['verbize']*7 + ['verbization']*2 + ['verbizations']*4)
    for word in words:
        stemmer.stem(word)
    assert stemmer.most_frequent('verb') == 'verbings'
    assert stemmer.most_frequent('verbif') == 'verbification'
    assert stemmer.most_frequent('verbiz') == 'verbize'
    assert stemmer.most_frequent('verbize') == 'verbizations'

    assert stemmer.counts['verb']['verbs'] == 5
    assert stemmer.counts['verb']['verbing'] == 1
    assert stemmer.counts['verbize']['verbization'] == 2
    assert stemmer.counts['verbif']['verbifications'] == 3

    # raises value error if stem has not been observed
    with pytest.raises(ValueError):
        stemmer.most_frequent('ver')
