from deft.mine import SnowCounter


def test_snow_counter():
    """Test stemmer

    Stemmer should be able to keep track of the
    most frequent word mapped to a particular stem.
    """
    snow = SnowCounter()
    words = ['verb', 'verbs', 'verbing', 'verbed', 'noun', 'nouns', 'nouning',
             'nouned', 'verb', 'verb', 'verb', 'verbed', 'noun', 'nouns',
             'nouns', 'nouning', 'nouning']
    for word in words:
        snow.stem(word)
    assert set(snow.counts['verb'].items()) == set([('verb', 4),
                                                    ('verbed', 2),
                                                    ('verbing', 1),
                                                    ('verbs', 1)])
    assert set(snow.counts['noun'].items()) == set([('noun', 2),
                                                    ('nouned', 1),
                                                    ('nouning', 3),
                                                    ('nouns', 3)])
    assert snow.most_frequent('verb') == 'verb'
    assert snow.most_frequent('noun') == 'nouning'
