from deft.disambiguate import DeftDisambiguator, load_disambiguator


def test_load_disambiguator():
    dd_test = load_disambiguator('TEST')
    assert dd_test.shortform == 'IR'
    assert hasattr(dd_test, 'classifier')
    assert hasattr(dd_test, 'recognizer')


def test_disambiguate():
    pass
