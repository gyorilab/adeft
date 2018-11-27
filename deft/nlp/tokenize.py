from nltk.tokenize import regexp_tokenize


def word_tokenize(text):
    return regexp_tokenize(text, r'\w+|[/\(\)\-\\]|[^\s\w/\(\)\-\\]')
