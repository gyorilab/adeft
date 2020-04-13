import os
import json

dashes = [chr(0x2212), chr(0x002d)] + [chr(c) for c in range(0x2010, 0x2016)]

stopwords_min = set(['a', 'an', 'the', 'and', 'or', 'of', 'with', 'at',
                     'from', 'into', 'to', 'for', 'on', 'by', 'be', 'been',
                     'am', 'is', 'are', 'was', 'were', 'in', 'that', 'as'])

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'stopwords.json'), 'r') as f:
    english_stopwords = json.load(f)

greek_alphabet = {
    u'\u0391': 'Alpha',
    u'\u0392': 'Beta',
    u'\u0393': 'Gamma',
    u'\u0394': 'Delta',
    u'\u0395': 'Epsilon',
    u'\u0396': 'Zeta',
    u'\u0397': 'Eta',
    u'\u0398': 'Theta',
    u'\u0399': 'Iota',
    u'\u039A': 'Kappa',
    u'\u039B': 'Lamda',
    u'\u039C': 'Mu',
    u'\u039D': 'Nu',
    u'\u039E': 'Xi',
    u'\u039F': 'Omicron',
    u'\u03A0': 'Pi',
    u'\u03A1': 'Rho',
    u'\u03A3': 'Sigma',
    u'\u03A4': 'Tau',
    u'\u03A5': 'Upsilon',
    u'\u03A6': 'Phi',
    u'\u03A7': 'Chi',
    u'\u03A8': 'Psi',
    u'\u03A9': 'Omega',
    u'\u03B1': 'alpha',
    u'\u03B2': 'beta',
    u'\u03B3': 'gamma',
    u'\u03B4': 'delta',
    u'\u03B5': 'epsilon',
    u'\u03B6': 'zeta',
    u'\u03B7': 'eta',
    u'\u03B8': 'theta',
    u'\u03B9': 'iota',
    u'\u03BA': 'kappa',
    u'\u03BB': 'lamda',
    u'\u03BC': 'mu',
    u'\u03BD': 'nu',
    u'\u03BE': 'xi',
    u'\u03BF': 'omicron',
    u'\u03C0': 'pi',
    u'\u03C1': 'rho',
    u'\u03C3': 'sigma',
    u'\u03C4': 'tau',
    u'\u03C5': 'upsilon',
    u'\u03C6': 'phi',
    u'\u03C7': 'chi',
    u'\u03C8': 'psi',
    u'\u03C9': 'omega',
}

greek_to_latin = {
   'alpha': 'a',
   'Alpha': 'A',
   'beta': 'b',
   'Beta': 'B',
   'gamma': 'c',
   'Gamma': 'C',
   'delta': 'd',
   'Delta': 'D',
}
