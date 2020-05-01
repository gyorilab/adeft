import os
import json

stopwords_min = set(['a', 'an', 'the', 'and', 'or', 'of', 'with', 'at',
                     'from', 'into', 'to', 'for', 'on', 'by', 'be', 'been',
                     'am', 'is', 'are', 'was', 'were', 'in', 'that', 'as'])

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'stopwords.json'), 'r') as f:
    english_stopwords = json.load(f)
