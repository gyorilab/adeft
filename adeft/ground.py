import os
import csv
import gzip
from copy import deepcopy
from collections import defaultdict

from adeft.locations import RESOURCES_PATH
from adeft.nlp.stem import greek_aware_stem
from adeft.nlp.preprocess import dashes, expand_dashes
from adeft.util import SearchTrie, get_candidate
from adeft.nlp.compare_strings import text_similarity


def load_default_grounding_terms():
    grounding_terms = {}
    grounding_map = defaultdict(list)
    with gzip.open(os.path.join(RESOURCES_PATH,
                                'grounding_terms.tsv.gz'), 'rt') as f:
        reader = csv.reader(f, delimiter='\t')
        for index, row in enumerate(reader):
            entry = {'name_space': row[2],
                     'id': row[3],
                     'canonical_name': row[4],
                     'type': row[5],
                     'raw_text': row[1]}
            grounding_terms[index] = entry
            grounding_map[row[1]].append(index)
    return grounding_terms, grounding_map


class AdeftGrounder(object):
    def __init__(self, groundings=None):
        if groundings is None:
            grounding_terms, grounding_map = load_default_grounding_terms()
        self.grounding_terms = grounding_terms
        self._trie = SearchTrie(grounding_map, expander=expand_dashes,
                                token_map=greek_aware_stem)
        self.type_priority = {'assertion': 0,
                              'name': 1,
                              'synonym': 2,
                              'previous': 3}

    def ground(self, text):
        results = []
        expansions = expand_dashes(text)
        for expansion in expansions:
            tokens, longform_map = get_candidate(expansion)
            processed_tokens = [greek_aware_stem(token) for token in tokens]
            grounding_keys, match_text = self._trie.search(processed_tokens)
            if grounding_keys is None:
                continue
            entity_tokens, _ = get_candidate(match_text)
            if entity_tokens == processed_tokens[-len(entity_tokens):]:
                longform_text = longform_map[len(entity_tokens)]
                for grounding_key in grounding_keys:
                    entry = deepcopy(self.grounding_terms[grounding_key])
                    entry['longform_text'] = longform_text
                    results.append(entry)
        result_dict = {}
        for result in results:
            raw_text = result['raw_text']
            grounding = (result['name_space'], result['id'])
            longform_text = result['longform_text']
            score = (text_similarity(longform_text, raw_text),
                     3 - self.type_priority[result['type']])
            if grounding not in result_dict or \
               score > result_dict[grounding]['score']:
                result_dict[grounding] = result
                result_dict[grounding]['score'] = score
        out = [result for result in result_dict.values()
               if result['score'][0] > 0]
        return sorted(out, key=lambda x: (-x['score'][0], -x['score'][1]))


def normalize(s):
    s = ''.join(s.split())
    s = ''.join([char for char in s if char not in dashes])
    return s.lower()
