from adeft.modeling.label import AdeftLabeler


# content for single shortform corpus building
longforms = {'integrated network and dynamical reasoning assembler':
             'our indra',
             'indonesian debt restructuring agency': 'other indra'}

text1 = ('The Integrated Network and Dynamical Reasoning Assembler'
         ' (INDRA) is an automated model assembly system interfacing'
         ' with NLP systems and databases to collect knowledge, and'
         ' through a process of assembly, produce causal graphs and'
         ' dynamical models. INDRA draws on natural language'
         ' processing systems and structured databases to collect'
         ' mechanistic and causal assertions, represents them in'
         ' standardized form (INDRA statements), and assembles them'
         ' into various modeling formalisms including causal graphs'
         ' and dynamical models.')

result1 = ('The INDRA'
           ' is an automated model assembly system interfacing'
           ' with NLP systems and databases to collect knowledge, and'
           ' through a process of assembly, produce causal graphs and'
           ' dynamical models. INDRA draws on natural language'
           ' processing systems and structured databases to collect'
           ' mechanistic and causal assertions, represents them in'
           ' standardized form (INDRA statements), and assembles them'
           ' into various modeling formalisms including causal graphs'
           ' and dynamical models.')


labels1 = set(['our indra'])


text2 = ('The Integrated Network and Dynamical Reasoning Assembler'
         ' (INDRA) is an automated model assembly system interfacing'
         ' with NLP systems and databases to collect knowledge, and'
         ' through a process of assembly, produce causal graphs and'
         ' dynamical models. The Indonesian Debt Restructuring Agency'
         ' (INDRA) shares the same acronym. Previously, without this'
         ' sentence the entire text would have been wiped away.'
         ' This has been fixed now.')

result2 = ('The INDRA'
           ' is an automated model assembly system interfacing'
           ' with NLP systems and databases to collect knowledge, and'
           ' through a process of assembly, produce causal graphs and'
           ' dynamical models. The INDRA'
           ' shares the same acronym. Previously, without this'
           ' sentence the entire text would have been wiped away.'
           ' This has been fixed now.')

labels2 = set(['our indra', 'other indra'])

text3 = ('In this sentence, (INDRA) appears but it is not preceded by a'
         ' recognized longform. Does it refer to the indonesian debt'
         ' restructuring agency (INDRA)? It\'s hard to say.')

result3 = ('In this sentence, INDRA appears but it is not preceded by a'
           ' recognized longform. Does it refer to the INDRA ?'
           ' It\'s hard to say.')

labels3 = set(['other indra'])

text4 = 'We cannot determine what INDRA means from this sentence.'

result_corpus = [(result[0], label, i)
                 for i, result in enumerate([(result1, labels1),
                                             (result2, labels2),
                                             (result3, labels3)])
                 for label in result[1]]

#  content for corpus building with synomous shortforms
groundings1 = {'nanoparticle': 'nano',
               'natriuretic peptide': 'peptide'}

groundings2 = {'nanoparticles': 'nano',
               'natriuretic peptides': 'peptide'}

text5 = ('The application of nanoparticles (NPs) for industrial processes'
         ' and consumer products is rising at an exponential rate.')
result5 = ('The application of NPs for industrial processes and consumer'
           ' products is rising at an exponential rate.')
labels5 = set(['nano'])

text6 = ('Understanding the mechanism of nanoparticle (NP) induced toxicity'
         ' is important for nanotoxicological and nanomedicinal studies.')
result6 = ('Understanding the mechanism of NP induced toxicity is important'
           ' for nanotoxicological and nanomedicinal studies.')
labels6 = set(['nano'])

text7 = ('Nanoparticle (NP) PET/CT imaging of natriuretic peptide (NP)'
         ' clearance receptor in prostate cancer.')
result7 = ('NP PET/CT imaging of NP clearance receptor in prostate cancer.')
labels7 = set(['nano', 'peptide'])

result_corpus2 = [(result[0], label, i) for
                  i, result in enumerate([(result5, labels5),
                                          (result6, labels6),
                                          (result7, labels7)])
                  for label in result[1]]


def test__process_text():
    labeler = AdeftLabeler({'INDRA': longforms})

    for text, result, labels in [(text1, result1, labels1),
                                 (text2, result2, labels2),
                                 (text3, result3, labels3)]:
        datapoints = labeler._process_text(text)
        assert len(datapoints) == len(labels)
        assert all([datapoint[0] == result for datapoint in datapoints])
        assert all([datapoint[1] in labels for datapoint in datapoints])

    assert labeler._process_text(text4) is None


def test__process_text_multiple():
    labeler = AdeftLabeler({'NP': groundings1, 'NPs': groundings2})
    for text, result, labels in [(text5, result5, labels5),
                                 (text6, result6, labels6),
                                 (text7, result7, labels7)]:
        datapoints = labeler._process_text(text)
        assert len(datapoints) == len(labels)
        assert all([datapoint[0] == result for datapoint in datapoints])
        assert all([datapoint[1] in labels for datapoint in datapoints])


def test_build_from_texts():
    labeler = AdeftLabeler({'INDRA': longforms})
    corpus = labeler.build_from_texts([(text1, 0),
                                       (text2, 1),
                                       (text3, 2),
                                       (text4, 3)])
    assert set(corpus) == set(result_corpus)


def test__build_from_texts_multiple():
    labeler = AdeftLabeler({'NP': groundings1, 'NPs': groundings2})
    corpus = labeler.build_from_texts([(text5, 0),
                                       (text6, 1),
                                       (text7, 2)])
    assert set(corpus) == set(result_corpus2)
