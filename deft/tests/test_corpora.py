from deft.recognize import LongformRecognizer
from deft.modeling.corpora import CorpusBuilder


longforms = ['integrated network and dynamical reasoning assembler',
             'indonesian debt restructuring agency']

text1 = ('The Integrated Network and Dynamical Reasoning Assembler'
         ' (INDRA) is an automated model assembly system interfacing'
         ' with NLP systems and databases to collect knowledge, and'
         ' through a process of assembly, produce causal graphs and'
         ' dynamical models. INDRA draws on natural language'
         ' processing systems and structured databases to collect'
         ' mechanistic and causal assertions, represents them in'
         ' standardized form (INDRA statements), and assembles them'
         ' into various modeling formalisms including causal graphs'
         ' and dynamical models')

result1 = ('The Integrated Network and Dynamical Reasoning Assembler'
           ' is an automated model assembly system interfacing with'
           ' NLP systems and databases to collect knowledge, and'
           ' through a process of assembly, produce causal graphs and'
           ' dynamical models. INDRA draws on natural language'
           ' processing systems and structured databases to collect'
           ' mechanistic and causal assertions, represents them in'
           ' standardized form (INDRA statements), and assembles them'
           ' into various modeling formalisms including causal graphs'
           ' and dynamical models')

labels1 = set([longforms[0]])


text2 = ('The Integrated Network and Dynamical Reasoning Assembler'
         ' (INDRA) is an automated model assembly system interfacing'
         ' with NLP systems and databases to collect knowledge, and'
         ' through a process of assembly, produce causal graphs and'
         ' dynamical models. The Indonesian Debt Restructuring Agency'
         ' (INDRA) shares the same acronym')

labels2 = set([longforms[0], longforms[1]])

result2 = ('The Integrated Network and Dynamical Reasoning Assembler'
           ' is an automated model assembly system interfacing with'
           ' NLP systems and databases to collect knowledge, and'
           ' through a process of assembly, produce causal graphs and'
           ' dynamical models. The Indonesian Debt Restructuring Agency'
           ' shares the same acronym')


text3 = ('In this sentence, (INDRA) appears but it is not preceded by a'
         ' recognized longform. Does it refer to the indonesian debt'
         ' restructuring agency (INDRA)?')

result3 = ('In this sentence, appears but it is not preceded by a'
           ' recognized longform. Does it refer to the indonesian debt'
           ' restructuring agency ?')

labels3 = set([longforms[1]])

text4 = 'We cannot determine what INDRA means from this sentence.'


def test__process_text():
    lfr = LongformRecognizer('INDRA', longforms)
    cb = CorpusBuilder(lfr)

    for text, result, labels in [(text1, result1, labels1),
                                 (text2, result2, labels2),
                                 (text3, result3, labels3)]:
        datapoints = cb._process_text(text)
        print(result)
        for datapoint in datapoints:
            print(datapoint[0])
        assert len(datapoints) == len(labels)
        assert all([datapoint[0] == result for datapoint in datapoints])
        assert all([datapoint[1] in labels for datapoint in datapoints])

    assert cb._process_text(text4) is None