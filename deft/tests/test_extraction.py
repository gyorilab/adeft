from deft.extraction import Processor
from deft.nlp.resources import stopwords


example_text1 = ('The Integrated Network and Dynamical Reasoning Assembler'
                 ' (INDRA) is an automated model assembly system interfacing'
                 ' with NLP systems and databases to collect knowledge, and'
                 ' through a process of assembly, produce causal graphs and'
                 ' dynamical models. INDRA draws on natural language'
                 ' processing systems and structured databases to collect'
                 ' mechanistic and causal assertions, represents them in'
                 ' standardized form (INDRA statements), and assembles them'
                 ' into various modeling formalisms including causal graphs'
                 ' and dynamical models')
result1 = 'the integrated network and dynamical reasoning assembler'

example_text2 = ('Integrated Network and Dynamical Reasoning Assembler'
                 ' (INDRA) generates executable models of pathway dynamics'
                 ' from natural language.')
result2 = 'integrated network and dynamical reasoning assembler'


example_text3 = ('The Indonesian Debt Restructuring Agency (INDRA) was'
                 ' established by the Jakarta Initiative in 1998.')
result3 = 'the indonesian debt restructuring agency'

example_text4 = ('An Indonesian Debt Restructuring Agency (INDRA) was'
                 ' established to provide foreign-exchange cover for'
                 ' Indonesian corporations with foreign currency denominated'
                 ' debt.')
result4 = 'an indonesian debt restructuring agency'


def test_get_candidates():
    """Test extraction of maximal longform candidate from sentence
    """
    # Test default processor
    processor1 = Processor('INDRA')
    # Test processor with optional excluded words
    processor2 = Processor('INDRA', stopwords)
    example = ('The Integrated Network and Dynamical Reasoning Assembler'
               ' (INDRA) is an automated model assembly system interfacing'
               ' with NLP systems and databases to collect knowledge, and'
               ' through a process of assembly, produce causal graphs and'
               ' dynamical models.')
    candidate1 = processor1._get_candidate(example)
    candidate2 = processor2._get_candidate(example)
    assert candidate1 == result1.split()
    assert candidate2 == ['dynamical', 'reasoning', 'assembler']


def test_extract():
    """Test extraction of maximal longform candidate from text
    """
    processor = Processor('INDRA')
    for text, result in zip([example_text1, example_text2,
                             example_text3, example_text4],
                            [result1, result2, result3, result4]):
        candidates = processor.extract(text)
        assert len(candidates) == 1
        assert candidates[0] == result.split()
