import pandas as pd
from acromine import AcroMine
from get_rawtext import extract_text_general


ER_df = pd.read_pickle('../data/ER_statements.pkl')
ER_df = ER_df.groupby('text_id').first()
ER_df = ER_df[~ER_df.fulltext.isna()]
ER_df.fulltext = ER_df.fulltext.apply(lambda x: ' '.join(x.split()))


ER_fulltexts = ER_df.fulltext.values

IR_df = pd.read_pickle('../data/IR_statements.pkl')
IR_df = IR_df.groupby('text_id').first()
IR_fulltexts = IR_df.fulltext
IR_fulltexts = [extract_text_general(fulltext) for
                fulltext in IR_fulltexts]
IR_fulltexts = [fulltext for fulltext in IR_fulltexts if fulltext]
mine = AcroMine('IR')
mine.consume(IR_fulltexts)
