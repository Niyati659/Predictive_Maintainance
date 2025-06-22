#non constant columns
import pandas as pd
column_names = ['unit_number', 'time_in_cycles',
                'op_setting_1', 'op_setting_2', 'op_setting_3',
                'T2', 'T24', 'T30', 'T50', 'P2',
                'P15', 'P30', 'Nf', 'Nc', 'epr',
                'Ps30', 'phi', 'NRf', 'NRc', 'BPR',
                'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
df=pd.read_csv("train_FD001.csv")
for i in df.columns:
        print(i)
    