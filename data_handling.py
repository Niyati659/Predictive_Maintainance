import pandas as pd
column_names = ['unit_number', 'time_in_cycles',
                'op_setting_1', 'op_setting_2', 'op_setting_3',
                'T2', 'T24', 'T30', 'T50', 'P2',
                'P15', 'P30', 'Nf', 'Nc', 'epr',
                'Ps30', 'phi', 'NRf', 'NRc', 'BPR',
                'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
file_ids = ['FD001', 'FD002', 'FD003', 'FD004']
for file_id in file_ids:
    for file_type in ['train', 'test']:
        txt_file = f"{file_type}_{file_id}.txt"
        csv_file = f"{file_type}_{file_id}.csv"
        df = pd.read_csv(txt_file, sep='\s+', header=None, names=column_names)
        df=df.dropna()
        nunique_counts = df.nunique()
        constant_cols=df.columns[nunique_counts<=1]
        df.drop(columns=constant_cols,inplace=True)
        df.to_csv(csv_file, index=False)
    rul_txt = f"RUL_{file_id}.txt"
    rul_csv = f"RUL_{file_id}.csv"
    df_rul = pd.read_csv(rul_txt, header=None, names=["RUL"])
    df_rul.dropna(inplace=True)
    df_rul.to_csv(rul_csv, index=False)
