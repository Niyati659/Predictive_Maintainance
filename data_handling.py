import pandas as pd
from sklearn.preprocessing import StandardScaler

# Calculates Remaining Useful Life (RUL) for each cycle

def computeRUL(df):
    # Get the last cycle number for each engine
    max_cycle = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycle.columns = ['unit_number', 'max_cycle']

    # Merge back so every row knows the last cycle of its engine
    df = df.merge(max_cycle, on='unit_number', how='left')

    # RUL is just max - current
    df['RUL'] = df['max_cycle'] - df['time_in_cycles']

    # Drop extra column
    df.drop(columns=['max_cycle'], inplace=True)

    return df

def prepare_features_and_target(df, scale=False, include_target=True):
    exclude_cols = ['unit_number', 'time_in_cycles']
    if include_target and 'RUL' in df.columns:
        exclude_cols.append('RUL')
    if 'dataset_id' in df.columns:
        exclude_cols.append('dataset_id')

    features = df.columns.difference(exclude_cols)
    X = df[features]

    y = df['RUL'] if include_target and 'RUL' in df.columns else None

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=features)

    return X, y, scaler


# Converts raw .txt data to .csv, drops constants
# Only needs to be run once

def convert_txt_to_csv():
    column_names = ['unit_number', 'time_in_cycles',
                    'op_setting_1', 'op_setting_2', 'op_setting_3',
                    'T2', 'T24', 'T30', 'T50', 'P2',
                    'P15', 'P30', 'Nf', 'Nc', 'epr',
                    'Ps30', 'phi', 'NRf', 'NRc', 'BPR',
                    'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
    
    file_ids = ['FD001', 'FD002', 'FD003', 'FD004']

    for file_id in file_ids:
        for mode in ['train', 'test']:
            raw_path = f"{mode}_{file_id}.txt"
            out_path = f"{mode}_{file_id}.csv"

            # Read file, clean it up
            df = pd.read_csv(raw_path, sep=r'\s+', header=None, names=column_names)
            df.dropna(inplace=True)

            # Drop any constant columns
            const_cols = df.columns[df.nunique() <= 1]
            df.drop(columns=const_cols, inplace=True)

            df.to_csv(out_path, index=False)
            print(f"[✓] Saved {out_path}")

        # Also convert RUL files
        rul_raw = f"RUL_{file_id}.txt"
        rul_out = f"RUL_{file_id}.csv"

        df_rul = pd.read_csv(rul_raw, header=None, names=["RUL"])
        df_rul.dropna(inplace=True)
        df_rul.to_csv(rul_out, index=False)
        print(f"[✓] Saved {rul_out}")

# Run file conversion directly if script is executed

if __name__ == "__main__":
    convert_txt_to_csv()
