import numpy as np
import pandas as pd


def convert_date_cols_to_datetime(df):
    date_col_names = ['QUOTE_DATE','EXPIRE_DATE']
    for d_col in date_col_names:
        df[d_col] = pd.to_datetime(df[d_col])
    return df


def drop_irrelevant_time_columns(df):
    cols = df.columns.tolist()
    cols.remove("QUOTE_UNIXTIME")
    cols.remove("EXPIRE_UNIX")
    cols.remove("QUOTE_READTIME")
    df = df[cols]
    return df

def drop_irrelevant_options_cols(df):
    cols = df.columns.tolist()
    cols.remove("C_SIZE")
    cols.remove("P_SIZE")
    df = df[cols]
    return df


def convert_numerical_cols_to_float(df):
    call_related_features = ["C_DELTA", "C_GAMMA", "C_VEGA", "C_THETA", "C_RHO", "C_IV", "C_VOLUME",
                             "C_LAST", "C_BID", "C_ASK"]
    other_numerical_cols = ["STRIKE", "UNDERLYING_LAST", "STRIKE_DISTANCE", "STRIKE_DISTANCE_PCT"]
    put_related_features = ["P_DELTA", "P_GAMMA", "P_VEGA", "P_THETA", "P_RHO", "P_IV", "P_VOLUME",
                             "P_LAST", "P_BID", "P_ASK"]
    numerical_cols = call_related_features + put_related_features + other_numerical_cols
    for ncol in numerical_cols:
        df[ncol] = pd.to_numeric(df[ncol], errors='coerce')

    return df


def format_column_names(df):
    columns = df.columns
    columns = [s.replace('[', '') for s in columns]
    columns = [s.replace(']', '') for s in columns]
    columns = [s.replace(' ', '') for s in columns]
    df.columns = columns
    df = convert_date_cols_to_datetime(df)
    df = drop_irrelevant_time_columns(df)
    df = drop_irrelevant_options_cols(df)
    df = convert_numerical_cols_to_float(df)

    return df


def main(path):
    df = pd.read_csv(path)
    df = format_column_names(df)


    pass


if __name__ == "__main__":
    path = "/Users/tom/Desktop/MBA/SemesterA/InvestmentTheory/Project/Data/aapl_2016_2020.csv"
    main(path)
