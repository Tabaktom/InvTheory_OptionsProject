import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from collections import defaultdict



class OptionPricer:
    def __init__(self, rfr_path: str):
        self.rfr = pd.read_csv(rfr_path)
        self.rfr["DATE"] = pd.to_datetime(self.rfr["DATE"])
        self.clean_price_col()

    def clean_price_col(self):
        self.rfr["DGS10"][self.rfr["DGS10"]=="."] = np.nan
        self.rfr["DGS10"] = self.rfr["DGS10"].ffill()
        self.rfr["DGS10"] = self.rfr["DGS10"].astype(float)
        self.rfr["DGS10"] = self.rfr["DGS10"]/100

    def calculate_option_price(self, x: pd.Series, option_type: str="call"):
        date = x["QUOTE_DATE"]
        if option_type=="call":
            iv_col_name = "C_IV"
        else:
            iv_col_name = "P_IV"
        rfr = round(self.rfr["DGS10"][self.rfr["DATE"] == date].values[0], 4)
        option_price = self.black_scholes(S=x["UNDERLYING_LAST"], K=x["STRIKE"], T=x["DTE"],
                                          r=rfr, sigma=x[iv_col_name], option=option_type)
        return option_price

    def black_scholes(self, S, K, T, r, sigma, option='call'):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        if option == 'call':
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        if option == 'put':
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

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

def get_dist_to_level_and_indices(dist_from_underlying_array, level):
    indices = []
    distances = []
    for ind, dist in enumerate(dist_from_underlying_array):
        if dist-level>=0:
            indices.append(ind)
            distances.append(dist-level)
    return distances, indices


def get_exp_date_strike_at_desried_levels(exp_date_to_strike_dist_from_underlying, desired_levels):
    exp_date_to_strike_at_desired_levels = defaultdict(list)
    for date, strike_dict in exp_date_to_strike_dist_from_underlying.items():
        (strike_prices, dist_from_underlying_array) = list(strike_dict.keys()), list(strike_dict.values())
        for level in desired_levels:
            distances, indices = get_dist_to_level_and_indices(dist_from_underlying_array, level)
            if len(distances)==0:
                continue
            relevant_distance_from_underlying = [dist_from_underlying_array[ind] for ind in indices]
            relevant_strikes = [strike_prices[ind] for ind in indices]

            closest_dist_to_level = min(distances)
            closest_dist_to_level_index = distances.index(closest_dist_to_level)
            strike_price_for_level = relevant_strikes[closest_dist_to_level_index]

            dist_to_underlying_closes_to_level = relevant_distance_from_underlying[closest_dist_to_level_index]

            exp_date_to_strike_at_desired_levels["EXPIRE_DATE"].append(date)
            exp_date_to_strike_at_desired_levels["LEVEL"].append(level)
            exp_date_to_strike_at_desired_levels["STRIKE"].append(strike_price_for_level)
            exp_date_to_strike_at_desired_levels["DIST_TO_UNDERLYING"].append(dist_to_underlying_closes_to_level)

    exp_date_to_strike_at_desired_levels = pd.DataFrame.from_dict(exp_date_to_strike_at_desired_levels)
    return exp_date_to_strike_at_desired_levels

