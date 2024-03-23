import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")


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

    def calculate_option_price(self, x: pd.Series):
        date = x["QUOTE_DATE"]
        rfr = round(self.rfr["DGS10"][self.rfr["DATE"] == date].values[0], 4)
        option_price = self.black_scholes(S=x["UNDERLYING_LAST"], K=x["STRIKE"], T=x["DTE"], r=rfr, sigma=x["C_IV"], option="call")
        return option_price

    def black_scholes(self, S, K, T, r, sigma, option='call'):
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = (math.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        if option == 'call':
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        if option == 'put':
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def main(path, rfr_path, vis=False):
    df = pd.read_csv(path)
    df = format_column_names(df)
    df = df[df["QUOTE_DATE"]<="2019-07-29"]
    df = df[df["DTE"]<=14]
    unique_exp_dates = df["EXPIRE_DATE"].unique()
    option_pricer = OptionPricer(rfr_path)
    exp_date_to_strike_dist_from_underlying = defaultdict(defaultdict)
    for i in unique_exp_dates:
        sub_df = df[df["EXPIRE_DATE"]==i]
        unique_strikes = sub_df["STRIKE"].unique()
        for s in unique_strikes:
            s_df = sub_df[sub_df["STRIKE"]==s]
            max_dte = s_df["DTE"].max()
            max_dte_df = s_df[s_df["DTE"]==max_dte]
            dist_from_underlying = (max_dte_df["STRIKE"].values - max_dte_df["UNDERLYING_LAST"].values)/max_dte_df["UNDERLYING_LAST"].values

            exp_date_to_strike_dist_from_underlying[str(i)][str(s)] = dist_from_underlying[0]

    desired_levels = [0, 0.1, 0.25]
    time_lag_levels = np.arange(1, 8, 1)
    time_lag_all_correlations = defaultdict(list)
    desired_levels_colors = ["blue", "green", "orange"]
    chosen_exp_date_strike_levels = get_exp_date_strike_at_desried_levels(exp_date_to_strike_dist_from_underlying, desired_levels)
    chosen_exp_date_strike_levels["EXPIRE_DATE"] = pd.to_datetime(chosen_exp_date_strike_levels["EXPIRE_DATE"])
    chosen_exp_date_strike_levels["STRIKE"] = chosen_exp_date_strike_levels["STRIKE"].astype(float)
    all_correlations = []
    num_options_missed_count = 0
    for ind, row in tqdm(chosen_exp_date_strike_levels.iterrows(), desc="Iterating over Options Chain:"):
        underlying_exist=False
        date = row["EXPIRE_DATE"]
        # strike = row["STRIKE"]
        options_chain = df[df["EXPIRE_DATE"]==date]
        if vis:
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True)

        options_chain_timeseries = defaultdict(defaultdict)

        chosen_exp_date_strike_levels_at_date = chosen_exp_date_strike_levels[chosen_exp_date_strike_levels["EXPIRE_DATE"]==date]
        unique_strikes = chosen_exp_date_strike_levels_at_date["STRIKE"].unique()
        unique_levels = chosen_exp_date_strike_levels_at_date["LEVEL"].unique()
        for strike, level in zip(unique_strikes, unique_levels):

            options_chain_strike = options_chain[options_chain["STRIKE"]==strike]
            options_chain_strike = options_chain_strike.sort_values(by="DTE")
            options_chain_strike = options_chain_strike.set_index("QUOTE_DATE")
            options_chain_strike["QUOTE_DATE"] = options_chain_strike.index

            new_suffix = f"_AT_STRIKE_{strike}"

            options_chain_strike[f"OPTION_PRICE_{new_suffix}"] = options_chain_strike.apply(lambda x:
                                                                              option_pricer.calculate_option_price(x), axis=1)
            # level = options_chain_strike["LEVEL"].unique()[0]
            level_index = desired_levels.index(level)
            level_color = desired_levels_colors[level_index]

            options_chain_strike[f"STRIKE_{new_suffix}"] = options_chain_strike["STRIKE"]
            options_chain_strike[f"C_GAMMA_{new_suffix}"] = options_chain_strike["C_GAMMA"]
            options_chain_strike[f"C_DELTA_{new_suffix}"] = options_chain_strike["C_DELTA"]

            if vis:
                if not underlying_exist:
                    options_chain_strike["UNDERLYING_LAST"].plot(ax=ax0, color="black")
                    underlying_exist=True

                options_chain_strike[f"STRIKE_{new_suffix}"].plot(ax=ax0, color=level_color)
                ax0.title.set_text('Price')
                ax0.legend()

                options_chain_strike[f"C_GAMMA_{new_suffix}"].plot(ax=ax1, color=level_color)
                ax1.title.set_text('Call GAMMA')
                ax1.legend()

                options_chain_strike[f"C_DELTA_{new_suffix}"].plot(ax=ax2, color=level_color)
                ax2.title.set_text('Call DELTA')
                ax2.legend()

                options_chain_strike[f"OPTION_PRICE_{new_suffix}"].plot(ax=ax3, color=level_color)
                ax3.title.set_text('Options Price (Black Scholes)')
                ax3.legend()

            options_chain_timeseries["UNDERLYING_LAST"] = options_chain_strike["UNDERLYING_LAST"]
            options_chain_timeseries[f"OPTION_PRICE_AT_LEVEL_{level}"] = options_chain_strike[f"OPTION_PRICE_{new_suffix}"]
            options_chain_timeseries[f"C_GAMMA_AT_LEVEL_{level}"] = options_chain_strike[f"C_GAMMA_{new_suffix}"]
            options_chain_timeseries[f"C_DELTA_AT_LEVEL_{level}"] = options_chain_strike[f"C_DELTA_{new_suffix}"]

        options_chain_corr = pd.DataFrame.from_dict(options_chain_timeseries).corr()
        if len(options_chain_corr)==10:
            for time_lag in time_lag_levels:
                lag_df = pd.DataFrame.from_dict(options_chain_timeseries)
                lag_df["UNDERLYING_LAST"] = lag_df["UNDERLYING_LAST"].shift(-time_lag)
                lag_df_values = lag_df.dropna()
                lag_df_values_corr = np.expand_dims(lag_df_values.corr().values, axis=0)
                if not np.isnan(np.sum(lag_df_values_corr)):
                    time_lag_all_correlations[f"LAG_{time_lag}"].append(lag_df_values_corr)

            columns = options_chain_corr.columns

        options_chain_corr = np.expand_dims(options_chain_corr.values, axis=0)
        if options_chain_corr.shape == (1, 10, 10) and not np.isnan(np.sum(options_chain_corr)):
            all_correlations.append(options_chain_corr)
        else:
            num_options_missed_count += 1
        # plt.show()

        # delta, volume (CALL), option price, laggs,
    all_correlations = np.concatenate(all_correlations, axis=0)
    correlation_mean = pd.DataFrame(np.mean(all_correlations, axis=0), columns=columns, index=columns)
    correlation_std = pd.DataFrame(np.std(all_correlations, axis=0), columns=columns, index=columns)

    lag_corr_dfs = {}
    for key, corr_matricies_list in time_lag_all_correlations.items():
        corr_matrix = np.concatenate(corr_matricies_list, axis=0)
        lag_correlation_mean = pd.DataFrame(np.mean(corr_matrix, axis=0), columns=columns, index=columns)
        lag_correlation_std = pd.DataFrame(np.std(corr_matrix, axis=0), columns=columns, index=columns)
        lag_correlation_min = pd.DataFrame(np.min(corr_matrix, axis=0), columns=columns, index=columns)
        lag_correlation_max = pd.DataFrame(np.max(corr_matrix, axis=0), columns=columns, index=columns)
        lag_corr_dfs[f"{key}_mean"] = lag_correlation_mean
        lag_corr_dfs[f"{key}_std"] = lag_correlation_std
        lag_corr_dfs[f"{key}_min"] = lag_correlation_min
        lag_corr_dfs[f"{key}_max"] = lag_correlation_max

    _=0

if __name__ == "__main__":
    path = "/Users/tom/Desktop/MBA/SemesterA/InvestmentTheory/Project/Data/aapl_2016_2020.csv"
    rfr_path = "/Users/tom/Desktop/MBA/SemesterA/InvestmentTheory/Project/Data/DGS10.csv"
    main(path, rfr_path)
