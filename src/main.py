import numpy as np
import pandas as pd
from math import *
from scipy.stats import norm
from scipy.optimize import fmin_bfgs
from scipy import *
from scipy.integrate import quad
from scipy import optimize
import matplotlib.pyplot as plt
from collections import defaultdict

from src.options_data_classes import *


# Calibration(kappa, theta, sigma, rho, v0)
def calibrate(init_val, market_datas):
    def error(x):
        kappa, theta, eta, rho, v0 = x
        print('x=', kappa, theta, eta, rho, v0)
        result = 0.0
        for i in range(0, len(market_datas)):
            s0, k = market_datas["S"].iloc[i], market_datas["K"].iloc[i]
            market_price, r  = market_datas["PRICE"].iloc[i], market_datas["mu"].iloc[i]
            T = market_datas["T"].iloc[i]
            # s0, k, market_price, r, T = market_datas.iloc[i]
            # print(s0, k, market_price, r, T)

            heston = HestonProcess(mu=r, kappa=kappa, theta=theta, eta=eta, rho=rho)
            opt = Option(s0=s0, v0=v0, T=T, K=k, call=True)

            heston_price = monte_carlo_simulation_LS(option=opt, process=heston, n=5, m=252)
            result += (heston_price - market_price) ** 2

        print('** resulting error: ', result)
        return result

    opt = optimize.least_squares(error, init_val)  # ??? or fmin, or leastsq
    return opt

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



def main(path):
    init_val = [1.1, 0.1, 0.4, -0.0, 0.1]
    df = pd.read_csv(path)
    df = format_column_names(df)
    df = df[df["QUOTE_DATE"]<="2019-07-29"]
    df = df[df["DTE"]<=14]
    unique_exp_dates = df["EXPIRE_DATE"].unique()
    # df["IS_CALL_IN_MONEY"] = df.apply(lambda x: is_call_in_the_money(x))
    all_options = defaultdict()

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
            all_options[f"{str(i)}_{str(s)}"] = s_df

    desired_levels = [0, 0.1, 0.25]
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

    first_key = list(all_options.keys())[35]

    first_option = all_options[first_key]
    relevant_cols = ["QUOTE_DATE", "P_VOLUME", "C_VOLUME", "UNDERLYING_LAST", "P_ASK", "C_BID", "STRIKE"]

    # sub_first_option_mean = sub_first_option_mean.reset_index(drop=False)
    options_chain = first_option[relevant_cols]

    options_chain = options_chain.set_index("QUOTE_DATE")
    options_chain["Implied_Forward_Rate"] = options_chain["STRIKE"] + (options_chain["P_ASK"]-options_chain["C_BID"])
    options_chain["PREMIUM"] = options_chain["P_ASK"]-options_chain["C_BID"]
    options_chain["PUT_CALL_RATIO"] = options_chain["P_VOLUME"]/options_chain["C_VOLUME"]
    # same strike sell put, buy call, same expiry,
    # strike + (put-call) = implied forward_rate

    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, ncols=1, sharex=True)
    options_chain["UNDERLYING_LAST"].plot(ax=ax0)
    options_chain["STRIKE"].plot(ax=ax0)
    ax0.title.set_text('Underlying Price')
    ax0.set_ylabel("Price")

    options_chain["Implied_Forward_Rate"].plot(ax=ax1)
    options_chain["STRIKE"].plot(ax=ax1)
    ax1.title.set_text('Implied Forward Rate')
    ax1.set_ylabel("Price")

    options_chain["P_ASK"].plot(ax=ax2)
    ax2.title.set_text('Put Ask Price')
    ax2.set_ylabel("Price")

    options_chain["C_BID"].plot(ax=ax3)
    ax3.title.set_text('Call Bid Price')
    ax3.set_ylabel("Price")

    options_chain["PREMIUM"].plot(ax=ax4)
    ax4.title.set_text('Put-Call Premium (P_ASK - C_BID)')
    ax4.set_ylabel("Price")

    options_chain["C_P_VOLUME"] = options_chain["C_VOLUME"].fillna(0)+options_chain["P_VOLUME"].fillna(0)
    ax5.bar(options_chain.index, options_chain["C_VOLUME"].fillna(0), color="g")
    ax5.bar(options_chain.index, options_chain["C_P_VOLUME"],
            bottom=options_chain["C_VOLUME"].fillna(0), color="r")
    ax5.title.set_text('Put-Call Volume')
    ax5.set_ylabel("Volume")



    plt.show()
    _=0


    # df["S"] = df["UNDERLYING_LAST"]
    # df["K"] = df["STRIKE"]
    # df["PRICE"] = df["C_BID"]
    # df["mu"] = df["C_RHO"] / (df["C_BID"] * 100)
    # df["T"] = df["DTE"] / 365
    #
    # df_mini = df.iloc[:10]

    # res = calibrate(init_val, df_mini)  # very slow!!!!!

    pass


if __name__ == "__main__":
    path = "/Users/tom/Desktop/MBA/SemesterA/InvestmentTheory/Project/Data/aapl_2016_2020.csv"
    main(path)
