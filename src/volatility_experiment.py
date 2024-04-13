import warnings
from tqdm import tqdm
import os
from collections import defaultdict
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from src.utils import *
warnings.filterwarnings("ignore")


def get_exp_date_to_strike_dist_from_underlying(df):
    df = format_column_names(df)
    df = df[df["QUOTE_DATE"] <= "2019-07-29"]
    df = df[df["DTE"] <= 14]
    unique_exp_dates = df["EXPIRE_DATE"].unique()

    exp_date_to_strike_dist_from_underlying = defaultdict(defaultdict)
    for i in unique_exp_dates:
        sub_df = df[df["EXPIRE_DATE"] == i]
        unique_strikes = sub_df["STRIKE"].unique()
        for s in unique_strikes:
            s_df = sub_df[sub_df["STRIKE"] == s]
            max_dte = s_df["DTE"].max()
            max_dte_df = s_df[s_df["DTE"] == max_dte]
            dist_from_underlying = (max_dte_df["STRIKE"].values - max_dte_df["UNDERLYING_LAST"].values) / max_dte_df[
                "UNDERLYING_LAST"].values

            exp_date_to_strike_dist_from_underlying[str(i)][str(s)] = dist_from_underlying[0]

    return exp_date_to_strike_dist_from_underlying, df


def main(path, rfr_path, out_dir, vis=False):
    df = pd.read_csv(path)
    exp_date_to_strike_dist_from_underlying, df = get_exp_date_to_strike_dist_from_underlying(df)

    option_pricer = OptionPricer(rfr_path)

    desired_levels = [-0.25, -0.1, 0, 0.1, 0.25]
    chosen_exp_date_strike_levels = get_exp_date_strike_at_desried_levels(exp_date_to_strike_dist_from_underlying, desired_levels)
    chosen_exp_date_strike_levels["EXPIRE_DATE"] = pd.to_datetime(chosen_exp_date_strike_levels["EXPIRE_DATE"])
    chosen_exp_date_strike_levels["STRIKE"] = chosen_exp_date_strike_levels["STRIKE"].astype(float)

    call_volatility_prems_corrs = []
    put_volatility_prems_corrs = []
    sentiment_volatility_corrs = []
    call_to_under_change_corrs = []
    put_to_under_change_corrs = []

    call_plot_df = pd.DataFrame(index=np.arange(15))
    put_plot_df = pd.DataFrame(index=np.arange(15))
    sentiment_plot_df = pd.DataFrame(index=np.arange(15))
    underlying_plot_df = pd.DataFrame(index=np.arange(15))
    implied_vol_smile_tensor = np.zeros((15, len(desired_levels), len(chosen_exp_date_strike_levels)))
    for ind, row in tqdm(chosen_exp_date_strike_levels.iterrows(), desc="Iterating over Options Chain:"):
        implied_vol_smile_matrix = pd.DataFrame(np.zeros((15, len(desired_levels))), columns=desired_levels)
        underlying_exist = False
        date = row["EXPIRE_DATE"]
        # strike = row["STRIKE"]
        options_chain = df[df["EXPIRE_DATE"]==date]

        chosen_exp_date_strike_levels_at_date = chosen_exp_date_strike_levels[chosen_exp_date_strike_levels["EXPIRE_DATE"]==date]
        unique_strikes = chosen_exp_date_strike_levels_at_date["STRIKE"].unique().tolist()
        unique_levels = chosen_exp_date_strike_levels_at_date["LEVEL"].unique().tolist()
        if len(unique_levels)!=len(desired_levels):
            continue
        level_zero_index = unique_levels.index(float(0))
        at_money_strike_price = unique_strikes[level_zero_index]
        unique_strikes.pop(level_zero_index)
        unique_levels.pop(level_zero_index)
        options_chain_at_money = options_chain[options_chain["STRIKE"] == at_money_strike_price]
        options_chain_at_money = options_chain_at_money.set_index("DTE", drop=False).sort_index() # change to quote date
        if len(options_chain_at_money) < 8:
            continue

        volatility_df = pd.DataFrame()
        for level_index, (strike, level) in enumerate(zip(unique_strikes, unique_levels)):

            options_chain_strike = options_chain[options_chain["STRIKE"]==strike]
            options_chain_strike = options_chain_strike.set_index("DTE", drop=False).sort_index() # change to quote date
            if len(volatility_df) == 0:
                volatility_df.index = options_chain_strike.index
                volatility_df["UNDERLYING_LAST"] = options_chain_strike["UNDERLYING_LAST"]
                underlying_plot_df[f"{str(date)}_AT_{at_money_strike_price}"] = volatility_df["UNDERLYING_LAST"]
                volatility_df["DTE"] = options_chain_strike["DTE"].astype(int)
                volatility_df.set_index("DTE", drop=False, inplace=True)
                volatility_df["PCT_CHANGE_UNDERLYING"] = volatility_df["UNDERLYING_LAST"].pct_change()

            # populate implied vol smile matrix
            if level == 0:
                implied_vol_series = options_chain_at_money["C_IV"]
            elif level > 0:
                implied_vol_series = options_chain_strike["C_IV"]
            else:
                implied_vol_series = options_chain_strike["P_IV"]

            implied_vol_smile_matrix[level] = implied_vol_series

            # calculate IV premium
            if level > 0:
                # call
                volatility_premium = options_chain_strike["C_IV"] - options_chain_at_money["C_IV"]
                options_chain_strike[f"OPTION_PRICE_CALL_AT_{strike}"] = options_chain_strike.apply(lambda x:
                                                                                                    option_pricer.calculate_option_price(
                                                                                                        x, "call"), axis=1)
                volatility_df["CALL_IV_PREM"] = volatility_premium
                volatility_df["IV_CALL_PCT_CHANGE"] = options_chain_strike["C_IV"].pct_change()
                call_plot_df[f"{str(date)}_AT_{strike}"] = volatility_df["CALL_IV_PREM"]
            else:
                # put
                volatility_premium = options_chain_strike["P_IV"] - options_chain_at_money["P_IV"]
                options_chain_strike[f"OPTION_PRICE_PUT_AT_{strike}"] = options_chain_strike.apply(lambda x:
                                                                                                   option_pricer.calculate_option_price(
                                                                                                       x, "put"), axis=1)
                volatility_df["PUT_IV_PREM"] = volatility_premium
                volatility_df["IV_PUT_PCT_CHANGE"] = options_chain_strike["P_IV"].pct_change()
                put_plot_df[f"{str(date)}_AT_{strike}"] = volatility_df["PUT_IV_PREM"]

        implied_vol_smile_tensor[:, :, ind] = implied_vol_smile_matrix.values
        sentiment_plot_df[str(date)] = volatility_df["CALL_IV_PREM"] - volatility_df["PUT_IV_PREM"]
        volatility_df["VOL_SENTIMENT"] = volatility_df["CALL_IV_PREM"] - volatility_df["PUT_IV_PREM"]
        volatility_df = volatility_df.dropna()
        corr_matrix = volatility_df.corr()

        call_corr = corr_matrix["UNDERLYING_LAST"].loc["CALL_IV_PREM"]
        put_corr = corr_matrix["UNDERLYING_LAST"].loc["PUT_IV_PREM"]
        sent_corr = corr_matrix["UNDERLYING_LAST"].loc["VOL_SENTIMENT"]

        call_to_under_change = corr_matrix["PCT_CHANGE_UNDERLYING"].loc["IV_CALL_PCT_CHANGE"]
        put_to_under_change = corr_matrix["PCT_CHANGE_UNDERLYING"].loc["IV_PUT_PCT_CHANGE"]
        if not np.isnan(call_to_under_change) and not np.isnan(put_to_under_change):
            call_to_under_change_corrs.append(call_to_under_change)
            put_to_under_change_corrs.append(put_to_under_change)
        call_volatility_prems_corrs.append(call_corr)
        put_volatility_prems_corrs.append(put_corr)
        sentiment_volatility_corrs.append(sent_corr)


    if vis:
        iv_prem_out_dir = os.path.join(out_dir, "IV_premium")
        os.makedirs(iv_prem_out_dir, exist_ok=True)
        for sent_col, call_col, put_col, under_col in zip(sentiment_plot_df.columns, call_plot_df.columns,
                                                      put_plot_df.columns, underlying_plot_df.columns):
            path = os.path.join(iv_prem_out_dir, f"Expiry_{under_col}.png")
            at_money_strike_price = float(under_col.split("AT_")[-1])
            fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True, figsize=(25, 15))
            # sent_col = sentiment_plot_df.columns[0]
            # call_col = call_plot_df.columns[0]
            # put_col = put_plot_df.columns[0]
            # under_col = underlying_plot_df.columns[0]

            underlying_plot_df.sort_index(ascending=False)[under_col].ffill().plot(ax=ax0)
            ax0.set_xlabel("Days to Expiry")
            ax0.title.set_text(f'Underlying Asset Price (T-14 at {under_col})')
            ax0.axhline(y=at_money_strike_price, c="black")

            # ax0.set_xticks(underlying_plot_df.sort_index(ascending=False).index.values)
            # ax0.invert_xaxis()

            sentiment_plot_df.sort_index(ascending=False)[sent_col].ffill().plot(ax=ax1)
            ax1.title.set_text(f'Volatility Sentiment Premium (T-14 at {sent_col})')
            ax1.axhline(y=0, c="black")
            # ax1.invert_xaxis()

            call_plot_df.sort_index(ascending=False)[call_col].ffill().plot(ax=ax2)
            ax2.title.set_text(f'Call Volatility Premium (T-14 at {call_col})')
            ax2.axhline(y=0, c="black")
            # ax2.invert_xaxis()

            put_plot_df.sort_index(ascending=False)[put_col].ffill().plot(ax=ax3)
            ax3.title.set_text(f'Put Volatility Premium (T-14 at {put_col})')
            ax3.axhline(y=0, c="black")
            ax3.set_xlabel("Days to Expiry")
            # ax3.set_xticks(put_plot_df.sort_index(ascending=False).index.values)
            ax3.invert_xaxis()
            plt.tight_layout()

            # plt.show()
            plt.savefig(path)
            plt.clf()


            # .interpolate(method='linear').plot()

    # Plot IV Smile for each DTE
    iv_smile_dir = os.path.join(out_dir, "IV_smile")
    os.makedirs(iv_smile_dir, exist_ok=True)
    for dte in range(len(implied_vol_smile_tensor)):
        file_path = os.path.join(iv_smile_dir, f"IV_smile_Dte_{dte}.png")
        dte_iv_smile_matrix = implied_vol_smile_tensor[dte, :, :]
        dte_iv_smile_matrix_mean = np.nanmean(dte_iv_smile_matrix, axis=1)
        dte_iv_smile_matrix_std = np.nanstd(dte_iv_smile_matrix, axis=1)
        dte_iv_smile_upper = dte_iv_smile_matrix_mean + dte_iv_smile_matrix_std
        dte_iv_smile_lower = dte_iv_smile_matrix_mean - dte_iv_smile_matrix_std

        fig = plt.Figure(figsize=(5, 5))
        plt.plot(desired_levels, dte_iv_smile_matrix_mean, label="Mean IV")
        plt.fill_between(desired_levels, dte_iv_smile_upper, dte_iv_smile_lower, alpha=0.2, label="Mean IV +- 1std")
        plt.xlabel("Option Strike Price Distance from Underlying Price at T-14")
        plt.ylabel("Implied Volatility")
        plt.title(f"Implied Volatility Smile at Days to Expiry={dte}")
        plt.legend()
        plt.tight_layout()

        plt.savefig(file_path)
        plt.clf()


def calculate_strategy_performance(path, distances_from_underlying_spot):
    df = pd.read_csv(path)
    ticker_name = os.path.basename(path).split("_")[0].upper()

    exp_date_to_strike_dist_from_underlying, df = get_exp_date_to_strike_dist_from_underlying(df)
    # distances_from_underlying_spot = [0.1, 0.25]
    relevant_cols = ["UNDERLYING_LAST", "C_IV", "P_IV", "C_BID", "C_ASK", "P_BID", "P_ASK", "DTE"]
    all_trades_df = []
    for distance_level in tqdm(distances_from_underlying_spot):
        desired_levels = [-distance_level, 0, distance_level]
        chosen_exp_date_strike_levels = get_exp_date_strike_at_desried_levels(exp_date_to_strike_dist_from_underlying,
                                                                              desired_levels)

        chosen_exp_date_strike_levels["EXPIRE_DATE"] = pd.to_datetime(chosen_exp_date_strike_levels["EXPIRE_DATE"])
        chosen_exp_date_strike_levels["STRIKE"] = chosen_exp_date_strike_levels["STRIKE"].astype(float)
        unique_expiry = chosen_exp_date_strike_levels["EXPIRE_DATE"].unique()

        for expiry_date in unique_expiry:
            contract_data = {"Put_OutOfMoney": None, "AtMoney": None, "Call_OutofMoney": None}
            contract_names = list(contract_data.keys())
            expiry_date_data = df[df["EXPIRE_DATE"]==expiry_date]
            strike_prices = chosen_exp_date_strike_levels["STRIKE"][chosen_exp_date_strike_levels["EXPIRE_DATE"]==expiry_date]
            strike_prices = strike_prices.sort_values(ascending=True)

            for ind, s in enumerate(strike_prices):
                level_name = contract_names[ind]
                strike_data = expiry_date_data[expiry_date_data["STRIKE"]==s]
                strike_data = strike_data.set_index("QUOTE_DATE")
                strike_data = strike_data[relevant_cols]
                contract_data[level_name] = strike_data

            if sum([isinstance(x, pd.DataFrame) for x in list(contract_data.values())])==3:
                contract_df = get_contract_df(strike_data, contract_data)
                trades_df = run_strategy(contract_df, ticker_name)
                all_trades_df.append(trades_df)

    all_trades = pd.concat(all_trades_df, axis=0).reset_index(drop=True)

    return all_trades


def get_contract_df(strike_data, contract_data):
    contract_df = pd.DataFrame(index=strike_data.index)
    contract_df["DTE"] = contract_data["AtMoney"]["DTE"]
    contract_df["UNDERLYING_LAST"] = contract_data["AtMoney"]["UNDERLYING_LAST"]
    contract_df["AT_MONEY_IV"] = contract_data["AtMoney"]["C_IV"]

    contract_df["CALL_BUY"] = contract_data["Call_OutofMoney"]["C_ASK"]
    contract_df["CALL_SELL"] = contract_data["Call_OutofMoney"]["C_BID"]
    contract_df["CALL_IV"] = contract_data["Call_OutofMoney"]["C_IV"]

    contract_df["PUT_BUY"] = contract_data["Put_OutOfMoney"]["P_ASK"]
    contract_df["PUT_SELL"] = contract_data["Put_OutOfMoney"]["P_BID"]
    contract_df["PUT_IV"] = contract_data["Put_OutOfMoney"]["P_IV"]

    contract_df["CALL_IV_PREM"] = contract_df["CALL_IV"] - contract_df["AT_MONEY_IV"]
    contract_df["PUT_IV_PREM"] = contract_df["PUT_IV"] - contract_df["AT_MONEY_IV"]
    contract_df["SENTIMENT_IV_PREM"] = contract_df["CALL_IV_PREM"] - contract_df["PUT_IV_PREM"]

    contract_df = contract_df.sort_index(ascending=True)
    return contract_df


def run_strategy(contract_df, ticker_name, threshold_delta_for_entry=0.02, threshold_for_reversal_exit=0.005):
    equity = None
    trades_df = pd.DataFrame(columns=["DayEnter", "DayExit", "PriceEnter", "PriceExit", "Position",
                                      "Profit/Loss", "ExitReason"], index=np.arange(len(contract_df)))
    equity_df = pd.DataFrame(columns=["Equity", "Returns", "PositionType",
                                      "CallIVPrem", "PutIVPrem"], index=contract_df.index)
    equity_df["CallIVPrem"] = contract_df["CALL_IV_PREM"]
    equity_df["PutIVPrem"] = contract_df["PUT_IV_PREM"]
    position = None
    entry_price = None
    average_historical_returns = get_historical_performance_of_contracts(ticker_name, contract_df.index[0])

    trade_num = 0
    for ind, row in contract_df.iterrows():
        exit_reason = np.nan
        current_position_return = mark_position_to_market(position, entry_price, row)
        equity_df["Returns"].loc[ind] = current_position_return
        if position is None:

            if row["CALL_IV_PREM"]>row["PUT_IV_PREM"]:
                skew = row["CALL_IV_PREM"]-row["PUT_IV_PREM"]
                if skew > threshold_delta_for_entry:
                    position = "LONG"
                    entry_price = row["UNDERLYING_LAST"]
                    trades_df["DayEnter"].iloc[trade_num] = ind
                    trades_df["PriceEnter"].iloc[trade_num] = entry_price
                    trades_df["Position"].iloc[trade_num] = position

            elif row["PUT_IV_PREM"]>row["CALL_IV_PREM"]:
                skew = row["PUT_IV_PREM"] - row["CALL_IV_PREM"]
                if skew > threshold_delta_for_entry:
                    position = "SHORT"
                    entry_price = row["UNDERLYING_LAST"]
                    trades_df["DayEnter"].iloc[trade_num] = ind
                    trades_df["PriceEnter"].iloc[trade_num] = entry_price
                    trades_df["Position"].iloc[trade_num] = position

        else:
            if position == "LONG":

                if current_position_return >= average_historical_returns["Max"]:
                    position = None
                    exit_reason = "TargetHit"
                    trades_df["ExitReason"].iloc[trade_num] = exit_reason
                    trades_df["DayExit"].iloc[trade_num] = ind
                    trades_df["PriceExit"].iloc[trade_num] = row["UNDERLYING_LAST"]
                    profit_loss = trades_df["PriceExit"].iloc[trade_num] - trades_df["PriceEnter"].iloc[trade_num]
                    trades_df["Profit/Loss"].iloc[trade_num] = profit_loss

                    trade_num += 1

                elif row["CALL_IV_PREM"]-row["PUT_IV_PREM"] <= threshold_for_reversal_exit:
                    position = None
                    exit_reason = "Reversal"
                    trades_df["ExitReason"].iloc[trade_num] = exit_reason
                    trades_df["DayExit"].iloc[trade_num] = ind
                    trades_df["PriceExit"].iloc[trade_num] = row["UNDERLYING_LAST"]
                    profit_loss = trades_df["PriceExit"].iloc[trade_num] - trades_df["PriceEnter"].iloc[trade_num]
                    trades_df["Profit/Loss"].iloc[trade_num] = profit_loss

                    trade_num += 1

            elif position == "SHORT":
                if current_position_return >= abs(average_historical_returns["Min"]):
                    position = None
                    exit_reason = "TargetHit"
                    trades_df["ExitReason"].iloc[trade_num] = exit_reason
                    trades_df["DayExit"].iloc[trade_num] = ind
                    trades_df["PriceExit"].iloc[trade_num] = row["UNDERLYING_LAST"]
                    profit_loss = -1*(trades_df["PriceEnter"].iloc[trade_num] - trades_df["PriceExit"].iloc[trade_num])
                    trades_df["Profit/Loss"].iloc[trade_num] = round(profit_loss, 4)

                    trade_num += 1

                elif row["PUT_IV_PREM"] - row["CALL_IV_PREM"] <= threshold_for_reversal_exit:
                    position = None
                    exit_reason = "Reversal"
                    trades_df["ExitReason"].iloc[trade_num] = exit_reason
                    trades_df["DayExit"].iloc[trade_num] = ind
                    trades_df["PriceExit"].iloc[trade_num] = row["UNDERLYING_LAST"]
                    profit_loss = -1*(trades_df["PriceEnter"].iloc[trade_num] - trades_df["PriceExit"].iloc[trade_num])
                    trades_df["Profit/Loss"].iloc[trade_num] = round(profit_loss, 4)

                    trade_num += 1

    if not position is None:
        position = None
        exit_reason = "Expiry"
        trades_df["ExitReason"].iloc[trade_num] = exit_reason
        trades_df["DayExit"].iloc[trade_num] = ind
        trades_df["PriceExit"].iloc[trade_num] = row["UNDERLYING_LAST"]
        if trades_df["Position"].iloc[trade_num] == "LONG":
            profit_loss = trades_df["PriceEnter"].iloc[trade_num] - trades_df["PriceExit"].iloc[trade_num]
        else:
            profit_loss = abs(trades_df["PriceExit"].iloc[trade_num] - trades_df["PriceEnter"].iloc[trade_num])
        trades_df["Profit/Loss"].iloc[trade_num] = round(profit_loss, 4)

    trades_df.dropna(inplace=True)
    trades_df["ReturnRate"] = trades_df["Profit/Loss"]/trades_df["PriceEnter"]
    return trades_df


def mark_position_to_market(position, entry_price, row):
    current_price = row["UNDERLYING_LAST"]
    if position is None:
        current_return = 0
    elif position == "LONG":
        current_return = (current_price - entry_price)/entry_price
    elif position == "SHORT":
        current_return = (entry_price - current_price) / entry_price
    return current_return


def get_historical_performance_of_contracts(ticker_name, final_date, num_contract_history=10):
    historical_performance = pd.DataFrame(columns=["Min", "Max", "Mean", "Std"], index=np.arange(num_contract_history))
    num_days_back = num_contract_history*14
    final_date = final_date.to_pydatetime()
    first_date = final_date - timedelta(num_days_back)
    final_date = final_date.strftime("%Y-%m-%d")
    first_date = first_date.strftime("%Y-%m-%d")
    data = yf.download(ticker_name, first_date, final_date, progress=False).reset_index(drop=False)
    data["Returns"] = data["Close"].pct_change()
    split_data = np.array_split(data, num_contract_history)
    for ind, contract in enumerate(split_data):
        contract.reset_index(drop=True)
        contract_cumulative_returns = contract["Close"].apply(lambda x: (contract["Close"].iloc[0]-x)/
                                                                        contract["Close"].iloc[0])
        contract["CumReturns"] = contract_cumulative_returns
        historical_performance["Min"].iloc[ind] = contract_cumulative_returns.min()
        historical_performance["Max"].iloc[ind] = contract_cumulative_returns.max()
        historical_performance["Mean"].iloc[ind] = contract_cumulative_returns.mean()
        historical_performance["Std"].iloc[ind] = contract_cumulative_returns.std()

    mean_performance = historical_performance.mean()
    return mean_performance


def get_position_daterange(trade: pd.Series, position_type="Long"):
    if trade.loc["Position"]==position_type:
        if trade["DayEnter"] == trade["DayExit"]:
            position_range = [trade["DayEnter"]]
        else:
            position_range = pd.date_range(trade["DayEnter"].to_pydatetime().strftime("%Y-%m-%d"),
                                           trade["DayExit"].to_pydatetime().strftime("%Y-%m-%d"))
    else:

        position_range = []
    return position_range


def run_experiment(path):
    distances_from_underlying_spot_all = [0.1, 0.25]
    distances_from_underlying = [distances_from_underlying_spot_all[0]]
    trades_df = calculate_strategy_performance(path, distances_from_underlying)
    ticker_name = os.path.basename(path).split("_")[0].upper()

    spy_data = yf.download(ticker_name, trades_df["DayEnter"].min().strftime("%Y-%m-%d"),
                           trades_df["DayExit"].max().strftime("%Y-%m-%d"), progress=False).reset_index(drop=False)
    spy_equity = spy_data["Adj Close"]/spy_data["Adj Close"].iloc[0]
    is_long_dates = trades_df.apply(lambda x: get_position_daterange(x, position_type="LONG"), axis=1)
    is_short_dates = trades_df.apply(lambda x: get_position_daterange(x, position_type="SHORT"), axis=1)
    _=0

    return




if __name__ == "__main__":
    path = "/Users/tom/Desktop/MBA/SemesterA/InvestmentTheory/Project/" \
           "Data/aapl_2016_2020.csv"
    rfr_path = "/Users/tom/Desktop/MBA/SemesterA/InvestmentTheory/Project/Data/DGS10.csv"
    out_dir = "/Volumes/Elements/Options_study/AAPL/"
    run_experiment(path)
    # main(path, rfr_path, out_dir, vis=False)


