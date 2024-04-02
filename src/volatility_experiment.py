import warnings
from tqdm import tqdm
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from src.utils import *
warnings.filterwarnings("ignore")




def main(path, rfr_path, out_dir, vis=False):
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

    for ind, row in tqdm(chosen_exp_date_strike_levels.iterrows(), desc="Iterating over Options Chain:"):
        underlying_exist=False
        date = row["EXPIRE_DATE"]
        # strike = row["STRIKE"]
        options_chain = df[df["EXPIRE_DATE"]==date]

        chosen_exp_date_strike_levels_at_date = chosen_exp_date_strike_levels[chosen_exp_date_strike_levels["EXPIRE_DATE"]==date]
        unique_strikes = chosen_exp_date_strike_levels_at_date["STRIKE"].unique().tolist()
        unique_levels = chosen_exp_date_strike_levels_at_date["LEVEL"].unique().tolist()
        if len(unique_levels)!=3:
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
        for strike, level in zip(unique_strikes, unique_levels):

            options_chain_strike = options_chain[options_chain["STRIKE"]==strike]
            options_chain_strike = options_chain_strike.set_index("DTE", drop=False).sort_index() # change to quote date
            if len(volatility_df) == 0:
                volatility_df.index = options_chain_strike.index
                volatility_df["UNDERLYING_LAST"] = options_chain_strike["UNDERLYING_LAST"]
                underlying_plot_df[f"{str(date)}_AT_{at_money_strike_price}"] = volatility_df["UNDERLYING_LAST"]
                volatility_df["DTE"] = options_chain_strike["DTE"].astype(int)
                volatility_df.set_index("DTE", drop=False, inplace=True)
                volatility_df["PCT_CHANGE_UNDERLYING"] = volatility_df["UNDERLYING_LAST"].pct_change()

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

    for sent_col, call_col, put_col, under_col in zip(sentiment_plot_df.columns, call_plot_df.columns,
                                                      put_plot_df.columns, underlying_plot_df.columns):
        path = os.path.join(out_dir, f"Expiry_{under_col}.png")
        at_money_strike_price = float(under_col.split("AT_")[-1])
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, sharex=True)
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
        # ax3.set_xticks(put_plot_df.sort_index(ascending=False).index.values)
        ax3.invert_xaxis()
        plt.tight_layout()


        # plt.show()
        plt.savefig(path)

        # .interpolate(method='linear').plot()
        _=0



if __name__ == "__main__":
    path = "/Users/tom/Desktop/MBA/SemesterA/InvestmentTheory/Project/" \
           "Data/aapl_2016_2020.csv"
    rfr_path = "/Users/tom/Desktop/MBA/SemesterA/InvestmentTheory/Project/Data/DGS10.csv"
    out_dir = "/Volumes/Elements/Options_study/AAPL/"
    main(path, rfr_path, out_dir)
