import numpy as np
import matplotlib.pyplot as plt


def plot_box_and_whisker(all_dfs, columns):
    fig, axes = plt.subplots(nrows=9, sharex=True, figsize=(45, 15))
    columns_to_plot = ["C_DELTA_AT_LEVEL_0.0",
                       "C_DELTA_AT_LEVEL_0.1",
                       "C_DELTA_AT_LEVEL_0.25",
                       "C_GAMMA_AT_LEVEL_0.0",
                       "C_GAMMA_AT_LEVEL_0.1",
                       "C_GAMMA_AT_LEVEL_0.25",
                       "OPTION_PRICE_AT_LEVEL_0.0",
                       "OPTION_PRICE_AT_LEVEL_0.1",
                       "OPTION_PRICE_AT_LEVEL_0.25"]
    underlying_col_index = columns.values.tolist().index("UNDERLYING_LAST")
    for ax, col in zip(axes, columns_to_plot):
        col_index = columns.values.tolist().index(col)
        all_ax_data = []
        lag_amount_list = []
        for key, data in all_dfs.items():
            lag_amount = int(key.split("_")[-1])
            lag_amount_list.append(lag_amount)
            relevant_data = data[:, underlying_col_index, col_index]
            all_ax_data.append(relevant_data)
        # all_ax_data = np.concatenate(all_ax_data, axis=1)
        ax.boxplot(all_ax_data)
        ax.set_xticks(np.arange(len(lag_amount_list)), lag_amount_list)
        ax.set_xlabel("Lag Time in Days")
        ax.set_ylabel(f"Correlation Distribution ")
        ax.title.set_text(f'Correlation Distribution between Underlying Price and {col}')
    plt.tight_layout()
    plt.show()
