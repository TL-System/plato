from http import client
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# exploration rate v.s. physical time
sns.set_theme(style="whitegrid")
sns.set_context(
    "talk",
    rc={
        "legend.fontsize": "large",
        # "axes.labelsize": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    },
)
x_value = {}
y_value = {}
for algorithm in ["Polaris", "Pisces", "Oort", "FedBuff"]:

    selection_file_name = "client_selection_" + algorithm.lower() + ".txt"

    with open(selection_file_name) as f:
        client_set = []
        num_list = []
        for line in f.readlines():
            print(type(line.split()))
            all_selection = line.split()

            first_round = all_selection[0:20]
            for temp in first_round:
                num_list.append(int(temp))
            exploration_rate = [10]

            counter = 1
            for temp in all_selection[20:]:
                counter += 1

                if int(temp) not in num_list:
                    num_list.append(int(temp))

                if counter % 10 == 0:
                    exploration_rate.append(100 * len(num_list) / 200.0)
    # print("length of exploration list: ", len(exploration_rate))
    # print("exploreation list: ", exploration_rate)

    filename_csv = "./" + algorithm.lower() + ".csv"
    df_temp = pd.read_csv(filename_csv)

    x_value[algorithm] = df_temp["elapsed_time"].values.tolist()
    y_value[algorithm] = exploration_rate

    # df_explore = pd.DataFrame(
    #    {"Elapsed time (s)": x_temp, "Exploration rate (%)": exploration_rate}
    # )  # [x_temp, exploration_rate],columns=["Elapsed time (s)", "Exploration rate (%)"])#.transpose()
    # df_explore.columns = ["Elapsed time (s)", "Exploration rate (%)"]

    # plt(x_temp, y=exploration_rate,label=method)
# ax.xlabel('clinet_id')
# ax.ylabel('# of being selected')
# ax.title('async_selected_clients_distribution_rand1_round250')
# print("y: ", y_value["Polaris"])

plt.plot(x_value["Polaris"], y_value["Polaris"], label="Polaris")
plt.plot(
    x_value["Pisces"][:113], y_value["Pisces"][:113], label="Pisces", linestyle="--"
)
plt.plot(x_value["Oort"][:70], y_value["Oort"][:70], label="Oort", linestyle=":")
plt.plot(
    x_value["FedBuff"][:65], y_value["FedBuff"][:65], label="FedBuff", linestyle="-."
)
plt.xlabel("Elapsed time (s)")
plt.ylabel("Exploration rate (%)")
figure_file_name = "cifar10_exploration_concen1.pdf"
plt.legend()
plt.savefig(figure_file_name)
