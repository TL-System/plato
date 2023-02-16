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
        "xtick.labelsize": "small",
        "axes.labelsize": "small",
        "xtick.labelsize": 9,
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
            exploration_rate = [0.1]

            counter = 1
            for temp in all_selection[20:]:
                counter += 1

                if int(temp) not in num_list:
                    num_list.append(int(temp))

                if counter % 10 == 0:
                    exploration_rate.append(len(num_list) / 200.0)
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
print("y: ", y_value["FedBuff"])

plt.plot(x_value["Polaris"], y_value["Polaris"], label="Polaris")
plt.plot(x_value["Pisces"], y_value["Pisces"], label="Pisces")
plt.plot(x_value["Oort"], y_value["Oort"], label="Oort")
plt.plot(x_value["FedBuff"][:82], y_value["FedBuff"][:82], label="FedBuff")

figure_file_name = "exploration_rate_all.pdf"
plt.legend()
plt.savefig(figure_file_name)
