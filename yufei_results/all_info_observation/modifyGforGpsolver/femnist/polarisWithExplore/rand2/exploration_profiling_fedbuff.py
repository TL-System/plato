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

with open("client_selection_fedbuff.txt") as f:
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
print("length of exploration list: ", len(exploration_rate))
print("exploreation list: ", exploration_rate)


filename_temp = "./fedbuff.csv"
df_temp = pd.read_csv(filename_temp)

x_temp = df_temp["elapsed_time"]

df_explore = pd.DataFrame(
    {"Elapsed time (s)": x_temp, "Exploration rate (%)": exploration_rate}
)  # [x_temp, exploration_rate],columns=["Elapsed time (s)", "Exploration rate (%)"])#.transpose()
# df_explore.columns = ["Elapsed time (s)", "Exploration rate (%)"]


g = sns.lineplot(x="Elapsed time (s)", y="Exploration rate (%)", data=df_explore)
# ax.xlabel('clinet_id')
# ax.ylabel('# of being selected')
# ax.title('async_selected_clients_distribution_rand1_round250')
figure_file_name = "exploration_rate_fedbuff.pdf"
plt.savefig(figure_file_name)
