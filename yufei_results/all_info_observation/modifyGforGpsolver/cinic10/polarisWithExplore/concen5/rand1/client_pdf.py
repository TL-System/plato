from http import client
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# my_file = Path("/path/to/file")
"""
Read .out file and draw clients distribution figure
"""
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
with open("client_selection_fedbuff.txt") as f:
    client_set = []
    num_list = []
    counter_array = np.zeros(200)
    for line in f.readlines():
        print(line.split())
        counter = 0
        for temp in line.split():
            if int(temp) <= 200:
                num_list.append(int(temp))
                if int(temp) == 200:
                    counter += 1

        print(len(num_list))

        print("skewness is: ", stats.skew(num_list))
        for item in num_list:
            counter_array[item - 1] += 1
        counter_pro_sorted = np.sort(counter_array) / len(num_list)

        accumulator = 0
        accumulator_list = [0]
        for pro in np.flip(counter_pro_sorted):
            accumulator += pro*100
            accumulator_list.append(accumulator)

        # print(accumulator_list)
with open("client_selection_polaris.txt") as f:
    client_set = []
    num_list = []
    counter_array = np.zeros(200)
    for line in f.readlines():
        print(len(line.split()))
        counter = 0
        for temp in line.split():
            if int(temp) <= 200:
                num_list.append(int(temp))
                if int(temp) == 200:
                    counter += 1
        # print(num_list)
        for item in num_list:
            counter_array[item - 1] += 1
        counter_pro_sorted = np.sort(counter_array) / len(num_list)

        accumulator = 0
        accumulator_list2 = [0]
        for pro in np.flip(counter_pro_sorted):
            accumulator += pro*100
            accumulator_list2.append(accumulator)

with open("client_selection_pisces.txt") as f:
    client_set = []
    num_list = []
    counter_array = np.zeros(200)
    for line in f.readlines():
        print(len(line.split()))
        counter = 0
        for temp in line.split():
            if int(temp) <= 200:
                num_list.append(int(temp))
                if int(temp) == 200:
                    counter += 1
        # print(num_list)
        for item in num_list:
            counter_array[item - 1] += 1
        counter_pro_sorted = np.sort(counter_array) / len(num_list)

        accumulator = 0
        accumulator_list3 = [0]
        for pro in np.flip(counter_pro_sorted):
            accumulator += pro*100
            accumulator_list3.append(accumulator)

with open("client_selection_oort.txt") as f:
    client_set = []
    num_list = []
    counter_array = np.zeros(200)
    for line in f.readlines():
        print(len(line.split()))
        counter = 0
        for temp in line.split():
            if int(temp) <= 200:
                num_list.append(int(temp))
                if int(temp) == 200:
                    counter += 1
        # print(num_list)
        for item in num_list:
            counter_array[item - 1] += 1
        counter_pro_sorted = np.sort(counter_array) / len(num_list)

        accumulator = 0
        accumulator_list4 = [0]
        for pro in np.flip(counter_pro_sorted):
            accumulator += pro*100
            accumulator_list4.append(accumulator)

df_all = pd.DataFrame(
    [accumulator_list2, accumulator_list3, accumulator_list4, accumulator_list]
).transpose()
df_all.columns = [
    "Polaris",
    "Pisces",
    "Oort",
    "FedBuff",
]
df_all.to_csv("client_pdf.csv", index=False)
sns.lineplot(
    data=df_all
    # palette="flare",
    # hue_norm=mpl.colors.LogNorm(),
)

# sns.lineplot(data=accumulator_list)
# sns.lineplot(data=accumulator_list2)
# sns.lineplot(data=accumulator_list3)
# sns.ecdfplot(data=counter_array_sorted)
# sns.histplot(data=num_list, stat="probability", kde=True, bins=200)
# fig, ax = plt.subplots()
# ax.hist(num_list, bins=200, kde=True)
# ax.xlabel('clinet_id')
# ax.ylabel('# of being selected')
# ax.title('async_selected_clients_distribution_rand1_round250')
plt.ylabel("Cumulative distribution (%)")
plt.xlabel("Client count (#)")
figure_file_name = "cinic10_distribution_concen5.pdf"
plt.savefig(figure_file_name)
