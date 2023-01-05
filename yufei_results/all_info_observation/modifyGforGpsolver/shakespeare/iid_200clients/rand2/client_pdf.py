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
sns.set_theme(style="darkgrid")
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
            accumulator += pro
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
            accumulator += pro
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
            accumulator += pro
            accumulator_list3.append(accumulator)

df_all = pd.DataFrame(
    [accumulator_list, accumulator_list2, accumulator_list3]
).transpose()
df_all.columns = ["FedBuff", "Polaris", "Pisces"]
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
figure_file_name = "prodf.pdf"
plt.savefig(figure_file_name)
