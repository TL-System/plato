from http import client
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# my_file = Path("/path/to/file")
"""
Read .out file and draw clients distribution figure
"""
new_outfile = open("client_selection_polaris.txt", "w")
new_outfile.close()
with open("async_CIFAR100_200_20_epoch5_3000_rand4.out") as f:
    for line in f.readlines():

        if "Selected clients" in line:
            loc = line.find("Selected clients")
            clients = line[loc + 19 :]
            loc_end = clients.find("]")
            # output file saves selected clients set
            outf = open("client_selection_polaris.txt", "a")
            # print("".join(clients[0:loc_end].split(",")))
            outf.write("".join(clients[0:loc_end].split(",")))
            outf.write(" ")
            outf.close()

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
        accumulator_list = [0]
        for pro in np.flip(counter_pro_sorted):
            accumulator += pro
            accumulator_list.append(accumulator)

        print(accumulator_list)

        sns.lineplot(data=accumulator_list)  # , stat="proportion")
        # sns.barplot(counter_pro_sorted, bins=200)

        # print(np.sort(counter_prob))
        # print("skewness is: ", stats.skew(num_list))
        # sns.histplot(data=num_list, stat="probability", bins=200)

        # fig, ax = plt.subplots()
        # ax.hist(num_list, bins=200)
        # ax.xlabel('clinet_id')
        # ax.ylabel('# of being selected')
        # ax.title('async_selected_clients_distribution_rand1_round250')
        figure_file_name = "distribution_polaris.pdf"
        plt.savefig(figure_file_name)
