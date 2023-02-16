from http import client
import numpy as np
import matplotlib.pyplot as plt

# my_file = Path("/path/to/file")
"""
Read .out file and draw clients distribution figure
"""
new_outfile = open("client_selection_pisces.txt", "w")
new_outfile.close()
with open("pisces_cifar10_rand3_200_20_5epoch_2000.out") as f:
    for line in f.readlines():

        if "Selected clients" in line:
            loc = line.find("Selected clients")
            clients = line[loc + 19 :]
            loc_end = clients.find("]")
            # output file saves selected clients set
            outf = open("client_selection_pisces.txt", "a")
            # print("".join(clients[0:loc_end].split(",")))
            outf.write("".join(clients[0:loc_end].split(",")))
            outf.write(" ")
            outf.close()

with open("client_selection_pisces.txt") as f:
    client_set = []
    num_list = []
    for line in f.readlines():
        # print(line.split())
        counter = 0
        for temp in line.split():
            if int(temp) <= 200:
                num_list.append(int(temp))
                if int(temp) == 200:
                    counter += 1
        # print(num_list)
        print(len(num_list))
        fig, ax = plt.subplots()
        ax.hist(num_list, bins=200)
        # ax.xlabel('clinet_id')
        # ax.ylabel('# of being selected')
        # ax.title('async_selected_clients_distribution_rand1_round250')
        figure_file_name = "distribution_pisces     .pdf"
        plt.savefig(figure_file_name)
