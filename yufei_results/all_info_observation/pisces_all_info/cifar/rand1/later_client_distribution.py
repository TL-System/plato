import matplotlib.pyplot as plt
import numpy as np

selection_count = np.zeros(500)
# with open("async_selected_clients.txt") as f:
print(selection_count[2])
with open("pisces_selected_clients.txt") as f:
    client_set = []
    for line in f.readlines():

        print(line.split(", ")[:500])
        fig, ax = plt.subplots()
        ax.hist(line.split(", ")[:500], 500)
        # ax.xlabel('clinet_id')
        # ax.ylabel('# of being selected')
        # ax.title('async_selected_clients_distribution_rand1_round250')
        figure_file_name = "pisces_selected_clients_distribution_later.pdf"  #'async_selected_clients_distribution.pdf'
        plt.savefig(figure_file_name)
        for item in line.split(", ")[:500]:

            print(type(int(item)))

            selection_count[item] += 1.0
    print("selection count: ", selection_count)
