from http import client
import numpy as np
import matplotlib.pyplot as plt
#my_file = Path("/path/to/file")
"""
Read .out file and draw clients distribution figure
"""
new_outfile = open("sirius_selected_clients.txt", "w")#open("async_selected_clients.txt", "w")
new_outfile.close()
#with open(
#        'async_selection_FEMNIST_lenet5_avg9_rand1_round250_multiplyPi_print.out'
#) as f:
with open('Sirius_CIFAR_resnet18_rand1_concen1.out') as f:
    for line in f.readlines():

        if 'Selected clients' in line:
            loc = line.find('Selected clients')
            clients = line[loc + 19:]
            loc_end = clients.find(']')
            # output file saves selected clients set
            outf = open("sirius_selected_clients.txt", "a")#open("async_selected_clients.txt", "a")
            outf.write(clients[0:loc_end])
            outf.write(' ')
            outf.close()

#with open("async_selected_clients.txt") as f:
with open("sirius_selected_clients.txt") as f:
    client_set = []
    for line in f.readlines():
        print(line.split())
        fig, ax = plt.subplots()
        ax.hist(line.split(), 500)
        #ax.xlabel('clinet_id')
        #ax.ylabel('# of being selected')
        #ax.title('async_selected_clients_distribution_rand1_round250')
        figure_file_name ='sirius_selected_clients_distribution.pdf' #'async_selected_clients_distribution.pdf'
        plt.savefig(figure_file_name)
