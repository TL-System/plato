import numpy as np
import matplotlib.pyplot as plt

testset = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10", "8", "1"]
print(testset)
fig, ax = plt.subplots()
ax.hist(testset, bins=10)
# ax.xlabel('clinet_id')
# ax.ylabel('# of being selected')
# ax.title('async_selected_clients_distribution_rand1_round250')
figure_file_name = "test_distribution.pdf"
plt.savefig(figure_file_name)
