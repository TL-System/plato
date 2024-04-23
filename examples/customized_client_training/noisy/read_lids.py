import torch
import os
import matplotlib.pyplot as plt

values = []
# List all files in the directory
cache_root =  os.path.expanduser("~/.cache")
lid_file = f"856755-2-alphas.pt"
lid_file = os.path.join(cache_root, lid_file)
lid_dict =  torch.load(lid_file)

i = 0
print("len of dic: ", len(lid_dict.items()))

#plt.figure(figsize=(10, 6))  # Initialize the figure

for key, value in lid_dict.items():
    # print(key, value)
    # i += 1
    # if i >= 5:
    #     break
    plt.plot(value)

plt.title('round vs lid')
plt.xlabel('round')
plt.ylabel('lid')
plt.grid(True)
plt.legend()  # Show legend to identify lines
# Save the figure as a PDF file
plt.savefig('alpha_per_sample.pdf', bbox_inches='tight')