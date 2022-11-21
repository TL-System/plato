import pickle
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

atten_data = []
key_map = {0:'p', 1:'q', 2:'v', 3:'atten'}
with open('atten_data.pickle', 'rb') as f:
    while True:
        try:
            atten_data.append(pickle.load(f))
        except EOFError:
            break

def convert_to_numpy(x):
    return x.detach().cpu().numpy()

def cross_bs_atten_map(key=3):
    for i in range(32):
        data_np = convert_to_numpy(atten_data[i][key].mean(dim=0).mean(dim=0))
        max_val = data_np.max()
        sns.heatmap(data_np / max_val * 1250 + 50, cbar=False, cmap='Blues',vmin=0, vmax=250, center=110)

        plt.xticks([])
        plt.yticks([])
        plt.savefig('./atten_visualize/layer_%s_head_mean_bs_mean_re.jpg' % i, bbox_inches='tight', pad_inches=0.)
        plt.close()

cross_bs_atten_map()
