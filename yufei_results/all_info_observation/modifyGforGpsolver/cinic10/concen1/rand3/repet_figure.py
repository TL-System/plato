import seaborn as sns
import csv
import os
from typing import Dict, List, Any
from matplotlib import markers

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import interpolate
import numpy as np

# from plato.config import Config
# plt.style.use(["ipynb", "use_mathtext", "colors10-ls"])

import pandas as pd

sns.set_theme(style="darkgrid")

# create a new csv file
# with open("collect_results.csv", "w", newline="") as file:
#    writer = csv.writer(file)
#    writer.writerow(["elapsed_time", "accuracy"])

x_collect = []
y_collect = []

# input results from rands file
for i in range(3):

    filename_temp = "fedbuff.csv"  # "./rand" + str(i + 1) + "/pisces.csv"
    df_temp = pd.read_csv(filename_temp)

    # perform interpolation
    x_temp = df_temp["elapsed_time"]
    y_temp = df_temp["accuracy"]
    f_temp = interpolate.interp1d(x_temp, y_temp)

    x_new = np.arange(30, 2000, 10)
    y_new = f_temp(x_new)

    x_collect.extend(x_new)
    y_collect.extend(y_new)

    # write into dataframe
df = pd.DataFrame([x_collect, y_collect]).transpose()
df.columns = ["elapsed_time", "accuracy"]

# save interpolate results into csv file
df.to_csv("collect_results.csv", index=False)

# draw figures directly from df
sns.lineplot(x="elapsed_time", y="accuracy", data=df)
plt.show()

# save figure as pdf file


# sns.lineplot(x=xra, y="accuracy", data=df)
# plt.show()
