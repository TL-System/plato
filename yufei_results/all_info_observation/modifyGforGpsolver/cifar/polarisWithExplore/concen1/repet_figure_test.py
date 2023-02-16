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

# sns.set_context("poster", font_scale=5, rc={"lines.linewidth": 5.5})
sns.set_theme(style="whitegrid")
sns.set_context(
    "talk",
    rc={
        "legend.fontsize": "large",
        # "axes.labelsize": 12,
        "xtick.labelsize": "small",
        # "axes.labelsize": "small",
        "xtick.labelsize": 10,
    },
)  # , rc={"lines.linewidth": 1.5})

# create a new csv file
# with open("collect_results.csv", "w", newline="") as file:
#    writer = csv.writer(file)
#    writer.writerow(["elapsed_time", "accuracy"])

x_all = []
y_all = []
z_all = []

# input results from rands file
for method_name in ["Pisces", "Polaris", "fedbuff"]:

    x_collect = []
    y_collect = []
    z_collect = []

    for i in range(5):

        filename_temp = "./rand" + str(i + 1) + "/" + method_name + ".csv"
        df_temp = pd.read_csv(filename_temp)

        # perform interpolation
        x_temp = df_temp["elapsed_time"]
        y_temp = df_temp["accuracy"]
        f_temp = interpolate.interp1d(x_temp, y_temp)

        x_min = x_temp.min()
        x_max = x_temp.max()

        x_new = np.arange(max(26, np.ceil(x_min)), min(x_max, 7000), 20)
        y_new = f_temp(x_new)

        x_collect.extend(x_new)
        y_collect.extend(y_new)
        z_collect.extend([method_name] * len(x_new))

    # write into dataframe
    x_all.extend(x_collect)
    y_all.extend(y_collect)
    z_all.extend(z_collect)
    # print(z_all)

    df = pd.DataFrame([x_collect, y_collect]).transpose()
    df.columns = ["elapsed_time", "accuracy"]

    # save interpolate results into csv file
    saving_name = "interpolate_results_" + method_name + ".csv"
    df.to_csv(saving_name, index=False)

# combine all interpolate results into one dataframe
df_all = pd.DataFrame([x_all, y_all, z_all]).transpose()
df_all.columns = ["Elapsed time", "Accuracy (%)", "Method"]
df_all.to_csv("interpolate_results_all.csv", index=False)

"""
df_pi = pd.read_csv("interpolate_results_pisces.csv")
df_po = pd.read_csv("interpolate_results_polaris.csv")

df_all = pd.DataFrame()
# df_all.columns = ["elapsed_time", "accuracy_pisces", "accuracy_polaris"]

df_all["elapsed_time"] = df_pi["elapsed_time"].copy()
df_all["accuracy_pisces"] = df_pi["accuracy"].copy()
df_all["method"] = df_pi["method"].copy()


df_all["accuracy_polaris"] = df_po["accuracy"].copy()

df_all.to_csv("interpolate_results_all.csv", index=False)
"""

# draw figures directly from df
sns.lineplot(
    x="Elapsed time", y="Accuracy (%)", data=df_all, hue="Method", style="Method"
)


# save figure as pdf file
# plt.show()
plt.savefig("repet_result_scale.pdf")
