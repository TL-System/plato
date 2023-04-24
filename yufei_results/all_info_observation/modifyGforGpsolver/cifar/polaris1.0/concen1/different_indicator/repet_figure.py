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

# sns.set_context("poster", font_scale=5, rc={"lines.linewidth": 2.5})
sns.set_theme(style="whitegrid")

sns.set_context(
    "talk",
    rc={
        "legend.fontsize": "small",
        "axes.labelsize": 13,
        # "xtick.labelsize": "small",
        # "axes.labelsize": "small",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    },
)


# create a new csv file
# with open("collect_results.csv", "w", newline="") as file:
#    writer = csv.writer(file)
#    writer.writerow(["elapsed_time", "accuracy"])

x_all = []
y_all = []
z_all = []


# Folder Path
path = "./"

# Change the directory
os.chdir(path)


# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    print("listing the file: ", file)


# iterate through all files in this directory
for file_name in os.listdir():
    if file_name.endswith(".csv"):

        df_temp = pd.read_csv(file_name)

        # perform interpolation
        x_temp = df_temp["elapsed_time"]
        y_temp = df_temp["accuracy"]
        f_temp = interpolate.interp1d(x_temp, y_temp)

        x_min = x_temp.min()
        x_max = x_temp.max()

        x_new = np.arange(max(23, np.ceil(x_min)), min(x_max, 1650), 40)
        y_new = f_temp(x_new) * 100

        # write into lists
        x_all.extend(x_new)
        y_all.extend(y_new)
        z_all.extend([file_name[:-4]] * len(x_new))

# combine all interpolate results into one dataframe
df_all = pd.DataFrame([x_all, y_all, z_all]).transpose()
df_all.columns = ["Elapsed time (s)", "Accuracy (%)", "Method"]

# save into a .csv file
# df_all.to_csv("interpolate_results_all.csv", index=False)


# draw figures directly from df
g = sns.lineplot(
    x="Elapsed time (s)", y="Accuracy (%)", data=df_all, hue="Method", style="Method"
)
g.legend_.set_title(None)

# save figure as pdf file
# plt.show()
plt.savefig("estimators.pdf")
