import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

log_file = pd.read_table('./norway_bus_16', header = None) # BW: 0 to 0.9, delta 5



delta = []
for i in range(len(log_file[0][:]) - 1):
    delta.append(log_file[0][i+1] - log_file[0][i])

delta_time = pd.DataFrame(delta, columns = ['delta_time'])
bw_list = log_file[1][:].values.tolist()
bw = pd.DataFrame(bw_list, columns = ['bw'])

print(bw.size)
print(delta_time.size)
result = pd.concat([delta_time, bw], axis=1)
print(result.head())
sns.scatterplot(data = result, x = "delta_time", y = "bw")
plt.show()
