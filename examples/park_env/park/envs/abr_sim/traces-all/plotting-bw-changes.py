import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#log_file = pd.read_table('./ferry.nesoddtangen-oslo-report.2011-01-29_1800CET.log', header = None) # BW: 0 to 7, delta 1 to 2.2
#log_file = pd.read_table('./metro.kalbakken-jernbanetorget-report.2010-09-13_1003CEST.log', header = None) # BW: 0 to 2.3, delta 1 to 1.35

#log_file = pd.read_table('./norway_car_2', header = None) # BW: 0.25 to 2.25, delta 1 to 7

#log_file = pd.read_table('./norway_car_8', header = None) # BW: 1 to 3.5, delta 0.25 to 2.25

#log_file = pd.read_table('./norway_tram_12', header = None) # BW: 0.4 to 1.4, delta 1 to 5

log_file = pd.read_table('./trace_5294_http---www.youtube.com', header = None) # BW: 0 to 0.2, delta 5

log_file = pd.read_table('./trace_28838_http---www.amazon.com', header = None) # BW: 0 to 4, delta 5

log_file = pd.read_table('./trace_28919_http---www.google.com-mobile-', header = None) # BW: 0.1 to 3.5, delta 5

log_file = pd.read_table('./trace_925800_http---www.ebay.com', header = None) # BW: 0 to 0.9, delta 5



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
