## Custom Plot Documentation Steps
**1.** run ```python all_server_rewards.py``` \
This consolidates all results across seeds into one csv file \
**2.** run ```python culsum_and_avg_plot.py``` \
This plots the culmunative average sum across seeds between different algorithms into one pdf \
**3.** run ```python seed_avg_and_plot_rewards.py``` \
This plots the average reward across seeds between different algorithms for different tasks into one pdf

## all_server_rewards.py
There are a total of four lists in this file that should be changed if one wants to consolidate more/less files.
### results_folders:
This list should be filled with the directories of the results generated. The format for this would be "results_insertAlgorithmNameHere_aggregate_seed_"
### file_names:
This list should be filled with the file name of inside the folders. It should be "filename" (exclude.csv)
### save_files:
This is where we want to save the combined dataframe. It should be "filename" (exclude.csv). It will automatically append _ALL.csv to it
### seeds:
This list should be filled with the seed numbers
### Important things to keep in mind:
* The folder name at an index i should match with the file name at an index i.
* An example: if I wanted to read from a folder called "results_critic_grad_lamda2_aggregate_seed_5" and the file name in this folder called "A2C_RL_SERVER_PERCENTILE_AGGREGATE.csv" and I want to save it in a file called "A2C_RL_SERVER_LAMDA2_ALL"
   ```shell
     The respective lists should be: results_folders = ["results_critic_grad_lamda2_aggregate_seed_"], 
     file_names = ["A2C_RL_SERVER_PERCENTILE_AGGREGATE"], save_files = ["A2C_RL_SERVER_LAMDA2"]
     Notice they are all at the same index i = 0
    ```

## culsum_and_avg_plot.py
There are a total of four lists in this file that should be changed if one wants to consolidate more/less files.
### experiments:
This list should be filled with the names of the csv files you want to read from. This reads from the csv files that were combined from the script before. This list should identically match the ```save_files``` list from above
### target_files:
This list should be filled with the file name of the csv file name we want to save to. It should be "filename" (exclude.csv)
### exp_names:
This should be the names of the algorithms 
### seeds:
This list should be filled with the seed numbers
### Important things to keep in mind:
* The folder name at an index i should match with the file name at an index i.
* Using same example from above: if I wanted to read from a file called "A2C_RL_SERVER_LAMDA2" (it's where all the data is saved into one csv file from all_server_rewards.py) and I want to save it in a csv file called "CULSUM_AVG_CASCADE" with the name of the experiment called Cascade
    ```shell
     The respective lists should be: experiments = ["A2C_RL_SERVER_LAMDA2"], target_files = ["CULSUM_AVG_CASCADE"], 
     exp_names = ["Cascade"]
     Notice they are all at the same index i = 0
    ```

## seed_avg_and_plot_rewards.py
There are a total of five lists in this file that should be changed if one wants to consolidate more/less files.
### experiments:
Exact same logic as above
### target_files:
This list should be filled file name we want to save to. It should be "filename.csv" 
### exp_task_names:
This should be the names of the algorithms 
### file_task_names:
This list should be filled with the number of tasks we have
### seeds:
This list should be filled with the seed numbers
### Important things to keep in mind:
* The folder name at an index i should match with the file name at an index i.
* Using same example from above: if I wanted to read from a file called "A2C_RL_SERVER_LAMDA2" (it's where all the data is saved into one csv file from all_server_rewards.py) and I want to save it in a csv file called "AVG_CASCADE" with the six different tasks
   ```shell
     The respective lists should be: experiments = ["A2C_RL_SERVER_LAMDA2"], target_files = ["AVG_CASCADE.csv"], 
    exp_task_names = ["Cascade"], file_task_names = ["task1","task2","task3","task4","task5","task6"]
     Notice they are all at the same index i = 0 and file_task_names has the list of total tasks
    ```

## For culmunative sum with one legend (culsum_clients_as_legend.py & culsum_experiments_as_legend.py)
Fill in the client list with the respective amount of clients
Same index rules and lists as culsum_and_avg_plot.py

### For the legend being clients
* The only thing that changes is the **ordering** of the lists elements. If there are clients being 6, 12, and 18.. the experiments list should be ordered ```[experiment same algorithm with 6, experiments same algorithm with 12, experiments same algorithm with 18, same idea for the next algorithm].``` This algorithm will be the same between 6, 12, 18. Keep in mind that if the experiments list is ordered this way the target_files list should be ordered in this way too. 

Example:
![legend_clients](https://user-images.githubusercontent.com/70533174/183153228-00cef3cb-553d-41ae-b92a-f9db0446d97f.png)

### For the legend being experiments
* The only thing that changes is the **ordering** of the lists elements. If there are clients being 6, 12, and 18.. the experiments list should be ordered ```[experiment diff algorithm with 6, experiment diff algorithm with 6, experiment diff algorithm with 6, ... same idea for 12 and 18].``` This algorithm will be the same between 6, 12, 18. Keep in mind that if the experiments list is ordered this way the target_files list should be ordered in this way too. 

Example:
![legend_experiments](https://user-images.githubusercontent.com/70533174/183153267-6d2a5e0a-dacf-458b-af9c-ca786f646a2a.png)
