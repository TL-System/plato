from audioop import avg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot(csv_to_plot, csv_std_min, csv_std_max, file_task_names, exp_task_name, num_tasks, num_experiments):
    #We start plotting here
    TSBOARD_SMOOTHING = 0.4
    INTERVAL = 5 #interval for how much you want to space out the grpah
    sns.color_palette("flare", as_cmap=True)

    csv = csv_to_plot

    sns.set(font_scale=2)
    sns.set_style("whitegrid")
    f, ax1 = plt.subplots(figsize=(8.5,5))

    csv = csv.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
    csv_std_min = csv_std_min.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()
    csv_std_max = csv_std_max.ewm(alpha=(1 - TSBOARD_SMOOTHING)).mean()

    marker_line = [(None, '-'), ('s', '-'), ('^', '-'), ('D', '-'), ('o', '-'), ('8', ':'), ('^', ':'), ('D', ':'),  ('s', ':'), (None, ':')]

    #Loop through the tasks
    #If there are n tasks & m experiments and nxm columns, # of tasks = nxm//m
    for i in range((len(csv.columns)//(num_experiments))):
        #retrive the column
        rewards = csv[i]
        rewards.index = [i+1 for i in range(len(rewards))]
        exp_task_index = 0
        #Axes for the first algorithm
        marker, line = marker_line[i%len(marker_line)]
        ax1 = sns.lineplot(data=rewards, linewidth=2.5, label= exp_task_name[exp_task_index], marker=marker, markersize=8,  linestyle=line)
        #std deviation
        #ax1.fill_between(rewards.index, csv_std_max[i].values, csv_std_min[i].values, alpha=0.5)
       
        #Axes for the next nth algorithm, if more than 2 algorithms, length csv.columns increases
        for j in range(i+num_tasks, len(csv.columns), num_tasks):
            rewards = csv[j]
            exp_task_index += 1
            rewards.index = [i+1 for i in range(len(rewards))]
            marker, line = marker_line[j%len(marker_line)]
            ax1 = sns.lineplot(data=rewards, linewidth=2.5, label= exp_task_name[exp_task_index], marker=marker, markersize=8,  linestyle=line)
            ax1.fill_between(rewards.index, csv_std_max[j].values, csv_std_min[j].values, alpha=0.5)

        ax1.set_xlabel("Training round", fontstyle='normal')
        ax1.set_ylabel(f'Average reward for {file_task_names[i]}')
        ax1.tick_params(axis='y')

        ax1.xaxis.set_major_formatter(plticker.FormatStrFormatter('%d'))
        loc = plticker.MultipleLocator(base=INTERVAL) # this locator puts ticks at regular intervals of two
        ax1.xaxis.set_major_locator(loc)

        plt.xlim(1, len(csv))

        pdf_name = file_task_names[i]
        if " " in pdf_name:
            pdf_name = pdf_name.replace(" ", "")

        #pdf_name = pdf_name + "_fedadp"
        #save each task in their own file
        f.savefig(f'{pdf_name}.pdf', bbox_inches='tight')

        #We clear the axis so that it doesn't superimpose tasks 2 & tasks3 on top of tasks1
        ax1.clear()

def combine_n_dataframes(csv_list, csv_min_list, csv_max_list, num_tasks):
    
    #put the list dataframes into one single big dataframe
    csv = pd.concat(csv_list, axis=1, join='inner')
    csv_min = pd.concat(csv_min_list, axis=1, join='inner')
    csv_max = pd.concat(csv_max_list, axis=1, join='inner')

    """For every n task & m experiments, there are nxm columns and also n x m standard deviations, 
    so their number of columns are equal, therefore we can just use length csv.columns"""
    #For renaming the columns
    for i in range(0, len(csv.columns), num_tasks):
        #no need to rename the first experiment
        if i == 0:
            continue
        
        for j in range(num_tasks):
            #Renaming the rest of the experiments
            csv.columns.values[i+j] = i+j
            csv_min.columns.values[i+j] = i+j
            csv_max.columns.values[i+j] = i+j
    return csv, csv_min, csv_max

def generate_avg(seeds, experiment, target_file, num_columns, num_tasks):
    
    NUM_ROWS = 16#24 #For only reading 20 rows of abr_sim
    eval_all_file = f'{experiment}_ALL'
    names_of_columns = []
    num_seeds = len(seeds)
    
    print(eval_all_file)
    for i in range(num_columns):
        names_of_columns.append(i)

    rewards = pd.read_csv(f'{eval_all_file}.csv', names=names_of_columns)
    
    if "ABR_SIM" in experiment:
        abr_sim_rows = []
        row_names = []
        curr_row_count = 0
        while True:
            if curr_row_count == 0:
                abr_sim_rows.append(curr_row_count*NUM_ROWS)
            else:
                abr_sim_rows.append(curr_row_count*NUM_ROWS-1)
            row_names.append(curr_row_count)
            curr_row_count += 1
            #We're done reading and  at the end of the file
            if(curr_row_count*NUM_ROWS == len(rewards)):
                break

        rewards = rewards.iloc[abr_sim_rows] #Read only every 24th row
        rewards.index = row_names #Renaming rows

    rewards_min = []
    rewards_max = []

    for i in range(num_tasks):
        #For generating the number of columns for number of seeds
        avglist = []
        curr_seed_count = 0
        while curr_seed_count < num_seeds:
            avglist.append(i+curr_seed_count*num_tasks)
            curr_seed_count += 1
        
        #Calculate the mean down the rows of the columns we want
        rewards[i] = rewards[avglist].mean(axis=1)
        #Calculate the minimum and maximum bound with the standard deviation
        rewards_min.append(rewards[i] - rewards[avglist].std(axis=1))
        rewards_max.append(rewards[i] + rewards[avglist].std(axis=1))

    rewards = rewards.iloc[:, 0:num_tasks]

    rewards_min = pd.DataFrame(rewards_min).transpose()
    rewards_max = pd.DataFrame(rewards_max).transpose()

    print(target_file)
    rewards.to_csv(target_file, index=False, header=None)

    return rewards, rewards_min, rewards_max


if __name__ == '__main__':
    seeds = [5,10,15]
    
    """To make it suppoValueError: Wrong number of items passed 3, placement implies 1rt more experiments/tasks/seeds, all you would need to do is 
    change the seeds, experiments, target_files, exp_tasks_names, and file_tasks_names list"""

    """make sure at index i, experiments[i] matches up with the target_files[i] and exp_task_names[i+j]
    where j is the index of the file_task_names list and it increments by 1 through the list until 
    len(file_task_names)"""

    """So for example, if experiments[0] is percentile aggregate for fihser, make sure target_files[0] is
    where you wawnt to save fisher experiments while exp_task_names[0+0] = fisher_task_1
    exp_task_names[0+1] = fisher_task_2, exp_task_names[0+2] = fisher_task_3"""

    experiments = [ "A2C_RL_SERVER_LAMDA2", "A2C_RL_SERVER_GRAD", "A2C_RL_SERVER_FEDADP", "A2C_RL_SERVER_FEDAVG",
    "A2C_RL_SERVER_MAS" ]
    target_files = ["AVG_OF_LAMDA2_EXPERIMENT.csv", "AVG_OF_GRAD_EXPERIMENT.csv", "AVG_OF_ADP_EXPERIMENT.csv", "AVG_OF_FEDAVG_EXPERIMENT.csv", 
    "AVG_OF_MAS_EXPERIMENT.csv"]
    exp_task_names = [ "Curriculum Critic Loss Grad + MAS lamda 2", "Curriculum Critic Loss Grad", "FedADP", "FedAvg", "MAS Lamda 2"]
    file_task_names = ["task 1", "task 2", "task 3", "task 4", "task 5", "task 6"]
    
    num_columns = len(file_task_names) * len(seeds)
   
    avg_rewards_of_experiments = []
    #The minimum and maximium bound of the experiments
    min_bound= []
    max_bound = []

    #loop through how many experiments we have
    for i in range(len(experiments)):
        rewards, rewards_min, rewards_max = generate_avg(seeds, experiments[i], target_files[i], num_columns, len(file_task_names))
        avg_rewards_of_experiments.append(rewards)
        min_bound.append(rewards_min)
        max_bound.append(rewards_max)
    
    plot_rewards, plot_min, plot_max = combine_n_dataframes(avg_rewards_of_experiments, min_bound, max_bound, len(file_task_names))
    plot(plot_rewards, plot_min, plot_max, file_task_names, exp_task_names, len(file_task_names), len(experiments))
