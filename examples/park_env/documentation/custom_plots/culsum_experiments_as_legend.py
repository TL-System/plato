from audioop import avg
from platform import python_branch
from shutil import which
from unittest import result
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot(clients, exp_name, culsum_plot=None, culsum_min=None, culsum_max=None):
    #We start plotting here
    TSBOARD_SMOOTHING_CULSUM = 0.6
    INTERVAL = 10.0 #interval for how much you want to space out the grpah
    sns.color_palette("flare", as_cmap=True)

    sns.set(font_scale=2.5)
    sns.set_style("whitegrid")

    """USE THIS IF NEEDED? Kinda skews numbers a little"""
    #culsum_plot = culsum_plot.ewm(alpha=(1- TSBOARD_SMOOTHING_CULSUM)).mean()
    culsum_min = culsum_min.ewm(alpha=(1- TSBOARD_SMOOTHING_CULSUM)).mean()
    culsum_max = culsum_max.ewm(alpha=(1- TSBOARD_SMOOTHING_CULSUM)).mean()
    

    #Change axes number X to however many you need
    #If only want task1, task2, culmunative, change num_sub_plot accordingly
    #By default it is 4 right now
    num_sub_plots = 3

    f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=num_sub_plots, figsize=(25,6))
    list_of_subplots = [ax1, ax2, ax3]

    marker_line = [(None, '-'), ('None', ':'), ('None', '-.'), ('^', '-'), ('o', '-'), ('8', ':'), ('^', ':'), ('D', ':'),  ('s', ':'), (None, ':'), ('^', '-')]
    culsum_order = 0
    which_plot = 0
    which_client = 0
    which_exp = 0
    
    for i in range(len(culsum_plot.columns)):
        #resets which client
        if which_exp == len(exp_name):
            which_exp = 0
       
        #new plot index
        if i % len(exp_name) == 0 and i != 0:
            which_plot += 1
            which_client += 1
            culsum_order = 0
    
        rewards = culsum_plot[i]
        rewards.index = [i+1 for i in range(len(rewards))]
        marker, line = marker_line[culsum_order%len(marker_line)]
        sns.lineplot(data=rewards, linewidth=4, legend=0, label=exp_name[which_exp], marker=marker, markersize=10, linestyle=line, ax=list_of_subplots[which_plot])
        #label=clients[which_client]
        list_of_subplots[which_plot].set_title(clients[which_client])
        which_exp += 1   

        list_of_subplots[which_plot].set_ylabel("Cumulative average reward")
        list_of_subplots[which_plot].set_ylim(-1000, 550) #Change this to wanted range

        list_of_subplots[which_plot].set_xlabel("Training round", fontstyle='normal')
        list_of_subplots[which_plot].xaxis.set_major_formatter(plticker.FormatStrFormatter('%d'))
        loc = plticker.MultipleLocator(base=INTERVAL) # this locator puts ticks at regular intervals of two
        list_of_subplots[which_plot].xaxis.set_major_locator(loc)

        #To maintain same plot
        culsum_order += 1

    handles, labels = f.axes[-1].get_legend_handles_labels()
    f.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.07, 0., 0.), borderaxespad=0.1, ncol=len(exp_name), fontsize=30)

    plt.tight_layout() 

    pdf_name = "OneLegendExperimentsWithClients"
    f.savefig(f'{pdf_name}.pdf', bbox_inches='tight')
    
def combine_n_dataframes(csv_list, csv_min_list, csv_max_list, num_culsum_columns):
    csv = pd.concat(csv_list, axis=1, join='inner')
    csv_min = pd.concat(csv_min_list, axis=1, join='inner')
    csv_max = pd.concat(csv_max_list, axis=1, join='inner')

    for i in range(num_culsum_columns):
        if i == 0:
            continue
        csv.columns.values[i] = i
        csv_min.columns.values[i] = i
        csv_max.columns.values[i] = i

    return csv, csv_min, csv_max
    
def generate_culsum_avg(seeds, experiment, target_file, num_columns, num_tasks):
    """Comments use 3 seeds as the example but extends up to n seeds"""
    eval_all_file = f'{experiment}_ALL'
    names_of_columns = []
    num_seeds = len(seeds)
    NUM_ROWS = 8#24

    for i in range(num_columns):
        names_of_columns.append(i)
    rewards = pd.read_csv(f'{eval_all_file}.csv', names=names_of_columns)
    print(eval_all_file)

    if "abr_sim" in experiment:
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

    for i in range(num_seeds):
        #Generating the sumlist
        sumlist = []
        
        curr_seed_count = 0
        while curr_seed_count < num_tasks:
            sumlist.append(i*num_tasks+curr_seed_count)
            curr_seed_count += 1
        #Calculate the sum down the rows of the columns we want
        rewards[i] = rewards[sumlist].sum(axis=1)

    #Only want the first 3 columns 
    rewards = rewards.iloc[:, 0:num_seeds]

    #Get the std dev of the 3 columns
    rewards_std_dev = rewards.std(axis=1)

    #Take the mean of the 3 columns along the rows
    rewards = rewards.mean(axis=1)

    rewards_min = rewards - rewards_std_dev
    rewards_max = rewards + rewards_std_dev

    rewards = pd.DataFrame(rewards)
    rewards_min = pd.DataFrame(rewards_min)
    rewards_max = pd.DataFrame(rewards_max)

    print(target_file)
    rewards.to_csv(f'{target_file}.csv', index=False, header=None)

    return rewards, rewards_min, rewards_max

if __name__ == '__main__':
    seeds = [5,10,15]
    NUM_TASKS = 6

    """Make sure that experiments at index i match up with target files at index i
    For example, percentile aggreagte fisher at index 0 for experiments should be the same for
    target_files at index 0 (both fisher). Make sure exp_names list matches too (at index 0 it is 
    also fisher)."""

    """To make it support more experiments, seeds, and tasks all you would need to do is change the
    number of tasks and the seeds, experiments, and target_files list"""

    experiments = ["6out6_fedavg", "6out6_fedadp", "6out6_critic_grad", "6out6_abr_sim",
     "12out12_fedavg", "12out12_fedadp", "18out18_fedadp", "12out12_abr_sim",
    "18out18_fedavg","18out18_fedadp","18out18_critic_grad", "18out18_abr_sim"]
    target_files = ["CULSUM_AVG_6out6_fedavg", "CULSUM_AVG_6out6_fedadp","CULSUM_AVG_6out6_critic_grad", "CULSUM_AVG_6out6_abr_sim",
    "CULSUM_AVG_12out12_fedavg", "CULSUM_AVG_12out12_fedadp", "CULSUM_AVG_12out12_critic_grad", "CULSUM_AVG_12out12_abr_sim",
    "CULSUM_AVG_18out18_fedavg", "CULSUM_AVG_18out18_fedadp", "CULSUM_AVG_18out18_critic_grad", "CULSUM_AVG_18out18_abr_sim"]
    exp_names = ["FedAvg", "FedADP", "Cascade", "RL-ABR"]

    #number of columns for the culmunating sum final csv file
    num_culsum_columns = len(experiments)

    #number of columns from the original csv file we are reading from
    num_original_columns = len(seeds) * NUM_TASKS

    avg_rewards_of_experiments = []
    #The minimum and maximium bound of the experiments
    min_bound= []
    max_bound = []

    clients = ["K = 6", "K = 12", "K = 18"]

    for i in  range(len(experiments)):
        #print(len(experiments), len(target_files))
        rewards, rewards_min, rewards_max = generate_culsum_avg(seeds, experiments[i], target_files[i], num_original_columns, NUM_TASKS)
        avg_rewards_of_experiments.append(rewards)
        min_bound.append(rewards_min)
        max_bound.append(rewards_max)

    #print(avg_rewards_of_experiments)
    plot_csv, plot_min, plot_max = combine_n_dataframes(avg_rewards_of_experiments, min_bound, max_bound, num_culsum_columns)
    print(plot_csv)
    plot(clients, exp_names, plot_csv, plot_min, plot_max)
