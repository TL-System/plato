from audioop import avg
from re import S, sub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def plot(avg_plot, avg_min, avg_max, file_task_names, exp_task_name, num_tasks, num_experiments, culsum_plot=None, culsum_min=None, culsum_max=None):
    #We start plotting here
    TSBOARD_SMOOTHING_CULSUM = 0.0
    TSBOARD_SMOOTHING_AVG = 0.4
    INTERVAL = 5.0 #interval for how much you want to space out the grpah
    
    sns.color_palette("flare", as_cmap=True)

    sns.set(font_scale=3)
    sns.set_style("whitegrid")
   
    culsum_exists = 0
    if culsum_plot is not None:
        culsum_plot = culsum_plot.ewm(alpha=(1- TSBOARD_SMOOTHING_CULSUM)).mean()
        culsum_min = culsum_min.ewm(alpha=(1- TSBOARD_SMOOTHING_CULSUM)).mean()
        culsum_max = culsum_max.ewm(alpha=(1- TSBOARD_SMOOTHING_CULSUM)).mean()
        culsum_exists = 1

    #Change axes number X to however many you need
    #If only want task1, task2, culmunative, change num_sub_plot accordingly
    #By default it is 4 right now
    num_sub_plots = (len(avg_plot.columns)//num_experiments) + culsum_exists
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols= num_sub_plots, sharex=True, figsize=(25,6))
    list_of_subplots = [ax1, ax2, ax3, ax4]
    

    if(len(list_of_subplots) != num_sub_plots):
        print("More axes than number of subplots")
        return
    
    avg_plot = avg_plot.ewm(alpha=(1- TSBOARD_SMOOTHING_AVG)).mean()
    avg_min = avg_min.ewm(alpha=(1- TSBOARD_SMOOTHING_AVG)).mean()
    avg_max = avg_max.ewm(alpha=(1- TSBOARD_SMOOTHING_AVG)).mean()

    print(avg_plot)

    subplot_idx = 0 #Which subplot to use
    
    marker_line = [(None, '-'), ('s', '-'), ('^', '-'), ('D', '-'), ('o', '-'), ('8', ':'), ('^', ':'), ('D', ':'),  ('s', ':'), (None, ':'), ('^', '-')]
    order = 0
    culsum_order = order
    
    if culsum_plot is not None:
        #For culSum
        for i in range(len(culsum_plot.columns)):
            #get csv index
            rewards = culsum_plot[i]
            rewards.index = [i+1 for i in range(len(rewards))]
            marker, line = marker_line[culsum_order%len(marker_line)]
            sns.lineplot(data=rewards, linewidth=2.5, marker=marker, markersize=10,  linestyle=line, ax=list_of_subplots[subplot_idx])
            #ax1.fill_between(rewards.index, csv_std_max[i].values, csv_std_min[i].values, alpha=0.5)
            list_of_subplots[subplot_idx].set_ylabel("Cumulative average reward")
            list_of_subplots[subplot_idx].set_ylim(-500,1000)

            list_of_subplots[subplot_idx].xaxis.set_major_formatter(plticker.FormatStrFormatter('%d'))
            loc = plticker.MultipleLocator(base=INTERVAL) # this locator puts ticks at regular intervals of two
            list_of_subplots[subplot_idx].xaxis.set_major_locator(loc)
            
            #To maintain same plot
            if culsum_order == 0:
                culsum_order += 1
            else:
                culsum_order += num_tasks
            
          
    ylims = [(0, 1000), (100, 500), (0, 300), (-100, 300)]
    #For avg reward
    for i in range(num_sub_plots-culsum_exists):
        subplot_idx += 1
        exp_task_index = 0

        rewards = avg_plot[i]
        rewards.index = [i+1 for i in range(len(rewards))]
        marker, line = marker_line[order%len(marker_line)]
        ymin, ymax = ylims[subplot_idx]
        list_of_subplots[subplot_idx].set_ylim(ymin, ymax)
        sns.lineplot(data=rewards, linewidth=2.5, legend=0, label=exp_task_name[exp_task_index], marker=marker, markersize=10,  linestyle=line, ax=list_of_subplots[subplot_idx])
        
        
        #Axes for the next nth algorithm, if more than 2 algorithms, length csv.columns increases
        for j in range(i+num_tasks, len(avg_plot.columns), num_tasks):
            rewards = avg_plot[j]
            rewards.index = [i+1 for i in range(len(rewards))]
            exp_task_index += 1
            marker, line = marker_line[(order+j-i-num_tasks+1)%len(marker_line)]
            sns.lineplot(data=rewards, linewidth=2.5, legend=0, label=exp_task_name[exp_task_index], marker=marker, markersize=10,  linestyle=line, ax=list_of_subplots[subplot_idx])
            #ax1.fill_between(rewards.index, csv_std_max[j].values, csv_std_min[j].values, alpha=0.5)

        list_of_subplots[subplot_idx].set_ylabel(f'Average reward for {file_task_names[i]}')
        list_of_subplots[subplot_idx].tick_params(axis='y')
        
        plt.xlim(1, len(avg_plot))
        ymin, ymax = ylims[subplot_idx]
        list_of_subplots[subplot_idx].set_ylim(ymin, ymax)

        list_of_subplots[subplot_idx].xaxis.set_major_formatter(plticker.FormatStrFormatter('%d'))
        loc = plticker.MultipleLocator(base=INTERVAL) # this locator puts ticks at regular intervals of two
        list_of_subplots[subplot_idx].xaxis.set_major_locator(loc)
       
    #Xaxis label
    plt.figtext(0.478,-0.05, "Training round")
    
    #Create legend of figure
    handles, labels = f.axes[-1].get_legend_handles_labels()
    f.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.2, 0., 0.), borderaxespad=0.1, ncol=num_columns, fontsize=30)
      
    #Automatic spacing between subplots
    plt.tight_layout()

    pdf_name = "OneLegendExp"
    f.savefig(f'{pdf_name}.pdf', bbox_inches='tight')

def combine_avg_n_dataframes(csv_list, csv_min_list, csv_max_list, num_tasks):
    
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

def combine_n_culsum_dataframes(csv_list, csv_min_list, csv_max_list, num_culsum_columns):
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

def generate_avg(seeds, experiment, target_file, num_columns, num_tasks):
    
    eval_all_file = f'{experiment}_ALL'
    names_of_columns = []
    num_seeds = len(seeds)
    
    print(eval_all_file)
    for i in range(num_columns):
        names_of_columns.append(i)

    rewards = pd.read_csv(f'{eval_all_file}.csv', names=names_of_columns)

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

def generate_culsum_avg(seeds, experiment, target_file, num_columns, num_tasks):
    """Comments use 3 seeds as the example but extends up to n seeds"""
    eval_all_file = f'{experiment}_ALL'
    names_of_columns = []
    num_seeds = len(seeds)

    for i in range(num_columns):
        names_of_columns.append(i)
    
    rewards = pd.read_csv(f'{eval_all_file}.csv', names=names_of_columns)
    print(eval_all_file)

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
    NUM_TASKS = 3
    """To make it suppoValueError: Wrong number of items passed 3, placement implies 1rt more experiments/tasks/seeds, all you would need to do is 
    change the seeds, experiments, target_files, exp_tasks_names, and file_tasks_names list"""

    """make sure at index i, experiments[i] matches up with the target_files[i] and exp_task_names[i+j]
    where j is the index of the file_task_names list and it increments by 1 through the list until 
    len(file_task_names)"""

    """So for example, if experiments[0] is percentile aggregate for fihser, make sure target_files[0] is
    where you wawnt to save fisher experiments while exp_task_names[0+0] = fisher_task_1
    exp_task_names[0+1] = fisher_task_2, exp_task_names[0+2] = fisher_task_3"""

    experiments = ["A2C_RL_SERVER_FED_AVG",
        "A2C_RL_SERVER_PRESET_AGGREGATE",  "A2C_RL_SERVER_PERCENTILE_CRITIC", "A2C_RL_SERVER_PERCENTILE_CRITIC_LOSS", "A2C_RL_SERVER_PERCENTILE_CRITIC_GRAD"] # "A2C_RL_SERVER_PERCENTILE_ACTOR", "A2C_RL_SERVER_PERCENTILE_ACTOR_LOSS", "A2C_RL_SERVER_PERCENTILE_ACTOR_GRAD",
    target_files = [ "AVG_OF_FEDAVG_EXPERIMENTS.csv",
    "AVG_OF_PRESET_EXPERIMENTS.csv", "AVG_OF_CRITIC_EXPERIMENTS.csv", "AVG_OF_CRITIC_LOSS_EXPERIMENTS.csv", "AVG_OF_CRITIC_GRAD_EXPERIMENTS.csv"] # "AVG_OF_ACTOR_EXPERIMENTS.csv", "AVG_OF_ACTOR_LOSS_EXPERIMENTS.csv", "AVG_OF_ACTOR_GRAD_EXPERIMENTS.csv",
   # exp_task_names = ["fisher_task_1", "fisher_task_2", "fisher_task_3", "fedavg_task_1", "fedavg_task_2", "fedavg_task_3"]
    exp_task_names = ["FedAvg", "Preset curriculum", "Curriculum: critic fisher",  "Curriculum: critic loss", "Curriculum: critic grad"] # "Curriculum actor fisher", "Curriculum actor loss", "Curriculum actor grad"
    file_task_names = ["task 1", "task 2", "task 3"]

    target_culsum_files = ["FEDAVG_CULSUM_AVG", "PRESET_CULSUM_AVG", "FISHER_CULSUM_AVG_CRITIC", "FISHER_CULSUM_AVG_CRITIC_LOSS", "FISHER_CULSUM_AVG_CRITIC_GRAD"] #"FISHER_CULSUM_AVG_ACTOR","FISHER_CULSUM_AVG_ACTOR_LOSS", "FISHER_CULSUM_AVG_ACTOR_GRAD"
    
    
    num_columns = len(file_task_names) * len(seeds)

    num_culsum_cols = len(experiments)

    avg_rewards_of_experiments = []
    #The minimum and maximium bound of the experiments
    min_bound= []
    max_bound = []

    culsum_avg_rewards_of_experiments = []
    #The minimum and maximium bound of the experiments
    culsum_min_bound= []
    culsum_max_bound = []

    #loop through how many experiments we have
    for i in range(len(experiments)):
        rewards, rewards_min, rewards_max = generate_avg(seeds, experiments[i], target_files[i], num_columns, len(file_task_names))
        avg_rewards_of_experiments.append(rewards)
        min_bound.append(rewards_min)
        max_bound.append(rewards_max)
        rewards, rewards_min, rewards_max = generate_culsum_avg(seeds, experiments[i], target_culsum_files[i], num_columns, NUM_TASKS)
        culsum_avg_rewards_of_experiments.append(rewards)
        culsum_min_bound.append(rewards_min)
        culsum_max_bound.append(rewards_max)
 
    plot_avg_rewards, plot_avg_min, plot_avg_max = combine_avg_n_dataframes(avg_rewards_of_experiments, min_bound, max_bound, len(file_task_names))
    plot_culsum_rewards, plot_culsum_min, plot_culsum_max = combine_n_culsum_dataframes(culsum_avg_rewards_of_experiments, culsum_min_bound, culsum_max_bound, num_culsum_cols)
    
  #  print(plot_culsum_rewards)

    #plot_avg(plot_avg_rewards, plot_avg_min, plot_avg_max, file_task_names, exp_task_names, len(file_task_names), len(experiments))
   # save_file = "blah"
    plot(plot_avg_rewards, plot_avg_min, plot_avg_max, file_task_names, exp_task_names, len(file_task_names), len(experiments),
    plot_culsum_rewards, plot_culsum_min, plot_culsum_max)
    #test2 = plot_culsum(plot_culsum_rewards, plot_culsum_min, plot_culsum_max, exp_task_names, save_file)
    #sns.lineplot(ax=test)
   # sns.lineplot(ax=test2)
