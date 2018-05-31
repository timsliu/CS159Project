# visualization functions
#
# This function contains functions for recording, storing, and graphing
# run information.
#
# Table of contents
# init_list - initializes master list for holding run information
# pickle_list - stores the master list as a numpy array in binary (.p) files
#               using the pickle library
# graph_length - graphs the run length versus episode of all recorded runs
# graph_reward - graphs the running reward versus episode recorded runs
# graph_gen - helper function for graph_length and graph_reward
# trial_summary - prints summary data (convergence time and standard deviation)
#                 of the trials by looking at the length of arrays
# mean_list - helper function for calculating average of many trials
# mass_run - used to invoke graphing and trial functions many times


# Revision History
# Tim Liu    05/29/18    planned file and wrote stub functions
# Tim Liu    05/30/18    wrote init_list, update_records, and pickle_list
# Tim Liu    05/30/18    wrote plotting functions
# Tim Liu    05/30/18    updated so all plots go to a single folder
# Tim Liu    05/30/18    fixed bug that miscalculated run lengths


import pickle
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# current directory to return to each time
HOME = os.getcwd()

# dictionary mapping each environment name to a directory    
env2dir = {'InvertedPendulum-v2': 'full_pend', 
           'HalfInvertedPendulum-v0': 'half_pend',
           'LongInvertedPendulum-v0': 'long_pend'}

# Update this with other environments!!

# global list of environment directories (not environment full names)
# used for error checking
env_list = [env2dir[key] for key in env2dir]

# global list of multitasking techniques - used for error checking
tech_list = ["hard_param", "soft_param", "fine_tuning", \
                         "distillation", "baseline"]

# global reward threshold
REWARD_THRESHOLD = 30


def init_list(env_names):
    '''initializes the global length_records and rr_records lists to
       be of the appropriate dimensions. The code in each technique must
       set the global length_records and rr_records list to the returned
       master_blank list.
       
       arguments: env_names - list with names of the environments
       return:    master_blank - blank list that will be filled with
                                 runtime records'''
    
    master_blank = [[] for x in range(len(env_names))]
    
    
    return master_blank

def update_records(roll_length, run_reward, length_records, rr_records):
    '''updates the master list of length records and running rewards
    to reflect a new episode.
    arguments: roll_length - array of the length of how long each environment
                              lasted in the most recent episode
               run_reward - array of the running rewards for each env
                                 in the most recent episode
                length_records - global list of lengths
                rr_records - global list of running rewards
    return: updated rr_records and length_records'''
    # iterate through the environments
    for i in range(len(roll_length)):
        # add most recent roll length for ith environment
        length_records[i].append(roll_length[i])
        # add most recent running reward for ith environment
        rr_records[i].append(run_reward[i])
    
    return length_records, rr_records

def update_fine(roll_length, run_reward, length_records, rr_records, env):
    '''custom function for updating the fine_tuning records - necessary
       because fine_tuning does not simultaneously train multiple
       environments
       arguments: env - index of the environment being trained'''
    
    # only update the currently training environment
    length_records[env].append(roll_length)
    # add most recent running reward for ith environment
    rr_records[env].append(run_reward)
    
    return length_records, rr_records
    

def pickle_list(technique, env_names, length_records, rr_records):
    '''stores the trial records as binary files using the pickle library. The
    function navigates to the correct directory (technique and environment)
    and then saves the length_records and rr_records lists.
    
    arguments: technique - multitasking technique used
               env_names - list of names of the environments
               length_records - 2D arry with each sub array holding the 
                                lengths of the runs for each environment
               rr_records - running reward records; 2D array with each
                            sub-array representing the running rewards for
                            each environment
    return: none
    '''
    # check the technique belongs in one of the technique directories
    if technique not in tech_list:
        print("Learning technique not recognized- exiting without save...")
        
    # iterate saving each environment
    for (envi, env) in enumerate(env_names):
        #path to the folder of trial records
        save_path = os.path.join(HOME, 'trial_records')
        # path to the correct technique
        save_path = os.path.join(save_path, technique)
        # path to the correct environment within the technique
        save_path = os.path.join(save_path, env2dir[env])
        # switch to that directory
        os.chdir(save_path)
        
        # construct file name - technique, environment, and time in
        # month, day, hour, minute, second format - for running reward
        file_name = technique + '_' + env2dir[env] + '_' + \
                    time.strftime("%m_%d_%H_%M_%S") + '_run_reward.p'
        # save the list of running rewards
        f = open(file_name, "wb" )
        pickle.dump(rr_records[envi], f)
        f.close()
        print(file_name, " successfully saved.")
        
        
        # construct file name - technique, environment, and time in
        # month, day, hour, minute, second format - for length each episode
        file_name = technique + '_' + env2dir[env] + '_' + \
                    time.strftime("%m_%d_%H_%M_%S") + '_length.p' 
        # save the list of roll lengths
        f = open(file_name, "wb" )
        pickle.dump(length_records[envi], f)
        f.close()
        print(file_name, " successfully saved.")

    # return to home directory at end
    os.chdir(HOME)
    return

def read_pickle(technique, env, filename):
    '''function for reading a binary pickle file which holds a list of
    run information
    arguments: technique (string) - multitask learning technique
               env (string) - environment string (see env2dir)
               filename - full name of the file to open'''
    # path to the folder of trial records
    search_path = os.path.join(HOME, 'trial_records')
    # path to the correct technique
    search_path = os.path.join(search_path, technique)
    # path to the correct environment within the technique
    search_path = os.path.join(search_path, env)
    # switch to directory the file is in
    os.chdir(search_path)
    
    f = open(filename, "rb")
    data_list = pickle.load(f)
    if "length" in filename:
        data_type = "   length: "
    if "_run_reward" in filename:
        data_type = "   running reward: "
        
    for i in range(len(data_list)):
        print("Episode ", i, data_type, data_list[i])
    
    # return to home directory
    os.chdir(HOME)
        
    return
    
def graph_length(technique, env):
    '''graphs the length of the run versus the episode of all trials for
    a given technique and environment; also graphs the average'''
    
    if technique not in tech_list:
        print("technique not found! - returning without plotting")
        print("List of techniques: ", tech_list)
        return
    if env not in env_list:
        print("environment not found! - returning without plotting")
        print("List of envs", env_list)
        return
    
    graph_gen(technique, env, "length")
    
    return

def graph_reward(technique, env):
    '''graphs the length of the run versus the episode of all trials for
    a given technique and environment; also graphs the average'''
       
    if technique not in tech_list:
        print("technique not found! - returning without plotting")
        print("List of techniques: ", tech_list)
        return
    if env not in env_list:
        print("environment not found! - returning without plotting")
        print("List of envs", env_list)
        return
    
    graph_gen(technique, env, "run_reward")
    
    return

def graph_gen(technique, env, data_type):
    '''helper function for graph_reward and graph_length. General plotter
    The function first looks for all data files with the correct technique,
    environment, and data_type (running reward or length). The function 
    then combines the data from each individual file into a single array
    and plots each individual trial. The function calls compute_mean_list
    to compute the average line, which is also plotted.'''

    # get single list with all trial data for technique, environment, and
    # data_type triplet
    data_list = get_combined_list(technique, env, data_type)
    print("%d data files found" %len(data_list))
    print("Computing mean of trials...")
    # list of the mean data
    mean_list = compute_mean_list(data_list)
        
    # change directory to where all the plots belong    
    plot_path = os.path.join(HOME, 'trial_records')
    os.chdir(os.path.join(plot_path, 'all_plots'))
    
    for trial in data_list:
        plt.plot(trial, linewidth = 1, color = 'lightskyblue')
        
    title_string = technique + "_" + env + "_" + data_type + '.png'
    # plot the average line
    plt.plot(mean_list, linewidth = 2, color = 'blue')
    # title and label the plot
    plt.title(title_string)
    plt.xlabel("Episodes")
    plt.ylabel(data_type)
    plt.grid(True)
    
    # save the figure
    plt.savefig(title_string)
    print("New file %s saved to all_plots" %title_string)
    plt.close()
        
    os.chdir(HOME)
    
    return

def trial_summary():
    '''generates summary data for all technique-environment pairs'''
    # switch to directory to dump all the data
    os.chdir(os.path.join(os.path.join(HOME, 'trial_records'), 'summary'))
    print("Overwriting previous summary file...")
    summary = open("summary.txt", 'w')
    # number of combinations for which summary data was generated
    num_sum = 0
    
    # go through all techniques and environment pairs
    for tech in tech_list:
        for env in env_list:
            # get all data combined
            roll_lengths = get_combined_list(tech, env, 'length')
            running_rewards = get_combined_list(tech, env, 'run_reward')
            
            if roll_lengths == []:
                # this technique-environment combo has no data! Move on!
                continue

            num_sum += 1  # increment number of pairings for which we have data
            # get average run data
            mean_lengths = compute_mean_list(roll_lengths)
            mean_rewards = compute_mean_list(running_rewards)
            
            # go back to the correct directory
            os.chdir(os.path.join(os.path.join(HOME, 'trial_records'),\
                                  'summary'))
            
                        
            # save the mean length         
            file_name = tech + '_' + env + '_mean_length.p'
            f = open(file_name, "wb" )
            pickle.dump(mean_lengths, f)
            f.close()    
            
            # save the mean running reward
            file_name = tech + '_' + env + '_mean_running_reward.p'
            f = open(file_name, "wb" )
            pickle.dump(mean_rewards, f)
            f.close() 
            
            # list of convergence times for each trial
            conv_times = get_convergence_times(running_rewards)
            out_string = "Technique:   %s   Environment:   %s \n\
            Average Convergence Time:            %.1f\n\
            Convergence Time Standard Deviation: %.1f\n\
            Number of trials:                    %d\n\n"\
                          %(tech, env, np.mean(conv_times), \
                            np.std(conv_times), len(conv_times))
            
            summary.write(out_string)
    summary.close()
    print("Summaries for %d technique environment combos generated" %num_sum)
    return

def get_convergence_times(running_rewards):
    '''returns a list of when the trials surpassed the reward threshold
    arguments: running_rewards - list of the running rewards of different
                                 trials
    return: convergence_times - list of episodes to converge'''
    
    # list of number of episodes for each trial to converge
    convergence_times = []
    
    # iterater through each trial
    for trial in running_rewards:
        i = 0 # number of passed episodes
        while trial[i] < REWARD_THRESHOLD:
            i += 1
        convergence_times.append(i)
      
    return convergence_times

def compute_mean_list(data_list):
    '''helper function takes an array of subarrays holding roll length
    or running reward and averages the lists together; the average is
    taken across all lists that have not yet terminated. The average
    data list has as many elements as the mean data_list so that the 
    end of the mean_list is not the average of a small number of trials
    arguments: data_list - list of all data
    return value: average_list - 1D list with the average at each episode'''
    
    if len(data_list) == 0:
        return []
    
    # list of how long each element is
    length_list = [len(x) for x in data_list]
    # empty list that will hold the averages
    average_list = [0 for x in range(int(np.mean(length_list)))]
    
    # iterate through average list
    for i in range(len(average_list)):
        trial_sum = 0    # sum for calculating average
        num_trials = 0   # number of trials with at least length i
        
        # iterate through all of the trials
        for t in range(len(data_list)):
            # check that trial hasn't ended
            if length_list[t] > i:
                # trial hasn't ended - it should contribute to the average
                trial_sum += data_list[t][i]
                num_trials += 1
        average_list[i] = trial_sum/num_trials
        
    return average_list

def get_combined_list(technique, env, data_type):
    '''helper function that puts the data from multiple trials
    together into a single list
    arguments: technique - multitask learning technique
               env - training environment
               data_type - (string) "length" or "run_reward"
    return value: data_list - single list with each element a list
                              representing one trial'''    
    # path to the folder of trial records
    search_path = os.path.join(HOME, 'trial_records')
    # path to the correct technique
    search_path = os.path.join(search_path, technique)
    # path to the correct environment within the technique
    search_path = os.path.join(search_path, env)
    # switch to directory the file is in
    os.chdir(search_path)
    
    # generate list of files with the data we're interested in
    file_list = [x for x in os.listdir() if data_type in x]
    print(file_list)
    
    # list of all the data
    data_list = []
    # iterate through the trials in the directory
    for trial_file in file_list:
        f = open(trial_file, "rb")
        # add trial data list to the main data list
        data_list.append(pickle.load(f))
        f.close()
        
    os.chdir(HOME)
        
    return data_list

def graph_compares(techs_evs, data_type, name = "comparison.png"):
    '''graphs the mean lines for multiple technique-environment pairs
    on a single graph
    arguments: tech_evs - list of tuples with each tuple a technique/
                          environment pair
               data_type - string either "length" or "run_reward"
               name - name of saved file'''
    
    # TODO
    
    return  

def mass_run():
    '''used to invoke graphing or summary functions multiple times'''
    
    return
