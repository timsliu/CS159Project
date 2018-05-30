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
# mass_run - used to invoke graphing and trial functions many times


# Revision History
# Tim Liu    05/29/18    planned file and wrote stub functions

import pickle
import os
#import matplotlib.pyplots as plt

# current directory to return to each time
HOME = os.getcwd()

# dictionary mapping each environment name to a directory    
env2dir = {'InvertedPendulum-v2': 'full_pend', 
           'HalfInvertedPendulum-v0': 'half_pend',
           'LongInvertedPendulum-v0': 'long_pend',
           'LongCartInvertedPendulum-v0': 'long_cart'}

# global list of multitasking techniques
tech_list = ["hard_param", "soft_param", "fine_tuning", \
                         "distillation", "baseline"]

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
        # path to the correct technique
        save_path = os.path.join(HOME, technique)
        # path to the correct environment within the technique
        save_path = os.path.join(save_path, env2dir[env])
        # switch to that directory
        os.chdir(save_path)
        
        # construct file name - technique, environment, and time in
        # month, day, hour, minute, second format - for running reward
        file_name = technique + '_' + env2dir[env] + '_' + \
                    time.strftime("%m_%d_%H_%M_%S") + '_run_reward.p'
        # save the list or running rewards
        pickle.dump(rr_records[envi], open( file_name, "wb" ) )
        
        # construct file name - technique, environment, and time in
        # month, day, hour, minute, second format - for length each episode
        file_name = technique + '_' + env2dir[env] + '_' + \
                    time.strftime("%m_%d_%H_%M_%S") + '_length.p' 
        # save the list of roll lengths
        pickle.dump(length_records[envi], open( file_name, "wb" ) )

    # return to home directory at end
    os.chdir(HOME)
    return

def graph_length(technique, env):
    '''graphs the length of the run versus the episode of all trials for
    a given technique and environment; also graphs the average'''
    
    #TODO
    
    return

def graph_reward(technique, env):
    '''graphs the length of the run versus the episode of all trials for
    a given technique and environment; also graphs the average'''
    
    #TODO
    
    return

def graph_gen(technique, env):
    '''helper function for graph_reward and graph_length'''
    #TODO
    
    return

def trial_summary(technique_env):
    '''for a given technique and environment outputs a text file
    with the average convergence time and standard deviation'''
    
    #TODO
    
    return

def mass_run():
    '''used to invoke graphing or summary functions multiple times'''
    
    return