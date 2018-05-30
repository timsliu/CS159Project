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
import matplotlib.pyplots as plt

# current directory to return to each time
HOME = os.getcwd()



def init_list(env_names):
    '''initializes the global length_records and rr_records lists to
       be of the appropriate dimensions. The code in each technique must
       set the global length_records and rr_records list to the returned
       master_blank list.
       
       arguments: env_names - list with names of the environments
       return:    master_blank - blank list that will be filled with
                                 runtime records'''
    
    master_blank = []
    
    #TODO
    
    return master_blank

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
    
    #TODO
    
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