import pandas as pd
import numpy as np

def user_input():
    '''
    This function creates all initial parameter for the training based on user inputs.
    '''
    
    episode_list = [eps for eps in range(100,501,100)]

    data = input("Specific or Random?: ").lower()
    
    ID = 497
    
    brain = input("Enter agent type (all, q or dqn): ").lower()
    
    model_based = False
    
    '''if brain == 'q':
        modeled = input("Model-based? Y or N: ").upper()
        if modeled == 'Y':
            model_based = True
        else:
            model_based = False
    
    if brain == 'dqn':
        model_based = False
    '''
    
    if data == 'actual':
        station_history = citi_data_processing(ID)
        print(station_history)
        
    else:
        station_history = None
    
    return episode_list, data, ID, brain, model_based, station_history
