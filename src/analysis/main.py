import pandas as pd 
import os
import numpy as np


if __name__ == "__main__":

    dir = os.path.dirname(__file__)

    # get directory above
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(dir)))
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')    

        

    # read all files in simulations/2022-06-14 as pandas dataframe
    df = pd.concat([pd.read_csv(os.path.join(data_out, 'simulations', '2022-06-14', file)) for file in os.listdir(os.path.join(data_out, 'simulations', '2022-06-14'))])

    # Group by 'alpha', 'N', and 'S' and calculate the mean 'Y_jumpsize'
    df['Y_jumpsize'] = df['Y_jumpsize'].abs()
    mean_jumpsize = df.groupby(['alpha', 'N', 'S', 's']).agg({'Y_jumpsize' : 'mean', 
                                                'mse' : 'mean', 
                                                'jump_neg' : 'mean',
                                                'jump_pos' : 'mean',
                                                'Y_jumpfrom' : 'mean',
                                                'Y_jumpto' : 'mean',
                                                'lambda' : 'mean', 
                                                'nu' : 'mean'}).reset_index()
    mean_jumpsize = mean_jumpsize.groupby(['alpha', 'N', 'S']).agg({'Y_jumpsize' : 'mean', 
                                                'mse' : 'mean', 
                                                'jump_neg' : 'mean',
                                                'jump_pos' : 'mean', 
                                                'Y_jumpfrom' : 'mean',
                                                'Y_jumpto' : 'mean',
                                                'lambda' : 'mean', 
                                                'nu' : 'mean'}).reset_index()

    # Create a new column 'N_S' that combines 'N' and 'S' as a tuple
    #?mean_jumpsize['N_S'] = mean_jumpsize.apply(lambda row: f"{row['N']}_{row['S']}", axis=1)

    # Create the pivot table with 'alpha' as rows and 'N_S' as columns
    pivot_table = mean_jumpsize.pivot_table(index='alpha', columns='N', values='Y_jumpsize')

    # Optional: sort the index and columns if needed
    pivot_table = pivot_table.sort_index(axis=0).sort_index(axis=1)

    # Display the pivot table
    print(pivot_table)