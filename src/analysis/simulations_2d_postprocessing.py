from FDD import FDD
from FDD.SURE import SURE
import numpy as np
import pandas as pd
import torch 
from matplotlib import pyplot as plt
import ray
import os

def moveUp(fn, times = 1):
    for _ in range(times):
        fn = os.path.dirname(fn)
    return fn


if __name__ == "__main__":
    
    # paths
    dir = os.path.dirname(__file__)
    # get directory above
    main_dir = moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')   
    
    # pull data from S3
    "aws s3 sync s3://ipsos-dvd/fdd/data/2022-06-14/ /Users/davidvandijcke/Dropbox (University of Michigan)/rdd/data/out/simulations --profile ipsos"
    
    # run the command string in cli
    fn = "2022-06-14"
    ffrom = f"'s3://ipsos-dvd/fdd/data/{fn}/'"
    fto = f"'/Users/davidvandijcke/Dropbox (University of Michigan)/rdd/data/out/simulations/{fn}/'"
    !aws s3 sync $ffrom $fto --profile ipsos

    # read all files in fto
    df = pd.concat([pd.read_csv(os.path.join(data_out, 'simulations', fn, file)) for file in os.listdir(os.path.join(data_out, 'simulations', fn)) if file.endswith(".csv")])

    # Group by 'alpha', 'N', and 'S' and calculate the mean 'Y_jumpsize'
    df['Y_jumpsize'] = df['Y_jumpsize'].abs()
    mean_jumpsize = df.groupby(['alpha', 'N', 'S', 's']).agg({'Y_jumpsize' : 'mean', 
                                                'mse' : 'mean', 
                                                'jump_neg' : 'mean',
                                                'jump_pos' : 'mean',
                                                'Y_jumpfrom' : 'mean',
                                                'Y_jumpto' : 'mean',
                                                'lambda' : 'mean', 
                                                'nu' : 'mean', 
                                                'sigma' : 'mean'}).reset_index()
    mean_jumpsize = mean_jumpsize.groupby(['alpha', 'N', 'S']).agg({'Y_jumpsize' : 'mean', 
                                                'mse' : 'mean', 
                                                'jump_neg' : 'mean',
                                                'jump_pos' : 'mean', 
                                                'Y_jumpfrom' : 'mean',
                                                'Y_jumpto' : 'mean',
                                                'lambda' : 'mean', 
                                                'nu' : 'mean', 
                                                'sigma' : 'mean'}).reset_index()

    # Create a new column 'N_S' that combines 'N' and 'S' as a tuple
    #?mean_jumpsize['N_S'] = mean_jumpsize.apply(lambda row: f"{row['N']}_{row['S']}", axis=1)

    # Create the pivot table with 'alpha' as rows and 'N_S' as columns
    pivot_table = mean_jumpsize.pivot_table(index='alpha', columns='N', values='Y_jumpsize')

    # Optional: sort the index and columns if needed
    pivot_table = pivot_table.sort_index(axis=0).sort_index(axis=1)

    # Display the pivot table
    print(pivot_table)
