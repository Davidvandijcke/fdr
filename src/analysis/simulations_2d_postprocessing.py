import pandas as pd
import os
from tabulate import tabulate

def moveUp(fn, times = 1):
    for _ in range(times):
        fn = os.path.dirname(fn)
    return fn

def format_value(value):
    if isinstance(value, int):
        return str(value)
    return "{:.4f}".format(value)


def create_latex_subtables(df, sigma):
    # Filter the dataframe by the sigma value
    df_sigma = df[df['sigma'] == sigma]

    # Get the unique alpha values, sorted
    alphas = sorted(df_sigma['alpha'].unique())

    # Initialize the final LaTeX table string
    latex_tables = "\\begin{subtable}{\\textwidth}\n\\centering\n"
    
    d = 0.25

    # Iterate over each alpha value
    for alpha in alphas:
        # Get the subset of the dataframe with the current alpha value
        df_alpha = df_sigma[df_sigma['alpha'] == alpha].sort_values('N')

        # Remove the 'alpha', 'S', 'lambda', 'nu', 'sigma', 'Y_jumpfrom', and 'Y_jumpto' columns
        df_alpha = df_alpha.drop(['S', 'lambda', 'nu', 'sigma', 'Y_jumpfrom', 'Y_jumpto'], axis=1)
        
        # reorder columns
        cols = ['N', 'alpha', 'Y_jumpsize', 'mse', 'mse_est', 'bias', 'jump_neg', 'jump_pos']
        df_alpha = df_alpha[cols]
        
        df_alpha.rename(columns={'mse' : 'MSE',
                                'jump_neg' : 'FNR',
                                'jump_pos' : 'FPR'}, 
                        inplace=True)
        
        df_alpha['N'] = df_alpha['N'].astype(int)
        # format value on df_alpha
        df_alpha = df_alpha.applymap(format_value)


        # Generate the LaTeX table
        latex_table = tabulate(df_alpha, tablefmt="latex_booktabs", headers="keys", showindex=False)

        # Get the lambda and nu values
        lambda_val = df_sigma.loc[df_sigma['alpha'] == alpha, 'lambda'].values[0]
        nu_val = df_sigma.loc[df_sigma['alpha'] == alpha, 'nu'].values[0]

        # Add the sub-table to the final LaTeX tables string
        nrow = df_alpha.shape[1]
        latex_tables += " d = %.2f \\\\ \n%s \n" % (d, latex_table)
        latex_tables += "\\\\ SURE: $\\lambda$ = %.4f, $\\nu$ = %.4f  \\\\ \n\n" % (lambda_val, nu_val)

        d += 0.25

    # replace the names that require maths (tabulate processes them as text)
    latex_tables = latex_tables.replace(f"alpha", f"$ \\alpha $")
    latex_tables = latex_tables.replace("Y\\_jumpsize", "$ \\hat{\\alpha} $")
    latex_tables = latex_tables.replace("mse\\_est", "MSE $\\tau_{\\mathrm{FD}}$")
    latex_tables = latex_tables.replace("bias", "Bias $\\tau_{\\mathrm{FD}}$")
        


    latex_tables += "\\end{subtable}"
    
    return latex_tables





if __name__ == "__main__":
    
    # paths
    dir = os.path.dirname(__file__)
    # get directory above
    main_dir = moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    
    # overleaf synced dropbox folder (change to your own overleaf path)
    tabs_dir = "/Users/davidvandijcke/Dropbox (University of Michigan)/Apps/Overleaf/rdd/tabs/"
    
    # pull data from S3

    # run the command string in cli
    fn = "2022-06-28"
    ffrom = f"'s3://ipsos-dvd/fdd/data/{fn}/'"
    fto = f"'/Users/davidvandijcke/Dropbox (University of Michigan)/rdd/data/out/simulations/{fn}/'"
    !aws s3 sync $ffrom $fto --profile ipsos

    # read all files in fto
    df = pd.concat([pd.read_csv(os.path.join(data_out, 'simulations', fn, file)) for file in os.listdir(os.path.join(data_out, 'simulations', fn)) if file.endswith(".csv")])
    #df = pd.read_csv("/Users/davidvandijcke/Downloads/simulations_2d_sigma_0.01_jsize_0.10661253981895451 (1).csv")

    # Group by 'alpha', 'N', and 'S' and calculate the mean 'Y_jumpsize'
    df['Y_jumpsize'] = df['Y_jumpsize'].abs()
    df['bias'] = df['Y_jumpsize'].abs() - df['alpha']
    df['mse_est'] = df['bias']**2 
    mean_jumpsize = df.groupby(['alpha', 'N', 'S', 's']).agg({'Y_jumpsize' : 'mean', 
                                                'mse' : 'mean', 
                                                'mse_est' : 'mean',
                                                'bias' : 'mean',
                                                'jump_neg' : 'mean',
                                                'jump_pos' : 'mean',
                                                'Y_jumpfrom' : 'mean',
                                                'Y_jumpto' : 'mean',
                                                'lambda' : 'mean', 
                                                'nu' : 'mean', 
                                                'sigma' : 'mean'}).reset_index()
    mean_jumpsize = mean_jumpsize.groupby(['alpha', 'N', 'S']).agg({'Y_jumpsize' : 'mean', 
                                                'mse' : 'mean', 
                                                'mse_est' : 'mean',
                                                'bias' : 'mean',
                                                'jump_neg' : 'mean',
                                                'jump_pos' : 'mean', 
                                                'Y_jumpfrom' : 'mean',
                                                'Y_jumpto' : 'mean',
                                                'lambda' : 'mean', 
                                                'nu' : 'mean', 
                                                'sigma' : 'mean'}).reset_index()


    # df = df[df['alpha'] > 0.107]
    # df = df[df['Y_jumpsize'].abs() > np.sqrt(df['nu'])]
    # df['Y_jumpsize'].abs().mean()

    # Generate the LaTeX tables for each sigma value
    latex_table_sigma_0_01 = create_latex_subtables(mean_jumpsize, 0.01)
    latex_table_sigma_0_05 = create_latex_subtables(mean_jumpsize, 0.05)

    # Write the tables to separate .tex files
    with open(os.path.join(tabs_dir, 'table_sigma_0_01.tex'), 'w') as f:
        f.write(latex_table_sigma_0_01)

    with open(os.path.join(tabs_dir, 'table_sigma_0_05.tex'), 'w') as f:
        f.write(latex_table_sigma_0_05)
