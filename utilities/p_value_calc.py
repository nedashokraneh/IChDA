import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import data_utils

def p_value_calc(df):

    weights = np.log2(df['weight'])
    dist_params = st.exponweib.fit(weights)
    p_values = 1 - st.exponweib.cdf(weights, dist_params[0], dist_params[1], dist_params[2], dist_params[3])
    df['weibull p-value'] = p_values

def main():


    parser = argparse.ArgumentParser(description='calculate p-values corresponding to log2 O/E interaction counts')

    parser.add_argument('-i', '--input_hic', required = True,
                       help='Path of input hic file')
    parser.add_argument('-l', "--log_dir", required = True,
                       help='log path')
    parser.add_argument('-f', "--first_chr", required = True,
                       help='first chromosome')
    parser.add_argument('-s', "--second_chr", required = True,
                       help='second chromosome')
    parser.add_argument('-o', '--out_dir', required = True,
                       help='Path of directory to store outputs')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    outfile_name = 'chr{}_chr{}.txt'.format(args.first_chr,args.second_chr)
    outfile_path = os.path.join(args.out_dir, outfile_name)

    if os.path.exists(outfile_path):
        print('p-values calculated before for chromosomes {} and {}...'.format(args.first_chr,args.second_chr))
    else:
        print('calculating p-values for chromosomes {} and {}...'.format(args.first_chr,args.second_chr))
        hic_data = pd.read_csv(args.input_hic, sep = "\t", header = None)
        hic_data = hic_data.dropna()
        weights = np.log2(hic_data.iloc[:,2])
        dist_params = st.exponweib.fit(weights)
        p_values = 1 - st.exponweib.cdf(weights, dist_params[0], dist_params[1], dist_params[2], dist_params[3])
        hic_data['weibull p-value'] = p_values
        outfile_name = 'chr{}_chr{}.txt'.format(args.first_chr,args.second_chr)
        outfile_path = os.path.join(args.out_dir, outfile_name)
        hic_data.to_csv(outfile_path, sep = "\t", index = False, header = None)

        ### loging learned distribution parameters
        dist_params_file = open(os.path.join(args.log_dir,'dist_params.txt'), 'a')
        dist_params_file.write('chr{} and chr{} params:\n'.format(args.first_chr,args.second_chr))
        for i in range(4):
            dist_params_file.write('{}\t'.format(dist_params[i]))
        dist_params_file.write("\n")

        ### loging figure of fitted distribution compared to the real histogram
        dist_fig_file = os.path.join(args.log_dir,'chr{}_chr{}.jpeg'.format(args.first_chr,args.second_chr))
        fig, ax = plt.subplots(figsize=(4.5,3))
        ax = sns.distplot(weights,label='counts histogram')
        x_min, x_max = ax.get_xlim()
        xs = np.linspace(x_min, x_max, 200)
        ys = st.exponweib.pdf(xs, dist_params[0], dist_params[1], dist_params[2], dist_params[3])
        ax = sns.lineplot(x=xs, y=ys,label='fitted weibull')
        ax.set_title("chromosomes {} and {}".format(args.first_chr,args.second_chr))
        ax.set(xlabel = None, ylabel = None)
        fig.savefig(dist_fig_file,facecolor='white', transparent=False)




if __name__ == "__main__":
    main()
