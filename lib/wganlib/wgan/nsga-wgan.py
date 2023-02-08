import sys
from absl import flags, app
from absl.flags import FLAGS
import pandas as pd
import time
from wgan.gan_models import *
from wgan.utils import *
import logging
from tqdm import tqdm
import warnings
from torchinfo import summary
import wandb
import os
from os.path import exists
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
import random
from torch.multiprocessing import Pool
from joblib import Parallel, delayed


os.chdir('../../..')

flags.DEFINE_boolean('perc',
                     False,
                     'set if perceptual loss will be applied.')

flags.DEFINE_boolean('p2p',
                     False,
                     'set if pixel-wise loss will be applied between fakes and ideals.')

flags.DEFINE_string('data_path',
                    f"{os.getcwd()}/scripts/data/",
                    'Set the dataset with real labels.')

flags.DEFINE_integer('critic_iter',
                     10,
                     'How many times Critic (Discriminator) will be trained more times than G')

flags.DEFINE_integer('max_gen',
                     5,
                     'Max number of Generations for NSGA-II to run.')

flags.DEFINE_integer('type',
                     2,
                     'Define the type of the Discriminator (Critic). Either WGAN-Clipping=1 or WGAN-GP=2')


########################
# Main Code Starts Here!
########################

def main(*args):
    FLAGS = flags.FLAGS
    warnings.filterwarnings('ignore')
    t_m = 12
    data_path = FLAGS.data_path
    file_save_path = './data/'
    # Uncomment 1 line for: Inline
    dataset = f"{data_path}multiclass_dataset_8181_254.csv"

    # Uncomment 1 line for: E-test
    #dataset = f"{data_path}E-test_modified_wrt_limits.csv"

    data = pd.read_csv(dataset)
    data_inline = data.iloc[:, 3:-2]

    pop_size = 45  # this is the population of each generation and not the size of individual genes
    max_gen = FLAGS.max_gen  # stop after 200 generations
    max_limit = 20  # this is the stopping criterion, if the top 10 does not change for 20 generations, then we stop.
    hps =   { # WGAN's parameters: TODO: design a better architecture for Generator and Discriminator (i.e. Critic)
            "seed": 50, "nchan": FLAGS.nchan, "osize": None, "ndf": FLAGS.ndf, "n_epochs": FLAGS.n_epochs,
            "batch_size": FLAGS.batch_size, "lr": 0.001, "weight_decay": .14, "beta1": .5, "workers": 8, "process_index": None,
            "critic_iter": FLAGS.critic_iter, "type": FLAGS.type, 'max_gen': FLAGS.max_gen, 'load_model': FLAGS.load_model,
            }
    hps = AttributeDict(hps)

    # Initialization
    # min_x=0
    # max_x=249
    master_list = data_inline.columns.tolist()
    for col_no_dev in master_list:
        if data_inline[col_no_dev].std() < 0.000001:
            master_list.remove(col_no_dev)
    #print(len(master_list))
    random.shuffle(master_list, myfunction)
    t_m = 12
    # d = 1
    # nu_val = 0.00195
    os.makedirs(file_save_path, exist_ok=True)  # This will create a folder if it is already not present.
    knob_list = []

    ########################
    # Train with KDE:
    # - kernel = "gaussian" or "epanechnikov"
    # Train with TimeGAN:
    # - kernel = "timegan"
    ########################

    kernel = "timegan"
    bandwidths = [1] if kernel=="timegan" else np.linspace(0.25, 1.5, 6)
    for bandwidth in bandwidths:
        if t_m == 12:
            nu_val = None
            if t_m == 0 or t_m == 5 or t_m == 14:
                nu_val = 0.03125
            if t_m == 1 or t_m == 3 or t_m == 4 or t_m == 6 or t_m == 7 or t_m == 11 or t_m == 13 or t_m == 15:
                nu_val = 0.0625
            if t_m == 2 or t_m == 12:
                nu_val = 0.006428
            if t_m == 8 or t_m == 9 or t_m == 10:
                nu_val = 0.125
            for d in range(0, 1):
                master_dict = {}
                gen_dict = {}
                best_cols_val = 0
                best_cols = []
                check_list = []
                break_list = []
                gen_no = 0
                logging.info(f"{t_m}, {nu_val}")
                solution = list(generate_parents(master_list, pop_size))
                log_file = "one-class_" + str(t_m) + "_" + str(pop_size) + '_' + 'train_DBSCAN_PCA_10032022'
                log_file_name = log_file + ".txt"
                file1 = open(file_save_path + log_file_name, "w")
                file1.writelines(str(nu_val))
                file1.writelines("\n")
                cwd = os.getcwd()
                if "timegan" in kernel:
                    os.makedirs(f"{kernel}-k-clusters/", exist_ok=True)
                else:
                    os.makedirs(f"{kernel}-k-clusters/bw-{bandwidth}", exist_ok=True)

                for gen_no in tqdm(range(max_gen), desc=f"Generation: {gen_no}"):
                    if "timegan" in kernel:
                        storage_folder = f"{kernel}-k-clusters/{gen_no}gen"
                    else:
                        storage_folder = f"{kernel}-k-clusters/bw-{bandwidth}/{gen_no}gen"
                    os.makedirs(storage_folder, exist_ok=True, mode=0o751)
                    file1.writelines(str(gen_no))
                    file1.writelines("\n")
                    logging.info(f"Generation: {gen_no} !")
                    parentpath = os.getcwd()+'/'+storage_folder
                    # function1_values = Parallel(n_jobs=3)(delayed(function1)(i, solution, data, t_m, nu_val, parentpath, master_list, hps) for i in range(pop_size))
                    function1_values = [function1(i, solution, data=data, t_m=t_m, nu_val=nu_val, parentpath=parentpath, master_list=master_list, kernel=kernel, bandwidth=bandwidth, gen_no=gen_no, hps=hps)[1] for i in range(0, pop_size)]
                    #print(function1_values)
                    function2_values = [function2(solution[i]) for i in range(0, pop_size)]
                    for i in range(0, pop_size):
                        for values in range(0, len(solution[i])):
                            file1.writelines(solution[i][values])
                            file1.writelines('\t')
                        file1.writelines(str(function1_values[i]))
                        file1.writelines("\t")
                        file1.writelines(str(function2_values[i]))
                        file1.writelines("\n")
                        key_val = "GEN_" + str(gen_no) + "_" + str(i)
                        master_dict[key_val] = str(function1_values[i])

                    #print(function1_values)
                    gen_dict[gen_no] = sum(function1_values) / pop_size
                    #print(function2_values)

                    if (0 in function1_values):
                        mutation(function1_values, solution)

                    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
                    # print("The best front for Generation number ",gen_no, " is:")
                    # for valuez in non_dominated_sorted_solution[0]:
                    #    print((solution[valuez]),end=" ")
                    # print("\n")
                    crowding_distance_values = []
                    for i in range(0, len(non_dominated_sorted_solution)):
                        crowding_distance_values.append(
                            crowding_distance(function1_values[:], function2_values[:],
                                              non_dominated_sorted_solution[i][:]))
                    # print(crowding_distance_values)
                    solution2 = solution[:]
                    # Generating offsprings
                    for items in solution:
                        if ('LOT_CHILD' in items):
                            items.remove('LOT_CHILD')
                        if ('mask_data' in items):
                            items.remove('mask_data')
                        # print(items)

                    while (len(solution2) <= 2 * pop_size):
                        # print(len(solution2), 2*pop_size)
                        a1 = random.randint(0, pop_size - 1)
                        b1 = random.randint(0, pop_size - 1)
                        a, b = crossover(solution[a1], solution[b1])
                        # print(len(a), len(b))
                        solution2.append(a)
                        solution2.append(b)
                        # print(len(solution2))

                    function1_values2 = [function1(i, solution2, data=data, t_m=t_m, nu_val=nu_val, parentpath=parentpath, master_list=master_list, kernel=kernel, bandwidth=bandwidth, gen_no=gen_no, hps=hps)[1] for i in range(0, 2 * pop_size)]
                    function2_values2 = [function2(solution2[i]) for i in range(0, 2 * pop_size)]
                    # print(function1_values2, function2_values2)

                    for i in range(0, pop_size):
                        for values in range(0, len(solution2[i])):
                            file1.writelines(solution2[i][values])
                            file1.writelines('\t')
                        file1.writelines("\n")
                        file1.writelines(str(function1_values2[i]))
                        file1.writelines("\t")
                        file1.writelines(str(function2_values2[i]))
                        file1.writelines("\n")
                        key_val = "GEN_" + str(gen_no) + "_" + str(i)
                        master_dict[key_val] = str(function1_values2[i])

                    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
                    crowding_distance_values2 = []
                    for i in range(0, len(non_dominated_sorted_solution2)):
                        crowding_distance_values2.append(
                            crowding_distance(function1_values2[:], function2_values2[:],
                                              non_dominated_sorted_solution2[i][:]))
                    new_solution = []
                    for i in range(0, len(non_dominated_sorted_solution2)):
                        non_dominated_sorted_solution2_1 = [
                            index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                            range(0, len(non_dominated_sorted_solution2[i]))]
                        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
                        front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                                 range(0, len(non_dominated_sorted_solution2[i]))]
                        front.reverse()
                        for value in front:
                            new_solution.append(value)
                            if (len(new_solution) == pop_size):
                                break
                        if (len(new_solution) == pop_size):
                            break
                    for items in solution2:
                        if ('LOT_CHILD' in items):
                            items.remove('LOT_CHILD')
                        if ('mask_data' in items):
                            items.remove('mask_data')
                    solution = [solution2[i] for i in new_solution]
                    final_dict, diff_no = sort_master(master_dict, max_limit, check_list)
                    check_list = list(final_dict.keys())
                    if len(break_list) > max_limit - 1:
                        break_list.pop(0)
                    break_list.append(diff_no)
                    if len(list(set(break_list))) == 1 and list(set(break_list))[0] == 0:
                        file1.writelines("Broke based on stop condition")
                        file1.writelines('\t')
                        file1.writelines(str(gen_no))
                        file1.writelines('\n')
                        break
                    os.chdir(cwd)

                file1.writelines('\n')
                file1.writelines(str(best_cols_val))
                file1.close()
                plt.rcParams["figure.figsize"] = (40, 10)
                plt.plot(gen_dict.keys(), gen_dict.values(), linewidth=5)
                plt.suptitle('Generations v/s Accuracy', fontsize=25)
                plt.xticks(fontsize=25, rotation=60)
                plt.yticks(fontsize=25)
                plt.xlabel('Generations', fontsize=25)
                plt.ylabel('Accuracy', fontsize=25)
                graph_save_file = log_file + ".png"
                plt.savefig(file_save_path + graph_save_file, dpi=300, bbox_inches='tight')
                plt.show()
    # Plotting
    f1 = [i * 1 for i in function1_values]
    f2 = [j * -1 for j in function2_values]
    plt.xlabel('Function 1', fontsize=15)
    plt.ylabel('Function 2', fontsize=15)
    plt.scatter(f2, f1)
    plt.savefig('f1-vs-f2.png')


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    app.run(main)


"""
pool_size = 4
with Pool(pool_size) as pool:
    function1_values_list = []
    for i in range(0, pop_size, pool_size):
        vars = []
        for p in range(pool_size):
            if i+p>pop_size-1:
                break
            vars.append((i+p, solution, data, t_m, nu_val, os.getcwd()))
        function1_values_list.append(pool.starmap(function1, vars))

    function1_values = []
    for fvalues in function1_values_list:
        for fv in fvalues:
            function1_values.append(fv[1])

    #function1_values = function1_values_list[1]
    print(type(function1_values))
    print(function1_values)


with Pool(pool_size) as pool:
    function1_values2_list = []
    for i in range(0, pop_size, pool_size):
        vars = []
        for p in range(pool_size):
            if i+p>pop_size-1:
                break
            vars.append((i+p, solution, data, t_m, nu_val, os.getcwd()))
        function1_values2_list.append(pool.starmap(function1, vars))

    function1_values2 = []
    for fvalues in function1_values2_list:
        for fv in fvalues:
            function1_values2.append(fv[1])
"""
