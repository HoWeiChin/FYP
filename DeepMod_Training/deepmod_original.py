import sys
import pathlib
import numpy as np
import os

path_to_deep_mod = '/mnt/mbi/home/e0031794/Documents/FYP/DeePyMoD_improvements/src' #pls change path to deepmod when necessary
sys.path.append(path_to_deep_mod)
sys.path.append('/mnt/mbi/home/e0031794/Documents/FYP/FYP_results_11_9_2019/refactored_code')

from deepymod.DeepMoD import DeepMoD
from deepymod.library_functions import library_1D
from parameters import ParamGetter
from pde_util import PDEManager
from result_util import PdePrinter
from gpu_util import GPUSetter

def pipeline(data_dir, regex, L1_term, deepmod_type, epoch):
    #data_dir is an absolute path
    all_files_folders = os.listdir(data_dir)
    target_folders = [os.path.join(data_dir, obj) for obj in all_files_folders if regex in obj] #search for subfolders

    #get all necessary configs
    lib_config, libary_terms = PDEManager().get_lib_config_and_terms()
    training_config, nn_layer_config = ParamGetter().get_trng_layer_params()
    output_config = {"output_directory": '', 'X_predict': ''}

    nn_layer_config['lambda'] = L1_term
    training_config['max_iterations'] = epoch
    GPUSetter().set_max_gpu()

    #subset == amt of subset from original data. i.e amt of slicing done
    for folder in target_folders:

        if '.npy' in folder or '.txt' in folder or 'noisy' in folder:
            continue

        subset = folder.split('/')[-1].split('_')[0]  #get subset i.e 800, 900

        #if subset != '9700':
            #continue
                
        for_google_dir = os.path.join(folder + '/google_drive_storage_' + deepmod_type) #we want to store relevant outputs in a folder and upload into google drive
        pathlib.Path(for_google_dir).mkdir(parents=True, exist_ok=True)
        
        file_name = os.path.join(for_google_dir, subset + '_' + 'ML_parameters.txt')
        ML_param_file = open(file_name, 'w+')
        ML_param_file.write('lib config: ' + str(lib_config) + '\n')
        ML_param_file.write('PDE lib config: ' + str(libary_terms) + '\n')
        ML_param_file.write('Training config: ' + str(training_config) + '\n')
        ML_param_file.write('NN architecture config: ' + str(nn_layer_config) + '\n')
        ML_param_file.write('DeepMod version: ' + deepmod_type)
        ML_param_file.close()
        
        full_x_t = np.load(os.path.join(folder, 'space_time_data_full_' + subset + '.npy'))
        x_t_train = np.load(os.path.join(folder, 'space_time_data_sampled_' + subset + '.npy'))
        bicoid_train = np.load(os.path.join(folder, 'bicoid_sampled_' + subset + '.npy'))
        
        save_dir =  os.path.join(folder, 'Out_subset_' + subset + '_' + deepmod_type)
        output_config['output_directory'] = save_dir
        output_config['X_predict'] = full_x_t

        sparse_vector, denoised, bit_mask = DeepMoD(
            x_t_train, bicoid_train, nn_layer_config,
            library_1D, lib_config, training_config, 
            output_config
            )

        #save results
        save_file_path = os.path.join(for_google_dir, 'pde_result_subset_' + subset +  '.txt')
        PdePrinter().save_pde(sparse_vector, libary_terms, save_file_path)

        sparse_path = os.path.join(for_google_dir, 'sparse_vec_subset_' + subset)
        denoised_path = os.path.join(for_google_dir, 'denoised_result_subset_' + subset)

        #save training output and plots
        np.save(sparse_path, sparse_vector)
        np.save(denoised_path, denoised)
        np.save(os.path.join(for_google_dir, 'sparse_pattern_' + subset), bit_mask)

if __name__ == "__main__":
    L1_list = [10e-6]
    for L1_term in L1_list:
        pipeline('/mnt/mbi/home/e0031794/Documents/FYP/FYP_results_11_9_2019/try_big_conc/1_trial',
        regex='2000_rowsubset', L1_term=L1_term, deepmod_type='Original DeepMod', epoch=50000)
