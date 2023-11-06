import pandas as pd
import numpy as np
import seaborn as sns
import sys
import os
sys.path.insert(0,"/Users/zyxu/Documents/R/23spring")
import torch
import json

def load_data(matrix_data_path = "./dataset/sim_matrix_data.csv", study_names_path = "./dataset/sim_study_name.csv"):
    # Load the matrix from the CSV file
    matrix_data = pd.read_csv(matrix_data_path)
    study_names = pd.read_csv(study_names_path)
    # print(study_names)
    if not 'study' in study_names.columns:
        study_names['study'] = study_names['x']
    return matrix_data, study_names

def construct_C_for_study(study, results, matrix_data):
    C = np.eye(matrix_data.shape[1])
    species_with_high_zeros = results.keys() & set(matrix_data.columns)
    for species in species_with_high_zeros:
        if study in results[species]:
            species_index = matrix_data.columns.get_loc(species)
            C[species_index, species_index] = 0
    return C


def pre_visz(
        matrix_data_path = "./dataset/sim_matrix_data.csv", 
        study_names_path = "./dataset/sim_study_name.csv",
        study_ground_truth_path_Us  ="./dataset/sim_study_groundTruth.json",
        study_ground_truth_path_V = None,
        optimized_V_path = "./output/sim/best_V.pt"
):  
    # print(matrix_data_path, study_names_path)
    matrix_data, study_names = load_data(matrix_data_path, study_names_path)

    rank = 20

    threshold = 0.98 * matrix_data.shape[0]
    cols_with_zeros = (matrix_data == 0).sum(axis=0) > threshold
    selected_species = matrix_data.columns[cols_with_zeros]
    threshold_study = 1

    results = {}

    ## result is a dict, each key is a species, each value of the key is a list containing studies which species has 0s on
    for species in selected_species:
        species_data = matrix_data[species]
        
        zero_data = study_names.copy()
        zero_data['zeros'] = (species_data == 0)
        
        # Count total and zero samples for each study
        total_samples_per_study = zero_data.groupby('study').size()
        zero_samples_per_study = zero_data.groupby('study').sum(numeric_only=True)['zeros']
        
        # Find studies with a high proportion of zeros
        high_zero_studies = (zero_samples_per_study / total_samples_per_study) >= threshold_study
        # print((zero_samples_per_study / total_samples_per_study))
        results[species] = high_zero_studies[high_zero_studies].index.tolist()

    unique_studies = study_names['study'].unique()


    ###### our method
    V = torch.load(optimized_V_path)
    optimized_V = V
    # Create C matrices for each study based on species with a high proportion of zeros
    # Constructing Cs for each study
    Cs = [construct_C_for_study(study, results, matrix_data) for study in unique_studies]

    # construct U and sigma
    studies_data = [matrix_data[study_names['study'] == study].values for study in unique_studies]
    study_decompositions = {}
    for i, study in enumerate(unique_studies):
        X = studies_data[i]
        print(np.dot(X, Cs[i]).shape)
        U, Sigma, _ = np.linalg.svd(np.dot(X, Cs[i]) @ V.detach().numpy(), full_matrices=False)
        U, Sigma = U[:, :rank], Sigma[:rank]
        study_decompositions[study] = (U, Sigma)

    


    ###### SVD
    # Assuming matrix_data is a DataFrame and study_names is a column vector of the same length
    unique_studies = study_names['study'].unique()
    study_decompositions_traditional = {}

    # Step 1: Perform SVD on the entire X matrix
    X_full = matrix_data.values
    U_full, Sigma, Vt_full = np.linalg.svd(X_full, full_matrices=False)

    print("rank: ", rank)

    Sigma = Sigma[:rank]
    Vt = Vt_full[:rank, :]


    for study in unique_studies:
        # Step 3: Select rows for the current study from the full U matrix
        study_rows = study_names['study'] == study
        U_study = U_full[study_rows, :rank]
        
        study_decompositions_traditional[study] = (U_study, Sigma, Vt.T)

    # =====

    # for study in unique_studies:
    #     # Step 1: Select rows for the current study
    #     X = matrix_data[study_names['x'] == study].values
        
    #     # Step 2: SVD decomposition
    #     U, Sigma, Vt = np.linalg.svd(X, full_matrices=False)
        
    #     # Step 3: Retain only top 1/10 singular values and vectors
    #     U = U[:, :rank]
    #     Sigma = Sigma[:rank]
    #     Vt = Vt[:rank, :]
        
    #     # Step 4: Construct C matrix
    #     C = np.eye(matrix_data.shape[1])
    #     species_with_high_zeros = results.keys() & set(matrix_data.columns)
    #     for species in species_with_high_zeros:
    #         if study in results[species]:
    #             species_index = matrix_data.columns.get_loc(species)
    #             C[species_index, species_index] = 0
        
    #     study_decompositions_traditional[study] = (U, Sigma, Vt, C)
    


    ###### ground truth
    # Load the JSON file
    with open(study_ground_truth_path_Us, 'r') as json_file:
        sim_study_groundTruth_Us = json.load(json_file)
    with open(study_ground_truth_path_V, 'r') as json_file:
        sim_study_ground_truth_V = json.load(json_file)
    
    return study_decompositions, study_decompositions_traditional, sim_study_groundTruth_Us, sim_study_ground_truth_V, optimized_V




