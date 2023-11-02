import pandas as pd
import numpy as np
import seaborn as sns
import sys
import os
# os.chdir("/Users/zyxu/Documents/R/23spring")
os.chdir("D:\OneDrive\OneDrive - UW-Madison\Kris\Code\curatedMetagenomicDataAnalyses\\notebook")
import torch
import torch.optim as optim
import time

def load_file():
    # Load the matrix from the CSV file
    matrix_data = pd.read_csv("CRC_matrix_data.csv")
    study_names = pd.read_csv("CRC_study_name.csv")
    study_names['study'] = study_names['x']
    return matrix_data, study_names

def select_species(matrix_data, study_names):
    threshold = 0.98 * matrix_data.shape[0]
    cols_with_zeros = (matrix_data == 0).sum(axis=0) > threshold
    selected_species = matrix_data.columns[cols_with_zeros]
    
    threshold_study = 1

    results = {}

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

    return results

def compute_loss(V, matrix_data, study_names, results):
        matrix_data_tensor = matrix_data.values
        total_loss = 0.0
        unique_studies = study_names['x'].unique()
        for study in unique_studies:
            study_indices = (study_names['x'] == study).values.nonzero()[0]
            X_i = matrix_data_tensor[study_indices]
            X_i = torch.tensor(X_i,dtype=torch.float32)
            
            # Construct Ci matrix
            # C_i = torch.eye(935)
            C_i = torch.eye(200)
            for species, high_zero_studies in results.items():
                if study in high_zero_studies:
                    species_idx = matrix_data.columns.get_loc(species)
                    C_i[species_idx, species_idx] = 0
            
            # Compute U, Sigma for current study and given V in a memory-efficient manner
            # print(X_i.dtype)
            # print(C_i.dtype)
            # print(V.dtype)
            
            intermediate = X_i @ C_i @ V
            U_i, Sigma_i, _ = torch.svd(intermediate)
            # print(Sigma_i)
            Sigma_i = Sigma_i[:10]  # Keep top 10 singular values
            
            
            # Compute the residual for current study
            # print(U_i.shape,torch.diag_embed(Sigma_i).shape,V.T.shape )
            # print((X_i @ C_i).shape)
            residual = X_i @ C_i - U_i @ torch.diag_embed(Sigma_i) @ V.T
            total_loss += torch.norm(residual, p='fro') ** 2

        return total_loss

def time_str(t):
  if t >= 3600:
    return '{:.1f}h'.format(t / 3600) # time in hours
  if t >= 60:
    return '{:.1f}m'.format(t / 60)   # time in minutes
  return '{:.1f}s'.format(t)

def train(matrix_data, study_names, results):
    # Adjusting the computation for memory efficiency

    # Define the loss function with memory-efficient computations
    _, _, Vt = torch.svd(torch.tensor(matrix_data.values))
    V = Vt[:10].t().type(torch.float32).requires_grad_(True)
    # print(V.shape)
    # assert False
    # Optimizer
    optimizer = optim.Adam([V], lr=0.01)
    start = time.time()
    # Optimization loop
    min_loss = 2e16
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = compute_loss(V, matrix_data, study_names, results)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item()} ---- time elapsed {time_str(time.time() - start)} | {time_str((time.time() - start)/(epoch+1)*num_epochs)}")
            torch.save(V, f'output/epoch{epoch + 1}_V.pt')

        if loss.item() <= min_loss:
                torch.save(V, 'output/best_V.pt')
                min_loss = loss.item()
                print(f"save best at epoch {epoch+1}!")
                
        torch.save(V, f'output/epoch_last_V.pt')

    optimized_V = V.detach().numpy()
    return optimized_V

def main():
    matrix_data, study_names = load_file()
    results = select_species(matrix_data, study_names)

    V = train(matrix_data, study_names, results)
    print(V.shape)
    print(V)

    

if __name__ == "__main__":
    main()