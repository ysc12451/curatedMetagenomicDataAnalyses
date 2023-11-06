import pandas as pd
import numpy as np
import seaborn as sns
import sys
import os

# Get the directory of the run.py script
run_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of run_dir, which is the project directory
project_dir = os.path.dirname(run_dir)

# Add the project directory to the Python path
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)
import torch
import torch.optim as optim
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_data", type=str, default="CRC_matrix_data.csv")
    parser.add_argument("--study_names", type=str, default="CRC_study_name.csv")
    parser.add_argument("--output_path", type=str, default="output")
    parser.add_argument("--rank", type=int, default=10)

    return parser.parse_args()

def load_file(args):
    # Load the matrix from the CSV file
    matrix_data = pd.read_csv(args.matrix_data)
    study_names = pd.read_csv(args.study_names)
    if not 'study' in study_names.columns:
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
        if not 'study' in study_names.columns:
            study_names['study'] = study_names['x']
        unique_studies = study_names['study'].unique()
        for study in unique_studies:
            study_indices = (study_names['study'] == study).values.nonzero()[0]
            X_i = matrix_data_tensor[study_indices]
            X_i = torch.tensor(X_i,dtype=torch.float32)
            
            # Construct Ci matrix
            C_i = torch.eye(matrix_data_tensor.shape[1])
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
            Sigma_i = Sigma_i[:V.shape[1]]  # Keep top 10 singular values
            
            
            # Compute the residual for current study
            # print(U_i.shape,torch.diag_embed(Sigma_i).shape,V.T.shape )
            # print((X_i @ C_i).shape)
            residual = X_i - U_i @ torch.diag_embed(Sigma_i) @ V.T @ C_i
            total_loss += torch.norm(residual, p='fro') ** 2

        return total_loss

def time_str(t):
  if t >= 3600:
    return '{:.1f}h'.format(t / 3600) # time in hours
  if t >= 60:
    return '{:.1f}m'.format(t / 60)   # time in minutes
  return '{:.1f}s'.format(t)

def train(args, matrix_data, study_names, results):
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = None
    # Adjusting the computation for memory efficiency

    # Define the loss function with memory-efficient computations
    _, _, Vt = torch.svd(torch.tensor(matrix_data.values))
    V = Vt[:args.rank].t().type(torch.float32).requires_grad_(True)
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
            torch.save(V, f'{output_path}/epoch{epoch + 1}_V.pt')

        if loss.item() <= min_loss:
                torch.save(V, f'{output_path}/best_V.pt')
                min_loss = loss.item()
                print(f"save best at epoch {epoch+1}!")
                
        torch.save(V, f'{output_path}/epoch_last_V.pt')

    optimized_V = V.detach().numpy()
    return optimized_V

def main():
    args = parse_args()
    matrix_data, study_names = load_file(args)
    results = select_species(matrix_data, study_names)

    V = train(args, matrix_data, study_names, results)
    print(V.shape)
    print(V)

    

if __name__ == "__main__":
    main()