print('==========================RESOURCES==========================')
import sys
import platform
import torch
import pandas as pd
import sklearn as sk

has_gpu = torch.cuda.is_available()
has_mps = getattr(torch,'has_mps',False)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Python Platform: {platform.platform()}")
print(f"PyTorch Version: {torch.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if has_gpu else "NOT AVAILABLE")
print("MPS (Apple Metal) is", "AVAILABLE" if has_mps else "NOT AVAILABLE")
print(f"Target device is {device}")
print('============================START============================')

import torch
import numpy as np

def dot_bracket_to_matrix(dot_bracket):
    
    matrix = [[0] * len(dot_bracket) for _ in range(len(dot_bracket))]
    memory1 = []
    memory2 = []
    memory3 = []
    memory4 = []
    memory5 = []

    for i, char in enumerate(dot_bracket):
        if char == '(' :
            memory1.append(i)
        elif char == ')' :
            j = memory1.pop()
            matrix[j][i] = matrix[i][j] = 1
        elif char == '[' :
            memory2.append(i)
        elif char == ']' :
            j = memory2.pop()
            matrix[j][i] = matrix[i][j] = 1
        elif char == '{' :
            memory3.append(i)
        elif char == '}' :
            j = memory3.pop()
            matrix[j][i] = matrix[i][j] = 1
        elif char == '<' :
            memory4.append(i)
        elif char == '>' :
            j = memory4.pop()
            matrix[j][i] = matrix[i][j] = 1
        elif char == 'A' :
            memory5.append(i)
        elif char == 'a' :
            j = memory5.pop()
            matrix[j][i] = matrix[i][j] = 1

    adjacency_matrix = np.array(matrix)

    return adjacency_matrix

def pad_structure(structure_maps, pad):
    
    i, j = structure_maps.shape
    padded_structure_maps = np.pad(structure_maps, ((0, pad - i), (0, pad - j)), mode='constant')
    padded_structure_maps = torch.tensor(padded_structure_maps, dtype=torch.float32)
    
    return padded_structure_maps

def pad_attention(attention_maps, pad):

    L, H, l, _ = attention_maps.shape
    padded_attention_maps = torch.zeros((L, H, pad, pad), dtype=attention_maps.dtype)

    for i in range(L):
        for j in range(H):
            padded_attention_maps[i, j, :l, :l] = attention_maps[i, j, :, :]

    return padded_attention_maps

def create_mask(length, max_length):

    mask = torch.zeros(max_length, max_length)
    mask[:length, :length] = 1

    return mask 

def remove_padding(binary_mask, padded_map):

    binary_mask = binary_mask.bool()  # Convert to boolean (T / F)
    
    indices = torch.nonzero(binary_mask)
    
    min_indices, _ = torch.min(indices, dim=0)
    max_indices, _ = torch.max(indices, dim=0)
    
    unpadded_map = padded_map[
        min_indices[0]:max_indices[0] + 1,
        min_indices[1]:max_indices[1] + 1
    ]
    
    return unpadded_map

def weigth_calculator(unpadded_target):
    
    n_samples = unpadded_target.shape[0]
    n_classes = 2

    weights = n_samples / (n_classes * torch.bincount(unpadded_target.reshape(-1).round().int()))
    if weights.shape == torch.Size([1]):
        weight = torch.ones([1])
    else: 
        weight = weights[1]

    return weight # they represent: [0, 1]

import seaborn as sns
import matplotlib.pyplot as plt

def plot_result(real, predictions, len, save_path_name, model, pdb, f1):
    real = real.reshape(len,len)
    predictions = predictions.reshape(len,len)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].set_title(f'Secondary Structure: {pdb}')
    axs[0].set_xlabel("Sequence")
    axs[0].set_ylabel("Sequence")
    sns.heatmap(real, cmap='binary', ax=axs[0], vmin=0, vmax=1, cbar=False)
    axs[0].invert_yaxis()

    axs[1].set_title(f'{model} prediction - F1 score: {f1:.2f}')
    axs[1].set_xlabel("Sequence")
    axs[1].set_ylabel("Sequence")
    sns.heatmap(predictions, cmap='binary', ax=axs[1], vmin=0, vmax=1, cbar=False)
    axs[1].invert_yaxis()

    plt.tight_layout()

    # Save or Show the contact maps as a single PNG file
    plt.savefig(save_path_name)
    #plt.show()
    plt.close(fig)

#--------------------------------------------------------------------

import json
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

class CustomAttentionClassifier(nn.Module):
    def __init__(self, length):
        super(CustomAttentionClassifier, self).__init__()

        # Define a CNN to process the attention maps
        # Assuming attention maps are of size [layers, heads, length, length]
        self.cnn = nn.Sequential(
            nn.Conv2d(660, 64, kernel_size=3, padding=1), # 660 is 33 * 20, reshaped attention maps
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * (length//4) * (length//4), 128), # Replace `length` with the actual length
            nn.ReLU(),
            nn.Linear(128, length*length)
        )
        
    def forward(self, attention_maps):

        # Pass attention maps through CNN
        logits = self.cnn(attention_maps)
        
        return logits

#------------------------------------------------LOAD EVERYTHING

# Load CNN
length = 1024  # Set the pad length here
custom_model = CustomAttentionClassifier(length)
state_dict = torch.load('./experiments/cnn/cnn.pt', map_location=device)
custom_model.load_state_dict(state_dict)
custom_model.to(device)

# Define optimizer
optimizer = AdamW(custom_model.parameters(), lr=2e-5)

# Define training parameters
num_epochs = 5
patience = 2
best_loss = float('inf')
num_epochs_without_improvement = 0

# Load LM
import torch
from rinalmo.pretrained import get_pretrained_model
lm = 'rinalmo'
lmmodel, alphabet = get_pretrained_model(model_name="giga-v1")
lmmodel = lmmodel.to(device=device)
lmmodel.eval()

# Load json test set
with open(f'../../data/2d_batches/cnntest.json') as f:
    testrnas = json.load(f)

#---------------------------------------------------TRAIN MODEL

#Discard gpu fork errors
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print('Evaluating on test set..')

from sklearn.metrics import f1_score, matthews_corrcoef

count = 0 
analytical = pd.DataFrame(columns=['PDBcode', 'CLS', 'Samples', 'F1 Score', 'MCC_score'])
summary = pd.DataFrame(columns=['PDBcode', 'CLS', 'Samples', 'F1 Score', 'MCC_score'])

total_f1 = 0
total_mcc = 0

custom_model.eval()
with torch.no_grad():

    for rna in tqdm(testrnas):
        rnaid = rna['GeneralID']
        sequence = rna['Sequence']
        length = len(sequence)
        structure = rna['S_structure']
        
        
        # Attention--------------------------------------------------------------------
        seqs = []
        seqs.append(sequence)
        tokens = torch.tensor(alphabet.batch_tokenize(seqs), dtype=torch.int64, device=device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = lmmodel(tokens)
        attn = torch.squeeze(outputs["attentions"])
        
        attention_maps = pad_attention(attn[:, :, 1:-1, 1:-1], 1024)
        attention_maps = attention_maps.view(1, 660, 1024, 1024).to(device) #(1, LxH, maxl, maxl)

        # Structure--------------------------------------------------------------------
        structure_map = dot_bracket_to_matrix(structure)

        # Mask-------------------------------------------------------------------------
        strmask = create_mask(length, 1024)

        # Forward pass
        logits = custom_model(attention_maps).reshape(-1, 1024, 1024)  # Pass the input tensors and masks to the model
        prediction = remove_padding(strmask, logits[0])

        prediction = (prediction >= 0.5).cpu().numpy()

        molecule_f1_score = f1_score(structure_map.flatten(), prediction.flatten(), average = 'macro')
        molecule_mcc_score = matthews_corrcoef(structure_map.flatten(), prediction.flatten())

        total_f1 += molecule_f1_score
        total_mcc += molecule_mcc_score
        count += 1

        # save individual scores to df
        analytical.loc[len(analytical)] = [rnaid, 'CNN', int('25715'), round(molecule_f1_score,2), round(molecule_mcc_score,2)]

        #save = './experiments/cnn/visuals/' + f'{rnaid}_2D_{lm}_cnn.pdf'
        #plot_result(structure_map, prediction, length, save, 'CNN', rnaid, molecule_f1_score)

    summary.loc[len(summary)] = ['average', 'CNN', int('25715'), round(total_f1/count,2), round(total_mcc/count,2)]

analytical.to_csv(f'./experiments/cnn/analytical_evaluation_2D_{lm}.csv', index=False)
summary.to_csv(f'./experiments/cnn/evaluation_2D_{lm}.csv', index=False)
        
print('===================================END===================================')
