print('==========================RESOURCES==========================')
import sys
import platform
import torch
import pandas as pd
import sklearn as sk

has_gpu = torch.cuda.is_available()
has_mps = getattr(torch,'has_mps',False)
device = "mps" if getattr(torch,'has_mps',False) \
    else "cuda" if torch.cuda.is_available() else "cpu"

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

def weigth_calculator(unpadded_target, coeff):
    
    n_samples = unpadded_target.shape[0]
    n_classes = 2

    weights = n_samples / (n_classes * torch.bincount(unpadded_target.reshape(-1).round().int()))
    if weights.shape == torch.Size([1]):
        weight = torch.ones([1])
    else: 
        weight = weights[1]

    return weight * coeff # they represent: [0, 1]

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
#-------------------ERNIE-RNA related functions----------------------


def seq_to_index(sequences):

    rna_len_lst = [len(ss) for ss in sequences]
    max_len = max(rna_len_lst)
    assert max_len <= 1022
    seq_nums = len(rna_len_lst)
    rna_index = np.ones((seq_nums,max_len+2))
    for i in range(seq_nums):
        for j in range(rna_len_lst[i]):
            if sequences[i][j] in set("Aa"):
                rna_index[i][j+1] = 5
            elif sequences[i][j] in set("Cc"):
                rna_index[i][j+1] = 7
            elif sequences[i][j] in set("Gg"):
                rna_index[i][j+1] = 4
            elif sequences[i][j] in set('TUtu'):
                rna_index[i][j+1] = 6
            else:
                rna_index[i][j+1] = 3
        rna_index[i][rna_len_lst[i]+1] = 2 # add 'eos' token
    rna_index[:,0] = 0 # add 'cls' token
    return rna_index, rna_len_lst


def extract_attnmap_of_ernierna(sequences, my_model, attn_len=None, layer_idx=13, head_idx=12):
    
    # Converts string to rna_index
    rna_index, rna_len_lst = seq_to_index(sequences)
    
    # extract embedding one by one
    if attn_len == None:
        attn_len = max(rna_len_lst)
    if head_idx == 12 and layer_idx == 13:
        attn_num = 156
    elif head_idx == 12 or layer_idx == 13:
        attn_num = head_idx if head_idx == 12 else layer_idx
    else:
        attn_num = 1
    rna_attn_map_embedding = np.zeros((len(sequences),attn_num,(attn_len+2), (attn_len+2)))
    with torch.no_grad():
        for i,(index,seq_len) in enumerate(zip(rna_index,rna_len_lst)):
            one_d, two_d = prepare_input_for_ernierna(index,seq_len)
            one_d = one_d.to(device)
            two_d = two_d.to(device)
            
            output = my_model(one_d,two_d,return_attn_map=True,i=layer_idx,j=head_idx).cpu().detach().numpy()
            
            rna_attn_map_embedding[i, :, :(seq_len+2), :(seq_len+2)] = output
        
    return rna_attn_map_embedding


#--------------------------------------------------------------------
#--------------------------------------------------------------------


import json
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn.functional as F

class CustomAttentionClassifier(nn.Module):
    def __init__(self, length):
        super(CustomAttentionClassifier, self).__init__()

        # Define a CNN to process the attention maps
        # Assuming attention maps are of size [layers, heads, length, length]
        self.cnn = nn.Sequential(
            nn.Conv2d(156, 64, kernel_size=3, padding=1), # 156 is 13 * 12, reshaped attention maps
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
        
        probabilities = F.softmax(logits, dim=1)
        
        return probabilities

#------------------------------------------------LOAD EVERYTHING

# Load CNN
length = 1024  # Set the pad length here
custom_model = CustomAttentionClassifier(length)
custom_model.to(device)

# Define optimizer
optimizer = AdamW(custom_model.parameters(), lr=2e-5)

# Define training parameters
num_epochs = 5
patience = 2
best_loss = float('inf')
num_epochs_without_improvement = 0
weigth_coeff = 1 # weigth scaling / intensity for contact prediction reward

# Load LM
lm = 'ernie'

import os
import time
import torch
import argparse
import numpy as np

from src.ernie_rna.tasks.ernie_rna import *
from src.ernie_rna.models.ernie_rna import *
from src.ernie_rna.criterions.ernie_rna import *
from src.utils import ErnieRNAOnestage, read_text_file, load_pretrained_ernierna, prepare_input_for_ernierna


def loadernie(arg_overrides = { "data": '/data/code/BERT/onestage_checkpoint_dict/' }, pretrained_model_path =  '/data/code/BERT/Pretrain_checkpoint/twocheckpoint_best.pt', device='cuda'):
    
    model_pretrained = load_pretrained_ernierna(pretrained_model_path,arg_overrides)
    ernie = ErnieRNAOnestage(model_pretrained.encoder).to(device)
    print('Model Loading Done!!!')
    ernie.eval()

    return ernie

parser = argparse.ArgumentParser()
parser.add_argument("--save_path", default='./results/ernie_rna_representations/test_seqs/', type=str, help="The path of output extracted by ERNIE-RNA")
parser.add_argument("--arg_overrides", default={ "data": './src/dict/' }, help="The path of vocabulary")
parser.add_argument("--ernie_rna_pretrained_checkpoint", default='./checkpoint/ERNIE-RNA_checkpoint/ERNIE-RNA_pretrain.pt', type=str, help="The path of ERNIE-RNA checkpoint")
parser.add_argument("--layer_idx_emb", default=12, type=int, help="The layer idx of which we extract embedding from, 12 for all layers")
parser.add_argument("--layer_idx_attn", default=13, type=int, help="The layer idx of which we extract attnmap from, 13 for all layers")
parser.add_argument("--head_idx_attn", default=12, type=int, help="The head idx of which we extract attnmap from, 12 for all heads")
parser.add_argument("--device", default=0, type=int, help="device")
args = parser.parse_args()

ernie = loadernie(arg_overrides=args.arg_overrides, pretrained_model_path=args.ernie_rna_pretrained_checkpoint, device=device)

# Load json dataset
with open(f'../../data/2d_batches/subtraining_25715.json') as f:
    trainingrnas = json.load(f)

with open(f'../../data/2d_batches/cnnvalidation.json') as f:
    validationrnas = json.load(f)

with open(f'../../data/2d_batches/cnntest.json') as f:
    testrnas = json.load(f)

#---------------------------------------------------TRAIN MODEL

#Discard gpu fork errors
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print('Training..')

for epoch in range(num_epochs):
    custom_model.train().to(device)

    for rna in tqdm(trainingrnas): # Every batch constitutes a molecule 
        rnaid = rna['GeneralID']
        sequence = rna['Sequence']
        length = len(sequence)
        structure = rna['S_structure']

        if length <= 1022:
            batch_loss = 0.0
            # Attention--------------------------------------------------------------------
            seqs_lst = []
            seqs_lst.append(sequence)

            attnmap = extract_attnmap_of_ernierna(seqs_lst, ernie, attn_len=None, layer_idx = 13, head_idx = 12)
            attention_weights = torch.tensor(np.squeeze(attnmap))
            attention_weights = attention_weights[:, 1:-1, 1:-1]
            LH, S1, S2 = attention_weights.shape
            attention_weights = torch.reshape(attention_weights, (13, 12, S1, S2))

            attention_maps = pad_attention(attention_weights, 1024)
            attention_maps = attention_maps.view(1, 156, 1024, 1024).float().to(device) #(1, LxH, maxl, maxl)

            # Structure--------------------------------------------------------------------
            structure_map = torch.tensor(dot_bracket_to_matrix(structure)).to(torch.float32).to(device)

            # Mask-------------------------------------------------------------------------
            strmask = create_mask(length, 1024)

            # Forward pass
            logits = custom_model(attention_maps).reshape(-1, 1024, 1024)  # Pass the input tensors and masks to the model

            prediction = remove_padding(strmask, logits[0])

            weight = weigth_calculator(structure_map, weigth_coeff)
                
            criterion = nn.BCEWithLogitsLoss(pos_weight = weight).to(device)

            batch_loss = criterion(prediction.unsqueeze(0), structure_map.unsqueeze(0))

            # Backward pass and optimization
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    
    # Validation
    custom_model.eval()
    with torch.no_grad():

        validation_loss = 0.0
        count = 0 
        for rna in tqdm(validationrnas):
            rnaid = rna['GeneralID']
            sequence = rna['Sequence']
            length = len(sequence)
            structure = rna['S_structure']
            
            if length <= 1022:
                count += 1
                batch_loss = 0.0
                # Attention--------------------------------------------------------------------
                seqs_lst = []
                seqs_lst.append(sequence)

                attnmap = extract_attnmap_of_ernierna(seqs_lst, ernie, attn_len=None, layer_idx = 13, head_idx = 12)
                attention_weights = torch.tensor(np.squeeze(attnmap))
                attention_weights = attention_weights[:, 1:-1, 1:-1]
                LH, S1, S2 = attention_weights.shape
                attention_weights = torch.reshape(attention_weights, (13, 12, S1, S2))

                attention_maps = pad_attention(attention_weights, 1024)
                attention_maps = attention_maps.view(1, 156, 1024, 1024).float().to(device) #(1, LxH, maxl, maxl)

                # Structure--------------------------------------------------------------------
                structure_map = torch.tensor(dot_bracket_to_matrix(structure)).to(torch.float32).to(device)

                # Mask-------------------------------------------------------------------------
                strmask = create_mask(length, 1024)

                # Forward pass
                logits = custom_model(attention_maps).reshape(-1, 1024, 1024)  # Pass the input tensors and masks to the model

                prediction = remove_padding(strmask, logits[0])

                weight = weigth_calculator(structure_map, weigth_coeff)
                    
                criterion = nn.BCEWithLogitsLoss(pos_weight = weight).to(device)

                batch_loss = criterion(prediction.unsqueeze(0), structure_map.unsqueeze(0))
                validation_loss += batch_loss.item()


    avg_loss = validation_loss / count
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_loss}")

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        num_epochs_without_improvement = 0
        # Save the best model
        torch.save(custom_model.state_dict(), './experiments/cnn/cnn.pt')
    else:
        num_epochs_without_improvement += 1
        if num_epochs_without_improvement >= patience:
            print("Early stopping! No improvement for", patience, "epochs.")
            break
    

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
        seqs_lst = []
        seqs_lst.append(sequence)

        attnmap = extract_attnmap_of_ernierna(seqs_lst, ernie, attn_len=None, layer_idx = 13, head_idx = 12)
        attention_weights = torch.tensor(np.squeeze(attnmap))
        attention_weights = attention_weights[:, 1:-1, 1:-1]
        LH, S1, S2 = attention_weights.shape
        attention_weights = torch.reshape(attention_weights, (13, 12, S1, S2))

        attention_maps = pad_attention(attention_weights, 1024)
        attention_maps = attention_maps.view(1, 156, 1024, 1024).float().to(device) #(1, LxH, maxl, maxl)

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

        if count <= 10:
            save = './experiments/cnn/visuals/' + f'{rnaid}_2D_{lm}_cnn.pdf'
            plot_result(structure_map, prediction, length, save, 'CNN', rnaid, molecule_f1_score)

    summary.loc[len(summary)] = ['average', 'CNN', int('25715'), round(total_f1/count,2), round(total_mcc/count,2)]

analytical.to_csv(f'./experiments/cnn/analytical_evaluation_2D_{lm}.csv', index=False)
summary.to_csv(f'./experiments/cnn/evaluation_2D_{lm}.csv', index=False)
        
print('===================================END===================================')
