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
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

def preprocess(attention):
    attention = symmetrize(attention)
    attention = apc(attention)
    return attention

def makefeatures(attention_tensor, contacts_tensor, localcontacts, undersampling):
    #-------ATTENTION
    #----------------------PREPROCESS--------------------------------------------------------------------
    preprocessed_attn = preprocess(attention_tensor) # OR ANY OTHER PREPROCESSING STEP TO BE INSERTED HERE
    #----------------------------------------------------------------------------------------------------
    tensor_shape = preprocessed_attn.shape
    num_samples = tensor_shape[2]*tensor_shape[3]
    transposed_tensor = preprocessed_attn.permute(2, 3, 0, 1)
    attention_array = transposed_tensor.reshape((num_samples, tensor_shape[0]*tensor_shape[1]))

    #-------SECONDARYCONTACTS
    tensor_shape = contacts_tensor.shape
    reshaped_tensor = contacts_tensor.reshape((tensor_shape[0]*tensor_shape[1],))
    contacts_array = reshaped_tensor

    # Combine attention and contacts arrays
    whole_array = np.concatenate((attention_array, contacts_array.reshape(-1, 1)), axis=1)

    # Filter local contacts
    ilist = np.repeat(np.arange(contacts_tensor.shape[0]), contacts_tensor.shape[0])
    jlist = np.tile(np.arange(contacts_tensor.shape[0]), contacts_tensor.shape[0])
    mask = np.abs(ilist - jlist) > localcontacts
    filtered_array = whole_array[mask]

    X = whole_array[:, :-1]  # Features array (excluding last column)
    y = whole_array[:, -1]  # Target array (last column)

    if undersampling:
        under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
        X_resampled, y_resampled = under_sampler.fit_resample(X, y) # Create final dataset balanced

        return X_resampled, y_resampled
    
    else:
        return X, y

def dot_bracket_to_matrix(dot_bracket):
    
    matrix = [[0] * len(dot_bracket) for _ in range(len(dot_bracket))]
    memory1 = []
    memory2 = []

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

    adjacency_matrix = np.array(matrix)

    return adjacency_matrix

def remove_padding(padded_attention, original_length):
    pad = 440 - original_length
    attention = padded_attention[:, :, :-pad, :-pad]
    return attention

#-------------------LOADMODEL

print('Loading model..')
import torch
from rinalmo.pretrained import get_pretrained_model

model, alphabet = get_pretrained_model(model_name="giga-v1")
model = model.to(device=device)
model.eval()

print('Model successfully loaded and running in',device,'.')

#-----------------------------------------------------------------------------

import os
import json
import pandas as pd
from tqdm import tqdm

features = pd.DataFrame()

#sample = 20

#sample = 20
for sample in ['20','60','140','240','362']:
    count = 0
    with open(f'../../data/batches/subtraining_{sample}.json') as f:
        rnas = json.load(f)

    for rna in tqdm(rnas):
        rnaid = rna['PDBcode']
        sequence = rna['Sequence']
        length = len(sequence)

        try:
            count += 1
            # Attention--------------------------------------------------------------------
            seqs = []
            seqs.append(sequence)

            tokens = torch.tensor(alphabet.batch_tokenize(seqs), dtype=torch.int64, device=device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model(tokens)

            attn = torch.squeeze(outputs["attentions"])
            attention_weights = attn[:, :, 1:-1, 1:-1].cpu()#.numpy()

            # Structure--------------------------------------------------------------------

            #structure = np.load('../../data/contacts95_nodiag/'+'pdb'+rnaid+'_contacts.npz')['b'] # Change source directory
            structure = np.load('../../data/contacts95/'+'pdb'+rnaid+'_contacts.npz')['b'] # Change source directory
            
            # MakeFeatures-----------------------------------------------------------------
            X , y = makefeatures(attention_weights, structure, 0, undersampling=True)

            if count == 1:
                X_train = X
                y_train = y
            else:
                X_train = np.concatenate((X_train, X), axis=0)
                y_train = np.concatenate((y_train, y), axis=0)

        except:
            print(rnaid)

    print('Samples:',count)
    print('X:',X_train.shape,'y:', y_train.shape)
    print('Saving..')

    #np.savez_compressed(f'../../data/training/rinalmo/training_{sample}.npz', x=X_train, y=y_train)
    np.savez_compressed(f'../../data/diagtest/rinalmo/training_{sample}.npz', x=X_train, y=y_train)


print('=============================END=============================')
