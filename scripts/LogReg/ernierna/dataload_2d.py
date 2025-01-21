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

#-----------------------------------------------------------------------------

print('Loading model..')

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

print('Model successfully loaded and running in',device,'.')

#-----------------------------------------------------------------------------

import os
import json
import pandas as pd
from tqdm import tqdm

features = pd.DataFrame()

#sample = 20
for sample in ['8000', '25715']:
    count = 0
    print(f'Data preparation for {sample} batch..')
    
    with open(f'../../data/2d_batches/subtraining_{sample}.json') as f:
        rnas = json.load(f)

    for rna in tqdm(rnas):
        rnaid = rna['GeneralID']
        sequence = rna['Sequence']
        structure = rna['S_structure']
        length = len(sequence)

        try:
            count += 1
            # Attention--------------------------------------------------------------------
            seqs_lst = []
            seqs_lst.append(sequence)

            attnmap = extract_attnmap_of_ernierna(seqs_lst, ernie, attn_len=None, layer_idx = 13, head_idx = 12)
            attention_weights = torch.tensor(np.squeeze(attnmap))
            attention_weights = attention_weights[:, 1:-1, 1:-1]
            LH, S1, S2 = attention_weights.shape
            attention_weights = torch.reshape(attention_weights, (13, 12, S1, S2))#.numpy()

            # Structure--------------------------------------------------------------------

            structure_map = dot_bracket_to_matrix(structure)
            #structure = np.load('../../data/contacts95_nodiag/'+'pdb'+rnaid+'_contacts.npz')['b'] # Change source directory
            
            # MakeFeatures-----------------------------------------------------------------
            X , y = makefeatures(attention_weights, structure_map, 4, undersampling=True)

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

    np.savez_compressed(f'../../data/training_2d/ernie/training_{sample}.npz', x=X_train, y=y_train)


print('=============================END=============================')
