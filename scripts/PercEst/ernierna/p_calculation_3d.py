print('=======================RESOURCES=======================')
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
print('=========================START=========================')

import numpy as np 

def remove_padding(padded_attention, original_length):
    pad = 440 - original_length
    attention = padded_attention[:, :, :-pad, :-pad]
    return attention

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

def add_diagonal_link(adjacency_matrix, length):
    modified_matrix = np.copy(adjacency_matrix)

    for i in range(length - 1):
        modified_matrix[i, i + 1] = modified_matrix[i + 1, i] = 1

    return modified_matrix

def calculate_p(contactmap, attentionmap, th):
    
    l, h, i, j = attentionmap.shape
    numerator = np.zeros((l, h))
    denominator = np.zeros((l, h))
    
    attentionmap_mask = attentionmap #> th

    for ll in range(l):
        for hh in range(h):
            numerator[ll][hh] = np.sum(contactmap * attentionmap[ll, hh, :, :] * attentionmap_mask[ll, hh, :, :])
            denominator[ll][hh] = np.sum(attentionmap[ll, hh, :, :] * attentionmap_mask[ll, hh, :, :])
            #print(numerator[ll][hh], denominator[ll][hh])
    return numerator, denominator

#-----------------------------------------------------------------------------

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

print('Calculating molecule-wise propability..')
import json
from tqdm import tqdm

cutoff = '8'
with open('../../data/3dset.json') as f:
    rnas = json.load(f)

grand_numerator = np.zeros((13, 12))
grand_denominator = np.zeros((13, 12))
probability = np.zeros((13, 12))

ids_excluded = []

count = 0


if __name__ == "__main__":
    
#---# p_calculation original 
    for rna in tqdm(rnas):
        rnaid = rna['PDBcode']
        sequence = rna['Sequence']
        length = len(sequence)

        try:
            if length >= 20:
                # Take attention
                seqs_lst = []
                seqs_lst.append(sequence)
                #print(seqs_lst)
                
                #attnmap = extract_attnmap_of_ernierna(seqs_lst, attn_len=None, arg_overrides=args.arg_overrides, pretrained_model_path=args.ernie_rna_pretrained_checkpoint, device=args.device, layer_idx = args.layer_idx_attn, head_idx = args.head_idx_attn)
                attnmap = extract_attnmap_of_ernierna(seqs_lst, ernie, attn_len=None, layer_idx = 13, head_idx = 12)
                
                attention_weights = torch.tensor(np.squeeze(attnmap))
                attention_weights = attention_weights[:, 1:-1, 1:-1]
                LH, S1, S2 = attention_weights.shape
                attention_weights = torch.reshape(attention_weights, (13, 12, S1, S2)).numpy()
                #print(attention_weights.shape) 
                
                # Take structure (contacts)
                #structure_map = dot_bracket_to_matrix(structure)
                #structure_map = add_diagonal_link(dot_bracket_to_matrix(structure), length)
                structure_map = np.load(f'../../data/contacts{cutoff}_nodiag/pdb'+rnaid+'_contacts.npz')['b'] # Change input directory
                #print(structure_map.shape)

                # Check shape is the same
                if attention_weights[0][0].shape != structure_map.shape:
                    ids_excluded.append(rnaid)
                
                # Calculate molecule P
                th = 0.0
                numerator, denominator = calculate_p(structure_map, attention_weights, th)
                grand_numerator += numerator
                grand_denominator += denominator

                count += 1

        except:
            print('Error on:', rnaid)
            print('Difference is:',int(attention_weights[0][0].shape[0]) - int(structure_map.shape[0]))
            print(attention_weights[0][0].shape,  structure_map.shape)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disgard parallelizing / fork errors

print('Molecule-wise calculation done!')
print('Any excluded ids:', len(ids_excluded), ids_excluded)
print('Calculating summary probability on icluded...')
for l in range(13):
    for h in range(12):
        probability[l][h] = grand_numerator[l][h] / grand_denominator[l][h] if grand_denominator[l][h] != 0 else 0
probability = probability * 100
np.savez_compressed(f'ernie_probability-{cutoff}-{th}.npz', p=probability)

#-----------------------------------------------------------------------------

import matplotlib.pyplot as plt

def heatdouble(heat,th):
    heat_2d = heat.reshape(13,12)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot heatmap on the first subplot
    im = ax1.imshow(heat_2d, cmap='Blues', vmin = 0, vmax = 100)
    ax1.invert_yaxis()
    ax1.set_title('Th = ' + str(th))
    ax1.set_xlabel("Heads")
    ax1.set_ylabel("Layers")
    fig.colorbar(im, ax=ax1)

    # Plot vertical barplot on the second subplot
    max_values = np.max(heat_2d, axis=1)
    ax2.barh(np.arange(len(max_values)), max_values)
    ax2.set_title('Max Values')
    ax2.set_xlabel("Max Value")
    ax2.set_ylabel("Layer")
    ax2.set_yticks(np.arange(len(max_values)))
    ax2.set_yticklabels(np.arange(1, len(max_values)+1))

    plt.savefig(f'./ernie_3D-{cutoff}Å-{th}.pdf', format='pdf')

heatdouble(probability, th)

print('Included: ', count)
print('All done! Bye.')

print('==========================END==========================')



