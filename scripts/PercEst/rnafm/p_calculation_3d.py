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
    o = numerator[4][2]/denominator[4][2]

    return numerator, denominator, o

#-----------------------------------------------------------------------------

print('Loading model..')

import torch
import fm

model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model = model.to(device)
model.eval()  # disables dropout for deterministic results

print('Model successfully loaded and running in',device,'.')

#-----------------------------------------------------------------------------

print('Calculating molecule-wise propability..')
import json
from tqdm import tqdm

cutoff = '8'
with open('../../data/3dset.json') as f:
    rnas = json.load(f)

grand_numerator = np.zeros((12, 20))
grand_denominator = np.zeros((12, 20))
probability = np.zeros((12, 20))

ids_excluded = []

count = 0
for rna in tqdm(rnas):
    rnaid = rna['PDBcode']
    sequence = rna['Sequence']
    length = len(sequence)

    try:
        if length >= 20:
            # Save attention
            data = [(rnaid, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)

            with torch.no_grad():
                batch_tokens = batch_tokens.to(device)
                results = model(batch_tokens, repr_layers=[12], need_head_weights=True)

            attention_weights = results["attentions"]            
            attention_weights = attention_weights.squeeze()
            attention_weights = attention_weights[:, :, 1:-1, 1:-1].cpu().numpy()

            
            # Save tertiary structure (contacts)
            structure_mer = np.load(f'../../data/contacts{cutoff}_nodiag/pdb'+rnaid+'_contacts.npz')['b']
            #structure_1mer = add_diagonal_link(dot_bracket_to_matrix(structure), length)
            #structure_3mer = to_3mer(structure_1mer).numpy()
            
            # Check shape is the same
            if attention_weights[0][0].shape != structure_mer.shape:
                ids_excluded.append(rnaid)

            # Calculate molecule P
            th = 0.0
            numerator, denominator, o = calculate_p(structure_mer, attention_weights, th)
            if o >= 1:
                print(rnaid)
            grand_numerator += numerator
            grand_denominator += denominator

            count += 1

    except:
        print('Error on:', rnaid)
        print('Difference is:',int(attention_weights[0][0].shape[0]) - int(structure_mer.shape[0]))
        print(attention_weights[0][0].shape,  structure_mer.shape)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # disgard parallelizing / fork errors

print('Molecule-wise calculation done!')
print('Any excluded ids:', len(ids_excluded), ids_excluded)
print('Calculating and saving summary probability on icluded...')
for l in range(12):
    for h in range(20):
        probability[l][h] = grand_numerator[l][h] / grand_denominator[l][h] if grand_denominator[l][h] != 0 else 0
probability = probability * 100
np.savez_compressed(f'rnafm_probability-{cutoff}-{th}.npz', p=probability)

#-----------------------------------------------------------------------------

import matplotlib.pyplot as plt

def heatdouble(heat,th):
    heat_2d = heat.reshape(12,20)

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
    #plt.show()
    plt.savefig(f'./rnafm_3D-{cutoff}Å-{th}.pdf', format='pdf')

heatdouble(probability, th)

print('Included :', count)
print('All done! Bye.')

print('==========================END==========================')

