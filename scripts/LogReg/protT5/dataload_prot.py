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

#-------------------LOADMODEL

mode = input("Please choose mode from [uniref50-half, uniref50-xl, bfd, bfd-xl] : ")

print('Loading model..')

if mode == 'uniref50-half':

    def remove_unwanted(attentions):
        attention_weights = []
        for layer in attentions:
            layer_attention_weights = []
            for head in layer:
                layer_attention_weights.append(head.detach().cpu().numpy())
            attention_weights.append(layer_attention_weights)
        attention_weights = np.squeeze(attention_weights, axis=1)
        attention_weights = attention_weights[:, :, :-1, :-1] # remove sep tokens
        #print(attention_weights.shape)
        
        return attention_weights
    
    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', output_attentions=True)
    model = model.to(device)

    num_layers = 24
    num_heads = 32

elif mode == 'uniref50-xl':

    def remove_unwanted(attentions):
        attention_weights = []
        for layer in attentions:
            layer_attention_weights = []
            for head in layer:
                layer_attention_weights.append(head.detach().cpu().numpy())
            attention_weights.append(layer_attention_weights)
        attention_weights = np.squeeze(attention_weights, axis=1)
        attention_weights = attention_weights[:, :, :-1, :-1] # remove sep tokens
        #print(attention_weights.shape)
        
        return attention_weights

    from transformers import T5EncoderModel, T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50', output_attentions=True)
    model = model.to(device)

    num_layers = 24
    num_heads = 32

elif mode == 'bfd':

    def remove_unwanted(attentions):
        attention_weights = []
        for layer in attentions:
            layer_attention_weights = []
            for head in layer:
                layer_attention_weights.append(head.detach().cpu().numpy())
            attention_weights.append(layer_attention_weights)
        attention_weights = np.squeeze(attention_weights, axis=1)
        attention_weights = attention_weights[:, :, 1:-1, 1:-1] # remove sep tokens
        #print(attention_weights.shape)
        
        return attention_weights

    from transformers import BertForMaskedLM, BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
    model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd", output_attentions=True)
    model = model.to(device)

    num_layers = 30
    num_heads = 16

elif mode == 'bfd-xl':

    def remove_unwanted(attentions):
        attention_weights = []
        for layer in attentions:
            layer_attention_weights = []
            for head in layer:
                layer_attention_weights.append(head.detach().cpu().numpy())
            attention_weights.append(layer_attention_weights)
        attention_weights = np.squeeze(attention_weights, axis=1)
        attention_weights = attention_weights[:, :, :-1, :-1] # remove sep tokens
        #print(attention_weights.shape)
        
        return attention_weights

    from transformers import T5Tokenizer, T5EncoderModel

    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_bfd', do_lower_case=False)
    model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_bfd', output_attentions=True)
    model = model.to(device)

    num_layers = 24
    num_heads = 32

else:
    print('No such model.. try again..')

print('Model successfully loaded and running in',device,'.')

#----------------------------

import os
import json
import pandas as pd
from tqdm import tqdm

undersampler = True

features = pd.DataFrame()
count = 0

with open('./tr_proteins.json') as f:
    proteins = json.load(f)

for protein in tqdm(proteins):
    proteinid = protein['PDBcode']
    sequence = protein['Sequence']
    length = len(sequence)

    try:
        count += 1
        # Get attention
        elements = []
        for char in sequence:
            elements.append(char)

        seq = ''
        for element in elements:
            seq = seq + element + ' '

        token_encoding = tokenizer(seq, return_tensors='pt').to(device)
        
        with torch.no_grad():
            result = model(**token_encoding, output_attentions=True)

        attentions = result.attentions
        attention_weights = torch.tensor(remove_unwanted(attentions))

        # Structure--------------------------------------------------------------------
        for contact in os.listdir('../../esm-2_3B/pdbs/'):
            if contact[:4] == proteinid:
                structure = np.load('../../esm-2_3B/pdbs/'+contact)['b']

        # MakeFeatures-----------------------------------------------------------------
        X , y = makefeatures(attention_weights, structure, 6, undersampling=undersampler)

        if count == 1:
            X_train = X
            y_train = y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    except:
        print(proteinid)

print('Samples:',count)
print('X:',X_train.shape,'y:', y_train.shape)
print('Saving..')

if undersampler:
    np.savez_compressed(f'./under_samples/20_tr_{mode}_under.npz', x=X_train, y=y_train)
else:
    np.savez_compressed(f'./samples/20_tr_{mode}.npz', x=X_train, y=y_train)

print('=============================END=============================')