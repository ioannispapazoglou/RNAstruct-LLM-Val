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
print(f"Target device is {device}, if specified..")
print('============================START============================')

import torch
import pandas as pd
import numpy as np

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

def makefeatures(attention_tensor, contacts_tensor):
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

    X = whole_array[:, :-1]  # Features array (excluding last column)
    y = whole_array[:, -1]  # Target array (last column)

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


#---------------------------


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


#---------------LOADLANGMODEL

lm = 'ernie'

print(f'Loading {lm} LM...')

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
 
lmmodel = loadernie(arg_overrides=args.arg_overrides, pretrained_model_path=args.ernie_rna_pretrained_checkpoint, device=device)

print(f'Language model successfully loaded in {device}.')

#----------------------------#

#---------LOAD MODEl---------#

import joblib

#N = input('Indicate smaple size : "20", "60", "140", "240", "all" : ')
#mode = input('Choose: "logreg", "mlp", "tr", "rf", "xgb", "lda", "knn", "gnb" or "exit" : ')

df = pd.DataFrame(columns=['PDBcode', 'CLS', 'Samples', 'F1 Score', 'MCC_score'])

for mode in ["logreg", "mlp", "tr", "rf", "xgb"]:
    for N in ["362"]:#"20", "60", "140", "240", "362"]:

        if mode == 'logreg':
            from sklearn.linear_model import LogisticRegression
            print('Running logistic regression. (cpu)')

            chosen_model = joblib.load(f"./experiments/logreg/diagmodels/{lm}_logreg_{N}.joblib")
            outname = f'{lm}_logreg_{N}'
            model = 'Logistic Regression'

        elif mode == 'mlp':
            from sklearn.neural_network import MLPClassifier
            print('Running multi-layer perceptron. (cpu)')

            chosen_model = joblib.load(f'./experiments/logreg/diagmodels/{lm}_mlp_{N}.joblib')
            outname = f'{lm}_mlp_{N}'
            model = 'Multi-Layer Perceptron'

        elif mode == 'tr':
            from sklearn import tree
            print('Running decision trees. (cpu)')

            chosen_model = joblib.load(f'./experiments/logreg/diagmodels/{lm}_tr_{N}.joblib')
            outname = f'{lm}_tr_{N}'
            model = 'Decision Trees'

        elif mode == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            print('Running random forest. (cpu)')

            chosen_model = joblib.load(f'./experiments/logreg/diagmodels/{lm}_rf_{N}.joblib')
            outname = f'{lm}_rf_{N}'
            model = 'Random Forest'

        elif mode == 'xgb':
            import xgboost as xgb
            print('Running xgboost classifier. (gpu)')

            chosen_model = joblib.load(f'./experiments/logreg/diagmodels/{lm}_xgb_{N}.joblib')
            outname = f'{lm}_xgb_{N}'
            model = 'XGBoost'

        print('ML model loaded. Loading test data..')

        import os
        import json
        from tqdm import tqdm

        total_f1 = 0
        total_mcc = 0

        #with open('../../data/indipedents/allsecstrdata.json') as f:
        with open('../../data/batches/test_15%.json') as f:
            rnas = json.load(f)

        count = 0

        for rna in tqdm(rnas):
            rnaid = rna['PDBcode']
            sequence = rna['Sequence']
            length = len(sequence)

            if rnaid == '4wfm_B.ent':

                try:
                    # Attention--------------------------------------------------------------------
                    seqs_lst = []
                    seqs_lst.append(sequence)
                        
                    attnmap = extract_attnmap_of_ernierna(seqs_lst, lmmodel, attn_len=None, layer_idx = 13, head_idx = 12)
                    
                    attention_weights = torch.tensor(np.squeeze(attnmap))
                    attention_weights = attention_weights[:, 1:-1, 1:-1]
                    LH, S1, S2 = attention_weights.shape
                    attention_weights = torch.reshape(attention_weights, (13, 12, S1, S2))#.numpy()
                    #print(attention_weights.shape) 
    

                    # Secondary--------------------------------------------------------------------
    
                    #structure = np.load(f'../../data/indipedents/contacts95_original/{rnaid.lower()}.pdb_contacts.npz')['b'] # Input !with! nearby contacts (diagonal)
                    structure = np.load(f'../../data/contacts95/pdb{rnaid}_contacts.npz')['b'] # Input !with! nearby contacts (diagonal)

                    # MakeFeatures-----------------------------------------------------------------
                    X , y_real = makefeatures(attention_weights, structure)

                    y_pred = chosen_model.predict(X)
                    y_pred = (y_pred >= 0.5)#.float().cpu()

                    np.savez(f'../../benchmark/lms/{lm}3d/{rnaid}.npz', b=y_pred.reshape(length,length))

                    from sklearn.metrics import f1_score
                    from sklearn.metrics import matthews_corrcoef

                    molecule_f1_score = f1_score(y_real, y_pred, average = 'macro')
                    molecule_mcc_score = matthews_corrcoef(y_real, y_pred)

                    total_f1 += molecule_f1_score
                    total_mcc += molecule_mcc_score

                    count += 1

                    # save individual scores to df
                    #df.loc[len(df)] = [rnaid, mode, N, round(molecule_f1_score,2), round(molecule_mcc_score,2)]

                    save = './experiments/logreg/diagvisuals/' + f'{rnaid}_3D_{lm}_{model}.pdf'
                    plot_result(y_real, y_pred, length, save, model, rnaid, molecule_f1_score)

                except:
                    print(rnaid)

        #df.loc[len(df)] = ['average', mode, N, round(total_f1/count,2), round(total_mcc/count,2)]

#df.to_csv(f'./experiments/logreg/analytical_evaluation_3D_{lm}_diag.csv', index=False)
#df.to_csv(f'./experiments/logreg/evaluation_3D_{lm}_diag.csv', index=False)
#df.to_csv(f'./experiments/logreg/indipedents_3D_{lm}_{N}_diag.csv', index=False)

print('Done. Bye!')
