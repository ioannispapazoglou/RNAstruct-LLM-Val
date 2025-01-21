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

#---------------LOADLANGMODEL

lm = 'rnafm'

print(f'Loading {lm} LM...')

import torch
import fm

lmmodel, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
lmmodel = lmmodel.to(device)
lmmodel.eval()  # disables dropout for deterministic results

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
                    data = [(rnaid, sequence)]
                    batch_labels, batch_strs, batch_tokens = batch_converter(data)

                    with torch.no_grad():
                        batch_tokens = batch_tokens.to(device)
                        results = lmmodel(batch_tokens, repr_layers=[12], need_head_weights=True)

                    attention_weights = results["attentions"]            
                    attention_weights = attention_weights.squeeze()
                    attention_weights = attention_weights[:, :, 1:-1, 1:-1].cpu()#.numpy()


                    # Secondary--------------------------------------------------------------------

                    #structure = np.load(f'../../data/indipedents/contacts95_original/{rnaid.lower()}.pdb_contacts.npz')['b'] # Input !with! nearby contacts (diagonal)
                    structure = np.load(f'../../data/contacts95/pdb{rnaid}_contacts.npz')['b'] # Input !with! nearby contacts (diagonal)

                    # MakeFeatures-----------------------------------------------------------------
                    X , y_real = makefeatures(attention_weights, structure)

                    y_pred = chosen_model.predict(X)
                    y_pred = (y_pred >= 0.5)#.float().cpu()

                    #np.savez(f'../../benchmark/lms/{lm}3d/{rnaid}.npz', b=y_pred.reshape(length,length))

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
