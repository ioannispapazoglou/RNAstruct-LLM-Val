# Predicting Biomolecular Structure from Large Language Model attention


## INSTALLATION

To repeat analysis install and run the Python scripts inside the conda environment:

conda env create -f attentionpred.yml

Usage instructions and installation steps for Miniconda / Anaconda refer to the official web page: https://docs.anaconda.com/free/miniconda/index.html

For LLM inference without utilization of the transformers library, run codes after installing their suggested conda encironments:

RiNALMo: https://github.com/lbcb-sci/RiNALMo
RNA-FM: https://github.com/ml4bio/RNA-FM
ERNIE-RNA: https://github.com/Bruce-ywj/ERNIE-RNA


## EXECUTION

Note: Make sure you change the input / output directories correspondingly!

#### Data generation:

1. Execute multicontacts.py to generate contact maps (mind the distance cutoff value)
2. Execute del_diag.py to delete nearby contacts (mind the neighbors hyperparameter value)

#### Language Model selection:

1. Execute p_calculation.py scripts per LLM (mind the thresshold θ value)

#### Structure prediction:

1.	Execute dataload.py for feature extraction and training dataset generation
2.	Execute training.py and predict.py for classifier training (or cnn finetuning) and evaluation
