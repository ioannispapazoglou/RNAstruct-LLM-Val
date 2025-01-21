import json
import random

inputfile = './2d_batches/test_15%.json'
n = 100  # Number of molecules to be included in the random sample
outputfile_random = './2d_batches/cnntest.json'
outputfile_remaining = './2d_batches/cnnvalidation.json'

with open(inputfile, 'r') as file:
    rnas = json.load(file)

random_samples = random.sample(rnas, n)

with open(outputfile_random, 'w') as file:
    json.dump(random_samples, file, indent=4)

# Save the remaining samples in validation
remaining_samples = [rna for rna in rnas if rna not in random_samples]
with open(outputfile_remaining, 'w') as file:
    json.dump(remaining_samples, file, indent=4)

print('Bye!')
