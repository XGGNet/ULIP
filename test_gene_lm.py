# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("ctheodoris/Geneformer")
# model = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer")


import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

# model_dna = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True).cuda()

# model_former = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer")

import gpn.model
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

model_gpn = AutoModel.from_pretrained("songlab/gpn-brassicales")
# model_gpn.eval()

# dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
# inputs = tokenizer(dna, return_tensors = 'pt')["input_ids"] #[1,17]
# hidden_states = model_dna(inputs)[0] # [1,sequence_length] -> [1, sequence_length, 768]

# input = torch.randn(1,17,768).cuda()
# x = model_dna.encoder(input,attention_mask=torch.ones(1,17,device=input.device),output_all_encoded_layers=False,subset_mask=None)[0]

input = torch.randn(1,17,512).cuda()
x = model_gpn.encoder(input)


# embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape) # expect to be 768

# embedding with max pooling
embedding_max = torch.max(hidden_states[0], dim=0)[0]
print(embedding_max.shape) # expect to be 768

print('')