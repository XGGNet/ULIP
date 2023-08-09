# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("ctheodoris/Geneformer")
model = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer")