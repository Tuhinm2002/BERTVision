import torch
import torch.nn as nn
from transformers import BertForSequenceClassification,BertTokenizer

model = BertForSequenceClassification.from_pretrained("./cpu model/")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


word = tokenizer(["im feeling rather rotten so im not very ambitious right now"],truncation=True,max_length=128,padding="max_length",
                 return_tensors="pt")

output = model(word['input_ids'])
print(torch.argmax(output['logits']))
