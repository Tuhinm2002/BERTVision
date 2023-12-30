import streamlit as st
from PIL import Image
import PIL
import easyocr

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification,BertTokenizer

model = BertForSequenceClassification.from_pretrained("./cpu model/")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
d = {0:"sadness",1:"joy",2:"love",3:"anger",4:"fear",5:"surprise"}

st.write("### BERT Vision 👁️👁️")
read = easyocr.Reader(['en'],gpu=False)
img = st.camera_input("Hint !! Take the image of a text")
if img is not None:
    img = Image.open(img)
    result = read.readtext(img)
    for bbox,text,_ in result:
        st.write("#### Obtained text from the image")
        st.write(text)
        word = tokenizer([text], truncation=True, max_length=128, padding="max_length",
                         return_tensors="pt")

        preds = model(word['input_ids'])
        st.write("### Word prediction")
        st.write(d[torch.argmax(preds['logits']).item()])

# cv.destroyAllWindows()
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
st.write("\n")
btn1 = st.button("FAQ ")
if btn1:
    st.write("What is FAQ ?!")

btn2 = st.button("Who made this ?")
if btn2:
    st.write("Guess who !!")

btn3 = st.button("Anything else you want to know ?? ")
if btn3:
    st.write("It may occasionally predict wrong outputs")

