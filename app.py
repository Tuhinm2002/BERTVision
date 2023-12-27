import streamlit as st
import cv2 as cv
import easyocr

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification,BertTokenizer

model = BertForSequenceClassification.from_pretrained("./cpu model/")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
d = {0:"sadness",1:"joy",2:"love",3:"anger",4:"fear",5:"surprise"}

st.write("### BERT Vision 👁️👁️")
read = easyocr.Reader(['en'],gpu=False)
stop_button = st.button("Stop")
start_button = st.button("Start")
if start_button:
    cap = cv.VideoCapture(0)
    frame_placeholder = st.empty()
    while cap.isOpened() and not stop_button:
        ret, frame = cap.read()

        if not ret:
            st.write("The video captures is ended")
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")
        result = read.readtext(frame)
        if len(result) > 1:
            for (bbox,text,prob) in result:
                word = tokenizer([text], truncation=True, max_length=128, padding="max_length",
                                 return_tensors="pt")

                preds = model(word['input_ids'])
                st.write(d[torch.argmax(preds['logits']).item()])

        if cv.waitKey(1) & 0xFF == ord("q") or stop_button:
            break

    cap.release()

cv.destroyAllWindows()
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

