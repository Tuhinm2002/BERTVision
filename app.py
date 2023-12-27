import streamlit as st
import cv2 as cv
import easyocr

import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class BERTModel(nn.Module):
    def __init__(self,model,num_classes):
        super(BERTModel,self).__init__()
        self.model = model
        self.final_layer = nn.Linear(in_features=768,out_features=num_classes)

    def forward(self,x):
        x = self.model(x)[1]
        x = self.final_layer(x)

        return x

model_0 = BERTModel(model,num_classes=6)
model_0.load_state_dict(torch.load("model_cpu.pt"))

st.write("### NLP Vision ")
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

                with torch.no_grad():
                    model_0.eval()
                    Y_pred = model_0(word['input_ids'])
                    print(Y_pred)

        if cv.waitKey(1) & 0xFF == ord("q") or stop_button:
            break

    cap.release()

cv.destroyAllWindows()


