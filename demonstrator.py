# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:55:46 2020

@author: LETONG WEI
"""
import subprocess
import sys
# from sample2 import predict
import tkinter.filedialog
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([256,256], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def predict(path):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(300).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(300, 512, vocab, 1)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load('models/encoder-5-3000.ckpt'))
    decoder.load_state_dict(torch.load('models/decoder-5-3000.ckpt'))

    # Prepare an image
    image = load_image(path, transform)
    image_tensor = image.to(device) # 1,3,w,h
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption[1:-1])
    print(sentence)
    return sentence

def resize(w, h, w_box, h_box, pil_image):  

  f1 = 1.0*w_box/w
  f2 = 1.0*h_box/h  
  factor = min([f1, f2])  
  width = int(w*factor)  
  height = int(h*factor)  
  return pil_image.resize((width, height), Image.ANTIALIAS)  


window = tk.Tk()
window.title("Deep Learning Project")
window.config(bg = "skyblue")
window.geometry("900x620")
window.resizable(0,0)

# create frame widget
left_frame = Frame(window, width = 200, height = 600, bg = "grey")
left_frame.grid(row = 0, column = 0, padx = 10, pady = 5)
right_frame = Frame(window, width = 650, height = 600, bg = "white")
right_frame.grid(row = 0, column = 1, padx = 10, pady = 5)

command_frame = Frame(left_frame, width=200, height=155, bg="grey")
command_frame.grid(row=0, column=0, padx=5, pady=5)
text_frame = Frame(left_frame, width=200, height=425, bg="white")
text_frame.grid(row=1, column=0, padx=5, pady=5)
command_frame.grid_propagate(0)
text_frame.grid_propagate(0)
right_frame.grid_propagate(0)
captions = tk.Label(text_frame, height=5, width=25, text="Image captions:", wraplength=150, bg="white", anchor="n")
captions.grid(row=0, column=0, padx=10, pady=5, sticky='W')

def open_file():
    entry_filename.delete(0, END)
    filename = tk.filedialog.askopenfilename(title='upload image file', filetypes=[('png', '*.png'),('jpeg', '*.jpeg'), ('jpg', '*.jpg')])
    entry_filename.insert('insert', filename)
    print(filename)
    img = Image.open(filename)
    w, h = img.size
    img = resize(w, h, 600, 550, img)
    tkimg = ImageTk.PhotoImage(img)
    display = Label(right_frame, image=tkimg, width=650, height=600, bg="white")
    display.image = tkimg
    display.grid(row = 0, column=0, padx=5, pady=5)

entry_filename = tk.Entry(command_frame, width=50)
entry_filename.grid(row=0, column=0, padx=10, pady=5)

button_import = tk.Button(command_frame, text="Upload Image", command=open_file, width=25)
button_import.grid(row=1, column=0, padx=10, pady=5, sticky='W')



def print_file():
    name = entry_filename.get()  
    print(name)    
    content = predict(name)

    text = tk.Label(text_frame, height=22, width=25, text=content, wraplength=150, bg="white", anchor="n")
    text.grid(row=1, column=0, padx=10, pady=5, sticky='W')


print_button = tk.Button(command_frame, text="Predict", command=print_file, width=25)
print_button.grid(row=2, column=0, padx=10, pady=5, sticky='W')
window.mainloop()
