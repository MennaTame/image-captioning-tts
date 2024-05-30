import os
import keras
import base64
import streamlit as st
import numpy as np
from gtts import gTTS
from tqdm import tqdm
from PIL import ImageOps
from IPython.display import Audio
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

model16 = keras.models.load_model('vgg16.h5')
model = keras.models.load_model('complete_model.h5')
tokenizer = Tokenizer()
max_length = 35

with open(os.path.join("C:/Users/roaam/OneDrive/Desktop/img captioning/", 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()

# create mapping of image to captions
mapping = {}
# process lines
for line in tqdm(captions_doc.split('\n')):
    # split the line by comma(,)
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension from image ID
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)

def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

clean(mapping)

all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

def idx_to_word(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
        in_text = 'startseq'

    # iterate over the max length of sequence
        for i in range(max_length):
        # encode input sequence
            sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
            sequence = pad_sequences([sequence], max_length)
        # predict next word
            yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
            yhat = np.argmax(yhat)
        # convert index to word
            word = idx_to_word(yhat, tokenizer)
        # stop if word not found
            if word is None:
                break
        # append word as input for generating next word
            in_text += " " + word
        # stop if we reach end tag
            if word == 'endseq':
                break
      
        return in_text
    
def img_caption(img):
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size)
    image = img_to_array(image)
    image = preprocess_input(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    feature = model16.predict(image, verbose=0)
    caption = predict_caption(model, feature, tokenizer, max_length)
    
    # Remove startseq and endseq tokens from the caption
    caption = caption.split()
    if 'startseq' in caption:
        caption.remove('startseq')
    if 'endseq' in caption:
        caption.remove('endseq')
    caption = ' '.join(caption)
    
    text = caption

# Language in which you want to convert
    language = 'en'

# Passing the text and language to the engine
    tts = gTTS(text=text, lang=language, slow=False)

# Save the speech to a file
    tts.save("output.mp3")

# Play the audio file
    Audio("output.mp3", autoplay=True)
    return caption


def autoplay_audio(file_path: str):

    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )





    