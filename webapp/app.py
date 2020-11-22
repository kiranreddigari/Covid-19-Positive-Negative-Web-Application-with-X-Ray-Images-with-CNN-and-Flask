from flask import Flask, render_template, request,jsonify
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D,Activation,MaxPooling2D
from keras.utils import normalize
from keras.layers import Concatenate
from keras import Input
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import keras.backend as K

img_size=100

app = Flask(__name__) 

def load_model():

	K.clear_session()
	input_shape=(100,100,1)
	inp=Input(shape=input_shape)
	convs=[]

	parrallel_kernels=[3,5,7]

	for k in range(len(parrallel_kernels)):

	    conv = Conv2D(128, parrallel_kernels[k],padding='same',activation='relu',input_shape=input_shape,strides=1)(inp)
	    convs.append(conv)

	out = Concatenate()(convs)
	conv_model = Model(inputs=inp, outputs=out)

	model = Sequential()
	model.add(conv_model)

	model.add(Conv2D(64,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Conv2D(32,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))

	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2,input_dim=128,activation='softmax'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.load_weights('model/model-016.h5')
	return model

label_dict={0:'Covid19 Negative', 1:'Covid19 Positive'}

def preprocess(img):

	img=np.array(img)

	if(img.ndim==3):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	reshaped=resized.reshape(1,img_size,img_size,1)
	return reshaped

@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image=preprocess(image)
	model=load_model()
	prediction =model.predict(test_image)
	result=np.argmax(prediction,axis=1)[0]
	accuracy=float(np.max(prediction,axis=1)[0])

	label=label_dict[result]

	print(prediction,result,accuracy)

	response = {'prediction': {'result': label,'accuracy': accuracy}}

	return jsonify(response)

app.run(debug=True)

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">