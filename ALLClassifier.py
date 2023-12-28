import os
import numpy as np
from PIL import ImageTk, Image
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import messagebox

print('Modules loaded')

##################################
#  ENSEMBLE MODEL INITIALIZATION #
##################################

# Model loading and parameter initialization

img_size = (224, 224)
channels = 3 # either BGR or Grayscale
color = 'rgb'
img_shape = (img_size[0], img_size[1], channels)
directory = os.getcwd()
directory = os.path.join(directory,'Models')

print('Initializing Models...')

# Model 1 (ConvNeXtTiny)
base_model = tf.keras.applications.ConvNeXtTiny(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
model1 = Sequential([
  base_model,
  BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
  Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
              bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
  Dropout(rate= 0.45, seed= 123),
  Dense(2, activation= 'softmax')
])
model1.compile(Adamax(learning_rate= 0.0012), loss= 'categorical_crossentropy', metrics= ['accuracy'])
model1.load_weights(os.path.join(directory,'ALL-final-ConvNeXtTiny-weights.h5'))
model1.summary()

# Model 2 (MobileNet)
base_model = tf.keras.applications.MobileNetV2(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
model2 = Sequential([
  base_model,
  BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
  Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
              bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
  Dropout(rate= 0.45, seed= 123),
  Dense(2, activation= 'softmax')
])
model2.compile(Adamax(learning_rate= 0.0012), loss= 'categorical_crossentropy', metrics= ['accuracy'])
model2.load_weights(os.path.join(directory,'ALL-final-MobileNetV2-weights.h5'))
model2.summary()

# Model 3 (EfficientNetV2B3)
base_model = tf.keras.applications.EfficientNetV2B3(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
model3 = Sequential([
  base_model,
  BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
  Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
              bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
  Dropout(rate= 0.45, seed= 123),
  Dense(2, activation= 'softmax')
])
model3.compile(Adamax(learning_rate= 0.0012), loss= 'categorical_crossentropy', metrics= ['accuracy'])
model3.load_weights(os.path.join(directory,'ALL-final-EficientNetV2B3-weights.h5'))
model3.summary()

# Model 4 (InceptionV3)
base_model = tf.keras.applications.InceptionV3(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
model4 = Sequential([
  base_model,
  BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
  Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
              bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
  Dropout(rate= 0.45, seed= 123),
  Dense(2, activation= 'softmax')
])
model4.compile(Adamax(learning_rate= 0.0012), loss= 'categorical_crossentropy', metrics= ['accuracy'])
model4.load_weights(os.path.join(directory,'ALL-final-InceptionV3-weights.h5'))
model4.summary()

# Model 5 (DenseNet121)
base_model = tf.keras.applications.DenseNet121(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
model5 = Sequential([
  base_model,
  BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
  Dense(256, kernel_regularizer= regularizers.l2(l= 0.016), activity_regularizer= regularizers.l1(0.006),
              bias_regularizer= regularizers.l1(0.006), activation= 'relu'),
  Dropout(rate= 0.45, seed= 123),
  Dense(2, activation= 'softmax')
])
model5.compile(Adamax(learning_rate= 0.0012), loss= 'categorical_crossentropy', metrics= ['accuracy'])
model5.load_weights(os.path.join(directory,'ALL-final-DenseNet121-weights.h5'))
model5.summary()

# Meta Model
meta_model = Sequential(
        [
            Dense(units=64, activation='relu', input_shape=(5,)),
            Dropout(0.15),  # Dropout layer for regularization
            Dense(units=32, activation='relu'),
            Dropout(0.15),  # Dropout layer for regularization
            Dense(units=1, activation='sigmoid')  # Output layer with sigmoid activation
        ]
    )
meta_model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])
meta_model.load_weights(os.path.join(directory,'ALL-7030-meta-model-top-5-weights-jul24.h5'))
meta_model.summary()

#########################
#  USER INTERFACE SECTION
#########################

class ClassifierUI(tk.Toplevel):
	frame = None
	class_image = None
	instruction1 = None
	instruction2 = None

	def __init__(self):
		super().__init__()
		self.title("ALL Cell Classifier")
		self.resizable(False, False)
		self.configure(bg='#333333')
		

	# === BEGIN HEADER ===
		header_frame = tk.Frame(self, width=500, bg="#333333")

		# Header frame components 
		reset_btn = tk.Button(header_frame, text='Reset', width=10,
								padx=35, pady=6,
								fg="black", bg="#cccccc", command=self.reset)
		tit = tk.Label(header_frame, text="ALL Cell Classifier", width=30, 
								fg="#fcba03", bg="#333333", 
								pady=6, font=("bold", 15))
		logout_btn = tk.Button(header_frame, text='Logout', width=10, 
								padx=35, pady=6,
								fg="black", bg="#cccccc", command=self.logout)

		# Header frame layout grid
		reset_btn.grid(row=0, column=0, sticky='nesw')
		tit.grid(row=0, column=1, sticky='ew')
		logout_btn.grid(row=0, column=2, sticky='nesw')

		header_frame.pack()
	# === END HEADER ===


	# === BEGIN BODY ===
		canvas = tk.Canvas(self, height=500, width=500, bg="#333333", highlightthickness=0)
		canvas.pack()

		self.frame = tk.Frame(self, bg='white', highlightthickness=0)
		self.frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

		self.instruction1 = tk.Label(self.frame, text="1. Click 'Choose image' and upload a blood cell image.",
						wraplength=320, bg="white", font=("Arial", 13)).place(relx=0.18, rely=0.7)
		self.instruction2 = tk.Label(self.frame, text="2. Click 'Classify image' to have the system analyze and classify the image.",
						wraplength=320, bg="white", font=("Arial", 13)).place(relx=0.18, rely=0.82)
	
		# Choose image button
		chose_image = tk.Button(self, text='Choose Image',
								padx=35, pady=10,
								fg="black", bg="#cccccc", command=self.load_img)
		chose_image.pack(side=tk.LEFT)

		# Classify image button
		self.class_image = tk.Button(self, text='Classify Image',
								padx=35, pady=10,
								fg="black", bg="#cccccc", command=self.classify)
		self.class_image.pack(side=tk.RIGHT)
		self.class_image["state"] = "disabled"
	# === END BODY ===

	# === RESET AND LOGOUT ===
	def reset(self):
		self.class_image["state"] = "disabled"
		for img_display in self.frame.winfo_children():
			img_display.destroy()
		self.instruction1 = tk.Label(self.frame, text="1. Click 'Choose image' and upload a blood cell image.",
						wraplength=320, bg="white", font=("Arial", 13)).place(relx=0.18, rely=0.7)
		self.instruction2 = tk.Label(self.frame, text="2. Click 'Classify image' to have the system analyze and classify the image.",
						wraplength=320, bg="white", font=("Arial", 13)).place(relx=0.18, rely=0.82)

	def logout(self):
		classifier_UI.withdraw()
		login_UI.deiconify()
	# === END RESET AND LOGOUT ===

	# === LOAD IMAGE ===
	def load_img(self):
		print('Choose button clicked')
		
		global img, image_data
		for img_display in self.frame.winfo_children():
			img_display.destroy()
		self.instruction1 = tk.Label(self.frame, text="1. Click 'Choose image' and upload a blood cell image.",
						wraplength=320, bg="white", font=("Arial", 13)).place(relx=0.18, rely=0.7)
		self.instruction2 = tk.Label(self.frame, text="2. Click 'Classify image' to have the system analyze and classify the image.",
						wraplength=320, bg="white", font=("Arial", 13)).place(relx=0.18, rely=0.82)

		image_data = filedialog.askopenfilename(initialdir="/", title="Choose an image",
										filetypes=[("BMP Files","*.bmp")])
		basewidth = 150 # Processing image for displaying
		img = Image.open(image_data)
		wpercent = (basewidth / float(img.size[0]))
		hsize = int((float(img.size[1]) * float(wpercent)))
		img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
		img = ImageTk.PhotoImage(img)
		
		# Display image

		# Console
		print()
		print("File: " + image_data)
		print()
	
		# UI
		file_name = image_data.split('/')
		panel = tk.Label(self.frame, text= str(file_name[len(file_name)-1]).upper()).pack()
		panel_image = tk.Label(self.frame, image=img).pack()
		self.class_image["state"] = "normal"
	# === END LOAD IMAGE ===

	# === CLASSIFY IMAGE ===
	def classify(self):
		print('Classify button clicked')
		print('Preprocessing image...')

		# Preprocess image
		img = Image.open(image_data)
		img = img.resize((224, 224), Image.Resampling.LANCZOS)
		img_array = tf.keras.preprocessing.image.img_to_array(img)
		img_expanded = np.expand_dims(img_array, axis=0)
		
		# Make base model predictions
		x = 0
		predictions = []
		print('Generating predictions...')
		print()
		for model in [model1, model2, model3, model4, model5]:
			pred = model.predict(img_expanded)
			x += 1
			print(f'Model {x} => ALL - {round(pred[0][0]*100,2)}%, HEM - {round(pred[0][1]*100,2)}%')
			predictions.append(pred[0][0])

		# Make ensemble prediction
		print()
		predictions1 = np.vstack(predictions)
		predictions1 = predictions1.T
		pred = meta_model.predict(np.array(predictions1))
		print(f'Stacking Ensemble => ALL - {round((1-pred[0][0])*100,2)}%, HEM - {round(pred[0][0]*100,2)}%')
		pred_value = (pred[0][0] > 0.5).astype(int) 
		if (pred_value == 0) :
			pred[0][0] = 1 - pred[0][0]

		pred_cells = ['ALL','HEM']
		result_text = pred_cells[pred_value] + ' (' + str(round(float(pred[0][0])*100, 2)) + '%)'

		# Display Results

		# Console
		print()
		print("Predicted cell classification")
		print(result_text)
		print()

		# UI
		emptyline1 = tk.Label(self.frame, text="", bg="white").pack()
		table  = tk.Label(self.frame, text="Predicted cell classification", bg="white", font=("bold", 14)).pack()
		emptyline2 = tk.Label(self.frame, text="", bg="white").pack()
		if (pred_value == 1) :
			result = tk.Label(self.frame, text=result_text, fg="red", bg="white", font=("bold", 14)).pack()
		else :
			result = tk.Label(self.frame, text=result_text, fg="green", bg="white", font=("bold", 14)).pack()
	# === END CLASSIFY IMAGE ===

class LoginUI(tk.Tk):
	username_entry = None
	password_entry = None
	
	def __init__(self):
		super().__init__()
		self.title("ALL Image Classifier")
		self.geometry('400x400')
		self.configure(bg='#333333')

		# Login frame
		frame = tk.Frame(bg='#333333')

		# Login frame elements
		login_label = tk.Label(
			frame, text="Login", bg='#333333', fg="#fcba03", font=("Arial", 30))
		username_label = tk.Label(
			frame, text="Username", bg='#333333', fg="#FFFFFF", font=("Arial", 16))
		self.username_entry = tk.Entry(frame, font=("Arial", 16))
		self.password_entry = tk.Entry(frame, show="*", font=("Arial", 16))
		password_label = tk.Label(
			frame, text="Password", bg='#333333', fg="#FFFFFF", font=("Arial", 16))
		login_button = tk.Button(
			frame, text="Login", bg="#fcba03", fg="#FFFFFF", font=("Arial", 16), command=self.login)

		# Login frame layout grid
		login_label.grid(row=0, column=0, columnspan=2, sticky="news", pady=40)
		username_label.grid(row=1, column=0)
		self.username_entry.grid(row=1, column=1, pady=20)
		password_label.grid(row=2, column=0)
		self.password_entry.grid(row=2, column=1, pady=20)
		login_button.grid(row=3, column=0, columnspan=2, pady=30)

		frame.pack()
	
	def login(self):
		username = "admin"
		password = "admin"
		if self.username_entry.get()==username and self.password_entry.get()==password:
			open_classifier_UI()
			self.reset_entries()
		else:
			messagebox.showerror(title="Error", message="Invalid username or password.")
	
	def reset_entries(self):
		self.username_entry.delete(0,tk.END)
		self.password_entry.delete(0,tk.END)
		

def open_classifier_UI():
	login_UI.withdraw()
	global isExtraWindowOpen
	global classifier_UI
	if isExtraWindowOpen == False:
		isExtraWindowOpen = True
		classifier_UI = ClassifierUI()
	else:
		classifier_UI.deiconify()


# run 
isExtraWindowOpen = False
classifier_UI = None
login_UI = LoginUI()
login_UI.mainloop()







