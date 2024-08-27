#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the dataset
train_df = pd.read_csv('sign_mnist_train.csv')
test_df = pd.read_csv('sign_mnist_test.csv')

# Separate labels and features
X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values
y_test = test_df.iloc[:, 0].values

# Normalize the images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape the images to 28x28
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=25)
y_test = to_categorical(y_test, num_classes=25)


# In[2]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(25, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# In[3]:


model.save('sign_language_model.h5')


# In[ ]:


import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import datetime
from tensorflow.keras.models import load_model

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection")
        
        self.model = load_model('sign_language_model.h5')
        
        self.label = tk.Label(root, text="Sign Language Detection", font=("Arial", 20))
        self.label.pack()

        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()
        
        self.video_button = tk.Button(root, text="Real-time Video", command=self.start_video)
        self.video_button.pack()
        
        self.prediction_label = tk.Label(root, text="", font=("Arial", 20))
        self.prediction_label.pack()
        
        self.video_label = tk.Label(root)
        self.video_label.pack()
        
        self.cap = None
        self.video_running = False

    def check_time(self):
        current_time = datetime.datetime.now().time()
        start_time = datetime.time(18, 0, 0)
        end_time = datetime.time(22, 0, 0)
        return start_time <= current_time <= end_time

    def preprocess_image(self, img):
        img = cv2.resize(img, (28, 28))
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)
        return img

    def upload_image(self):
        if not self.check_time():
            self.prediction_label.config(text="Model operational between 6 PM and 10 PM")
            return
        
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = self.preprocess_image(image)
            prediction = self.model.predict(image)
            predicted_label = np.argmax(prediction)
            self.prediction_label.config(text=f"Prediction: {chr(predicted_label + 65)}")

    def start_video(self):
        if not self.check_time():
            self.prediction_label.config(text="Model operational between 6 PM and 10 PM")
            return

        self.cap = cv2.VideoCapture(0)
        self.video_running = True
        self.show_frame()

    def show_frame(self):
        if not self.video_running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = self.preprocess_image(gray)
            prediction = self.model.predict(img)
            predicted_label = np.argmax(prediction)
            self.prediction_label.config(text=f"Prediction: {chr(predicted_label + 65)}")

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            imgtk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.root.after(10, self.show_frame)

    def stop_video(self):
        self.video_running = False
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop_video)
    root.mainloop()


# In[ ]:




