#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install tensorflow')


# In[28]:


# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
from warnings import filterwarnings
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        os.path.join(dirname, filename)


# # # Data Preperation

# In[29]:


# create our labels
labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']


# In[30]:


# # # append all images to lists
# # X_train = []
# # y_train = []
# # image_size = 300
# # for i in labels:
# #     folderPath = os.path.join('C:\Users\indra\OneDrive\Desktop','Training',i)
# #     for j in tqdm(os.listdir(folderPath)):
# #         img = cv2.imread(os.path.join(folderPath,j))
# #         img = cv2.resize(img,(image_size, image_size))
# #         X_train.append(img)
# #         y_train.append(i)
        
# # for i in labels:
# #     folderPath = os.path.join('C:\Users\indra\OneDrive\Desktop','Testing',i)
# #     for j in tqdm(os.listdir(folderPath)):
# #         img = cv2.imread(os.path.join(folderPath,j))
# #         img = cv2.resize(img,(image_size,image_size))
# #         X_train.append(img)
# #         y_train.append(i)
# import os
# import cv2
# from tqdm import tqdm

# # Initialize lists
# X_train = []
# y_train = []
# image_size = 300

# # Define labels (assuming you have a list of labels)
# labels = ['label1', 'label2', 'label3']  # Replace with your actual labels

# # Append training images to lists
# for i in labels:
#     folderPath = os.path.join('C:\\Users\\indra\\OneDrive\\Desktop\\Brain-Tumor-Classification-DataSet-master', 'Training', i)
#     for j in tqdm(os.listdir(folderPath)):
#         img = cv2.imread(os.path.join(folderPath, j))
#         img = cv2.resize(img, (image_size, image_size))
#         X_train.append(img)
#         y_train.append(i)

# # Append testing images to lists
# for i in labels:
#     folderPath = os.path.join('C:\\Users\\indra\\OneDrive\\Desktop\\Brain-Tumor-Classification-DataSet-master', 'Testing', i)
#     for j in tqdm(os.listdir(folderPath)):
#         img = cv2.imread(os.path.join(folderPath, j))
#         img = cv2.resize(img, (image_size, image_size))
#         X_train.append(img)
#         y_train.append(i)
import os
import cv2
from tqdm import tqdm

# Initialize lists
X_train = []
y_train = []
image_size = 300

# Define labels (assuming you have a list of labels)
labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']  # Replace with your actual labels

# Append training images to lists
for i in labels:
    folderPath = os.path.join('C:\\Users\\indra\\OneDrive\\Desktop\\Brain-Tumor-Classification-DataSet-master', 'Training', i)
    print(f"Checking path: {folderPath}")
    if not os.path.exists(folderPath):
        print(f"Path does not exist: {folderPath}")
        continue
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)

# Append testing images to lists
for i in labels:
    folderPath = os.path.join('C:\\Users\\indra\\OneDrive\\Desktop\\Brain-Tumor-Classification-DataSet-master', 'Testing', i)
    print(f"Checking path: {folderPath}")
    if not os.path.exists(folderPath):
        print(f"Path does not exist: {folderPath}")
        continue
    for j in tqdm(os.listdir(folderPath)):
        img = cv2.imread(os.path.join(folderPath, j))
        img = cv2.resize(img, (image_size, image_size))
        X_train.append(img)
        y_train.append(i)


# In[33]:


# convert to arrays
X_train = np.array(X_train)
y_train = np.array(y_train)


# In[34]:


# create figure to look at sample image
k=0
fig, ax = plt.subplots(1,4,figsize=(20,20))
fig.text(s='Example Image From Each Label',
         size=18,
         fontweight='bold',
         fontname='monospace',
         color='black',
         y=0.62,
         x=0.4,
         alpha=0.8)

# upload random image
import random
for i in labels:
    j=random.randint(0, len(X_train)-1)
    while True :
        if y_train[j]==i:
            ax[k].imshow(X_train[j])
            ax[k].set_title(y_train[j])
            ax[k].axis('off')
            k+=1
            break
        j+=1


# In[39]:


# shuffle data
X_train, y_train = shuffle(X_train,y_train, random_state=101)


# In[45]:


# look at shape
X_train.shape


# In[50]:


# create the training and testing split from our data
X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.2,random_state=101)


# In[60]:


# # convert labels to categorical numbers
# y_train_new = []
# for i in y_train:
#     y_train_new.append(labels.index(i))
# y_train = y_train_new
# y_train = tf.keras.utils.to_categorical(y_train)


# y_test_new = []
# for i in y_test:
#     y_test_new.append(labels.index(i))
# y_test = y_test_new
# y_test = tf.keras.utils.to_categorical(y_test)
# import tensorflow as tf

# # Convert labels to categorical numbers
# def convert_labels_to_categorical(y, labels):
#     y_new = []
#     for i in y:
#         if i in labels:
#             y_new.append(labels.index(i))
#         else:
#             raise ValueError(f"Label {i} is not in the list of labels.")
#     return tf.keras.utils.to_categorical(y_new)

# # Assuming labels is defined
# labels = ['glioma_tumor','no_tumor','meningioma_tumor','pituitary_tumor']  # Replace with your actual labels

# # Convert y_train
# try:
#     y_train = convert_labels_to_categorical(y_train, labels)
# except ValueError as e:
#     print(e)

# # Convert y_test
# try:
#     y_test = convert_labels_to_categorical(y_test, labels)
# except ValueError as e:
#     print(e)
import tensorflow as tf
import numpy as np

# Check if labels are already one-hot encoded
def is_one_hot_encoded(y):
    return isinstance(y[0], np.ndarray) and len(y[0]) == len(labels)

# Convert labels to categorical numbers if not already one-hot encoded
def convert_labels_to_categorical(y, labels):
    if is_one_hot_encoded(y):
        return np.array(y)
    y_new = []
    for i in y:
        if i in labels:
            y_new.append(labels.index(i))
        else:
            raise ValueError(f"Label {i} is not in the list of labels.")
    return tf.keras.utils.to_categorical(y_new)

# Assuming labels is defined
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']  # Replace with your actual labels

# Convert y_train
try:
    y_train = convert_labels_to_categorical(y_train, labels)
except ValueError as e:
    print(e)

# Convert y_test
try:
    y_test = convert_labels_to_categorical(y_test, labels)
except ValueError as e:
    print(e)


# # # Transfer Learning

# In[61]:


# import EfficientNetB0 weights
effnet = EfficientNetB0(weights='imagenet',
                        include_top=False,
                        input_shape=(image_size,image_size,3))


# In[62]:


# apply weights to our model
model = effnet.output

# global average for easier computation
model = tf.keras.layers.GlobalAveragePooling2D()(model)

# dropout to avoid overfitting
model = tf.keras.layers.Dropout(rate=0.5)(model)

# output layer
model = tf.keras.layers.Dense(4,activation='softmax')(model)
model = tf.keras.models.Model(inputs=effnet.input, outputs = model)


# In[63]:


# look at model summary
model.summary()


# In[64]:


# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer = 'Adam',
              metrics= ['accuracy'])


# In[65]:


n = 15


# In[ ]:


# train the model on our data
history = model.fit(X_train,y_train,validation_split=0.1, epochs = 5, verbose=1, batch_size=32)


# # # Model Evaluation

# In[ ]:


# look at chart of accuracy & loss
filterwarnings('ignore')

epochs = [i for i in range(n)]
fig, ax = plt.subplots(1,2,figsize=(14,7))
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']

fig.text(0.5, 0.9, 'Performance Over Epochs',
         ha='center',
         fontsize=20,
         fontweight='bold')

sns.despine()
ax[0].plot(epochs, train_acc, marker='o',color='blue',
           label = 'Training')
ax[0].plot(epochs, val_acc, marker='o',color='red',
           label = 'Validation')
ax[0].legend(frameon=False)
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')

sns.despine()
ax[1].plot(epochs, train_loss, marker='o',color='blue',
           label ='Training')
ax[1].plot(epochs, val_loss, marker='o',color='red',
           label = 'Validation')
ax[1].legend(frameon=False)
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')

fig.show()


# In[ ]:


plt.savefig('accuracy_loss.png')


# In[18]:


# create predicted values to compare vs actual ones
pred = model.predict(X_test)


# In[19]:


pred = np.argmax(pred,axis=1)
y_test_new = np.argmax(y_test,axis=1)


# In[20]:


# plot confusion matrix
fig,ax=plt.subplots(1,1,figsize=(14,7))
sns.heatmap(confusion_matrix(y_test_new,pred),
            ax=ax,xticklabels=labels,
            yticklabels=labels,
            annot=True,
            cmap='Blues',
            alpha=0.7,
            linewidths=2,
            linecolor='black',
            cbar=False)
fig.text(s='Heatmap of the Confusion Matrix',size=18,fontweight='bold',
             fontname='monospace',color='black',y=0.92,x=0.28,alpha=0.8)

plt.show()


# In[21]:


plt.savefig('confusion_matrix.png')


# In[22]:


print(classification_report(y_test_new,pred, target_names=labels))


# In[23]:


random_index = np.random.randint(0, len(X_test))
random_img = X_test[random_index]  
predictions = model.predict(random_img.reshape(1, image_size, image_size, 3))  # Reshape and preprocess the image

# Interpret the model's predictions
predicted_class = np.argmax(predictions)  # Get the index of the class with the highest probability
predicted_label = labels[predicted_class]  # Convert class to label
confidence = predictions[0][predicted_class]

actual_index = y_test[random_index]  # Get the one-hot encoded actual class
actual_class = np.argmax(actual_index)  
actual_label = labels[actual_class] 

# Display the image and prediction information
print(f"\033[94mPredicted label: {predicted_label}\033[0m \n\033[92mActual label: {actual_label}\033[0m \n\033[93mConfidence: {confidence*100:.2f}%\033[0m\n")
plt.figure(figsize = (3,3))
plt.imshow(random_img)
plt.axis('off')  
plt.show()


# # # Download Model

# In[ ]:


model.save('effNetB0_300.keras')


# In[ ]:




