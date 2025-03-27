# Machine-learning-final-project-
it is a face mask recognition using ML and DL 
!pip install kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d ashishjangra27/face-mask-12k-images-dataset
!unzip face-mask-12k-images-dataset.zip

# Step 3: Import Libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt

# Step 4: Define Paths and Parameters
train_dir = '/content/Face Mask Dataset/Train'
validation_dir = '/content/Face Mask Dataset/Validation'
test_dir = '/content/Face Mask Dataset/Test'

img_size = (150, 150)
batch_size = 32

# Step 5: Data Preparation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Step 6: Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Step 7: Train the Model
callbacks = [
    ModelCheckpoint('best_model.h5', save_best_only=True),
    EarlyStopping(patience=3, restore_best_weights=True)
]

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=callbacks
)

# Step 8: Evaluate Performance
# Plot training history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Train Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.show()

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_generator)
print(f'\nTest Accuracy: {test_acc*100:.2f}%')

# Step 9: Save the Model
model.save('face_mask_detector.h5')

# Step 10: Real-Time Face Mask Detection
# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect face and predict mask
def detect_mask(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (150, 150))
        face_roi = face_roi / 255.0
        face_roi = np.expand_dims(face_roi, axis=0)

        # Predict mask
        prediction = model.predict(face_roi)
        label = "Mask" if prediction[0][0] < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame



import cv2
sample_mask_img = cv2.imread('/content/WIN_20250312_14_07_59_Pro.jpg')
sample_mask_img = cv2.imread('/content/WIN_20250312_14_42_01_Pro.jpg')
sample_mask_img = cv2.resize(sample_mask_img,(128,128))
plt.imshow(sample_mask_img)
sample_mask_img = np.reshape(sample_mask_img,[1,128,128,3])
sample_mask_img = sample_mask_img/255.0

import cv2
import matplotlib.pyplot as plt
#face_model = cv2.CascadeClassifier('../input/haarcascades/haarcascade_frontalface_default.xml')
face_model= cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
#img = cv2.imread('/content/WIN_20250312_14_07_59_Pro.jpg')
img = cv2.imread('/content/WIN_20250312_14_42_01_Pro.jpg')
img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

faces = face_model.detectMultiScale(img) #returns a list of (x,y,w,h) tuples

out_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image

#plotting
for (x,y,w,h) in faces:
    cv2.rectangle(out_img,(x,y),(x+w,y+h),(0,0,255),1)
plt.figure(figsize=(10,10))
plt.imshow(out_img)
mask_label = {0:'MASK',1:'NO MASK'}
new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
for i in range(len(faces)):
    (x,y,w,h) = faces[i]
    crop = new_img[y:y+h,x:x+w]
    # Resize the crop to the input shape expected by the model (150, 150)
    crop = cv2.resize(crop,(150,150))
    crop = np.reshape(crop,[1,150,150,3])/255.0
    mask_result = model.predict(crop)
    # Get the predicted class (0 or 1)
    predicted_class = mask_result.argmax()
    # Access the label using the predicted class
    label = mask_label[predicted_class]
    if(mask_result[0][0]>=0.5):
        cv2.putText(new_img,mask_label[1],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
        print(mask_label[1])
    else:
        cv2.putText(new_img,mask_label[0],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
        print(mask_label[0])
    cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,255,0),2)
plt.figure(figsize=(10,10))
plt.imshow(new_img)
print(mask_result)


mask_label = {0:'MASK',1:'NO MASK'}
new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #colored output image
for i in range(len(faces)):
    (x,y,w,h) = faces[i]
    crop = new_img[y:y+h,x:x+w]
    # Resize the crop to the input shape expected by the model (150, 150)
    crop = cv2.resize(crop,(150,150))  # Resized to (150, 150)
    crop = np.reshape(crop,[1,150,150,3])/255.0 # Reshaped to (1, 150, 150, 3)
    mask_result = model.predict(crop)
    cv2.putText(new_img,mask_label[mask_result.argmax()],(x, y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0))
    print(mask_label[mask_result.argmax()])
    cv2.rectangle(new_img,(x,y),(x+w,y+h),(0,255,0),2)
plt.figure(figsize=(10,10))
plt.imshow(new_img)
print(mask_result)


#for live feed
import tkinter as tk
import cv2
import numpy as np

results={0:'without mask',1:'mask'}
GR_dict={0:(0,0,255),1:(0,255,0)}

rect_size = 4
# Try different camera indices if necessary (e.g., 1, 2, etc.)
cap = cv2.VideoCapture(0)
print(cap)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

haarcascade = cv2.CascadeClassifier('/home/user_name/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

while True:
    (rval, im) = cap.read()

    # Check if frame is read correctly
    if not rval:
        print("Error: Could not read frame.")
        break

    im=cv2.flip(im,1,1)

    rerect_size = cv2.resize(im,(128,128))
    faces = haarcascade.detectMultiScale(rerect_size)
    for f in faces:
        (x, y, w, h) = [v * rect_size for v in f]

        face_img = im[y:y+h, x:x+w]
        rerect_sized=cv2.resize(face_img,(128,128))
        normalized=rerect_sized/255.0
        reshaped=np.reshape(normalized,(1,128,128,3))
        reshaped = np.vstack([reshaped])
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]

        cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
        cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
        cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(10)

    if key == 27:
        break

cap.release()

cv2.destroyAllWindows()
