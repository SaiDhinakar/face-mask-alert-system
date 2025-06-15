import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.metrics import Precision, Recall, Accuracy
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import kagglehub
import shutil

# clear tensorflow warning for cpu
os.system('clear')

if not os.path.exists('1/data'):
    default_path = kagglehub.dataset_download("omkargurav/face-mask-dataset")
    target_folder = "./"
    shutil.move(default_path, target_folder)
    path = os.path.join(target_folder, '1', "data")
    print("Dataset installed at:", path)

else:
    target_folder = './'
    path = os.path.join(target_folder, '1', "data")
    print("Dataset path:", path)

# Check if dataset path exists
if not os.path.exists(path):
    raise FileNotFoundError(f"Dataset path {path} does not exist. Please ensure the dataset is downloaded correctly.")

# Load and normalize data
data = image_dataset_from_directory(path, color_mode='grayscale', image_size=(256, 256))
data = data.map(lambda x, y: (x/255.0, y))  # Normalize pixel values to [0,1]

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)+1
test_size = int(len(data)*.1)+1
print(len(data))
print(train_size, test_size, val_size)
print((train_size+test_size+val_size)==len(data))

train_data = data.take(train_size)
valid_data = data.skip(train_size).take(val_size)
test_data = data.skip(train_size + val_size).take(test_size)

# Optimize data pipeline for better performance
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
valid_data = valid_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_data = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)),
    MaxPooling2D(),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(),

    Flatten(),

    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    'adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

log_dir = './logs'
tensorflow_callback = TensorBoard(log_dir=log_dir) 
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

hist = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=5,
    callbacks=[tensorflow_callback, early_stopping_callback]
)
model.save('face_mask_detection.h5')



pre = Precision()
re = Recall()
acc = Accuracy()

for batch in test_data.as_numpy_iterator():
  x, y = batch
  pred = model.predict(x)
  pre.update_state(y, pred)
  re.update_state(y, pred)
  acc.update_state(y, pred)
  
print(f'Precision : {pre.result():.4f}\nRecall : {re.result():.4f}\nAccuracy : {acc.result():.4f}')

# Test with sample images
test_images = [
    (os.path.join(path, 'with_mask', 'with_mask_10.jpg'), 'with_mask'),
    (os.path.join(path, 'without_mask', 'without_mask_10.jpg'), 'without_mask')
]

for img_path, label in test_images:
    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        if image is not None:
            # Convert to grayscale if needed (cv2.imread loads as BGR by default)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.expand_dims(image, axis=-1)  # Add channel dimension
            
            resized_image = tf.image.resize(image, (256, 256))
            pred = model.predict(np.expand_dims(resized_image / 255.0, 0))
            
            print(f"Prediction for {label} image: {pred[0][0]:.4f} ({'with_mask' if pred[0][0] > 0.5 else 'without_mask'})")
        else:
            print(f"Warning: Could not load image {img_path}")
    else:
        print(f"Warning: Image file {img_path} does not exist")



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot loss
ax1.plot(hist.history['loss'], color='blue', label='train loss')
ax1.plot(hist.history['val_loss'], color='orange', label='val loss')
ax1.set_title('Model Loss', fontsize=16)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()

# Plot accuracy
ax2.plot(hist.history['accuracy'], color='blue', label='train accuracy')
ax2.plot(hist.history['val_accuracy'], color='orange', label='val accuracy')
ax2.set_title('Model Accuracy', fontsize=16)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('model_training_results.png', dpi=300)
