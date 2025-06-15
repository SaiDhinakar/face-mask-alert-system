import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# Load the trained model
model_path = 'model/face_mask_detection.h5'
model_path = 'face_mask_detection.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Please run train.py first to create the model.")

print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError("Unable to load the face cascade classifier xml file")

print("Face cascade classifier loaded successfully!")

# Get test images with proper paths and labels
test_dir = 'test'

if not os.path.exists(test_dir):
    raise FileNotFoundError(f"Test directory '{test_dir}' not found.")

test_images = []
path = '1/data'
for test_dir in os.listdir(path):
    for filename in os.listdir(os.path.join(path,test_dir))[:500]:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(path,test_dir, filename)
            # Extract label from filename (assumes filenames like 'with_mask_10.jpg' or 'without_mask_19.jpg')
            if 'with_mask' in filename.lower():
                label = 'with_mask'
            elif 'without_mask' in filename.lower():
                label = 'without_mask'
            else:
                label = 'unknown'
            test_images.append((img_path, label))

print(f"Found {len(test_images)} test images")

correct_predictions = 0
total_predictions = 0

for img_path, label in test_images:
    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        if image is not None:
            # Detect faces using Haar Cascade
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if True or len(faces) > 0:
                # Process the first detected face
                # (x, y, w, h) = faces[0]
                # face_roi = gray[y:y+h, x:x+w]
                
                # Resize face ROI and add channel dimension
                # face_roi = np.expand_dims(face_roi, axis=-1)
                face_roi = np.expand_dims(gray, axis=-1)
                resized_face = tf.image.resize(face_roi, (256, 256))
                
                # Make prediction
                pred = model.predict(np.expand_dims(resized_face / 255.0, 0), verbose=0)
                
                predicted_label = 'with_mask' if pred[0][0] <= 0.5 else 'without_mask'
                confidence = pred[0][0] if pred[0][0] >= 0.5 else 1 - pred[0][0]
                
                # Check if prediction is correct
                is_correct = predicted_label == label
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                status = "✓" if is_correct else "✗"
                print(f"{status} {os.path.basename(img_path)}: True={label}, Predicted={predicted_label}, Confidence={confidence:.4f} [Face detected: {len(faces)} faces]")
            # else:
            #     print(f"⚠ {os.path.basename(img_path)}: No face detected - skipping prediction")
        else:
            print(f"Warning: Could not load image {img_path}")
    else:
        print(f"Warning: Image file {img_path} does not exist")

# Print summary statistics
if total_predictions > 0:
    accuracy = correct_predictions / total_predictions
    print(f"\n--- Test Results Summary ---")
    print(f"Total images tested: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {total_predictions - correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
else:
    print("\nNo faces were detected in any test images.")

# Optional: Create a visual output showing detected faces
# print(f"\nProcessing images for visual output...")
# output_dir = 'test_results'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

for img_path, label in test_images:  # Process first 5 images for visualization
    if os.path.exists(img_path):
        image = cv2.imread(img_path)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Draw rectangles around faces and add predictions
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = np.expand_dims(face_roi, axis=-1)
                resized_face = tf.image.resize(face_roi, (256, 256))
                pred = model.predict(np.expand_dims(resized_face / 255.0, 0), verbose=0)
                
                predicted_label = 'With Mask' if pred[0][0] <= 0.5 else 'Without Mask'
                confidence = pred[0][0] if pred[0][0] >= 0.5 else 1 - pred[0][0]
                
                # Color based on prediction
                color = (0, 255, 0) if predicted_label == 'With Mask' else (0, 0, 255)
                
                # Draw rectangle and text
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                cv2.putText(image, f"{predicted_label}: {confidence:.2f}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(image, f"True: {label}", (x, y+h+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            face_roi = np.expand_dims(gray, axis=-1)
            resized_face = tf.image.resize(face_roi, (256, 256))
            pred = model.predict(np.expand_dims(resized_face / 255.0, 0), verbose=0)
            
            predicted_label = 'With Mask' if pred[0][0] <= 0.5 else 'Without Mask'
            confidence = pred[0][0] if pred[0][0] >= 0.5 else 1 - pred[0][0]
            
            # Color based on prediction
            # color = (0, 255, 0) if predicted_label == 'With Mask' else (0, 0, 255)
            
            # Draw rectangle and text
            # cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            # cv2.putText(image, f"{predicted_label}: {confidence:.2f}", (0,0), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # cv2.putText(image, f"True: {label}", (0,0), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            # Save result
            # output_path = os.path.join(output_dir, f"result_{os.path.basename(img_path)}")
            # cv2.imwrite(output_path, image)

# print(f"Visual results saved in '{output_dir}' directory")