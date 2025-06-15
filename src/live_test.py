import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model('face_mask_detection.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError("Unable to load the face cascade classifier xml file")

print("Model and face cascade classifier loaded successfully!")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Preprocess face for prediction
        face_roi = np.expand_dims(face_roi, axis=-1)  # Add channel dimension
        resized_face = tf.image.resize(face_roi, (256, 256))
        
        # Make prediction
        pred = model.predict(np.expand_dims(resized_face / 255.0, 0), verbose=0)
        
        predicted_label = 'With Mask' if pred[0][0] >= 0.5 else 'Without Mask'
        confidence = pred[0][0] # if pred[0][0] >= 0.5 else 1 - pred[0][0]
        
        # Choose color based on prediction
        color = (0, 255, 0) if predicted_label == 'With Mask' else (0, 0, 255)  # Green for mask, Red for no mask
        
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Add prediction text
        label_text = f"{predicted_label}: {confidence:.2f}"
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Add instructions
    cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Face Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
