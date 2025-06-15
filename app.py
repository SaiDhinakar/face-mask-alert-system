from flask import Flask, render_template, request, Response, jsonify
import cv2
import os
import numpy as np
import tensorflow as tf
import base64
import io
from PIL import Image

app = Flask(__name__, template_folder = 'templates')

model_path = 'model/face_mask_detection.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Please run train.py first to create the model.")

print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise IOError("Unable to load the face cascade classifier xml file")

print("Face cascade classifier loaded successfully!")


@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/image-test', methods=['POST'])
def image_test():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the image file into a numpy array
        img_array = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Error occurred in image processing. Please try a different image.'}), 400
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return jsonify({'error': 'No faces detected in the image'}), 400
        
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        face_roi = np.expand_dims(face_roi, axis=-1)
        resized_face = tf.image.resize(face_roi, (256, 256))
        
        pred = model.predict(np.expand_dims(resized_face / 255.0, 0), verbose=0)
        
        # Convert numpy float32 to regular Python float
        pred_value = float(pred[0][0])
        
        predicted_label = 'With Mask' if pred_value <= 0.5 else 'Without Mask'
        
        confidence = pred_value if pred_value >= 0.5 else 1 - pred_value
        
        color = (0, 255, 0) if predicted_label == 'With Mask' else (0, 0, 255)
        
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 3)
        cv2.putText(image, f'{predicted_label}: {confidence*100:.1f}%', 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Convert image to base64 for display
        _, img_encoded = cv2.imencode('.jpg', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        return jsonify({
            'success': True,
            'prediction': predicted_label,
            'confidence': round(float(confidence * 100), 2),
            'image': f'data:image/jpeg;base64,{img_base64}'
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/upload', methods=['GET'])
def upload():
    return render_template('upload_image.html')

@app.route('/live-test')
def live_test():
    return render_template('live_test.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Global variable for camera
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return None
    return camera

def generate_frames():
    camera = get_camera()
    if camera is None:
        return
        
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process frame for face mask detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = np.expand_dims(face_roi, axis=-1)
                resized_face = tf.image.resize(face_roi, (256, 256))
                
                pred = model.predict(np.expand_dims(resized_face / 255.0, 0), verbose=0)
                # Convert numpy float32 to regular Python float
                pred_value = float(pred[0][0])
                
                predicted_label = 'With Mask' if pred_value <= 0.5 else 'Without Mask'
                confidence = pred_value if pred_value >= 0.5 else 1 - pred_value
                
                color = (0, 255, 0) if predicted_label == 'With Mask' else (0, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.putText(frame, f'{predicted_label}: {confidence*100:.1f}%', 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None
    return jsonify({'status': 'Camera stopped'})
    
if __name__ == '__main__':
    app.run(debug=False, threaded=True)