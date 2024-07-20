from flask import Flask, request, jsonify
import tensorflow as tf
from keras.models import load_model
import datetime
import numpy as np
import cv2
from mtcnn import MTCNN

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
import cv2
from sklearn.model_selection import train_test_split
import mtcnn
from PIL import Image

IMG_SIZE = 224
BATCH_SIZE = 32
unique_students = np.array(['210303108114', '210303108221', '210303108324', '210303108332',
       '2203031080049', '2203031080092', '2203031080127', '2203031080174',
       '31', '42', '46'], dtype='<U13')

def process_image(image_path, img_size = IMG_SIZE):
    """
    Takes an image file path and turns the image into a Tensor
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=tf.constant([224, 224]))
    return image

def create_data_batches(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creates batches of data out of image (X) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle if it's validation data.
    Also shuffles test data as input (no labels)
    """
    if test_data:
        print('Creating test data batches...')
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

    elif valid_data:
        print('Creating validation data batches...')
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = data.map(get_image_label).batch(BATCH_SIZE)
        return data_batch

    else :
        print('Creating training data batches...')
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data = data.map(get_image_label)
        data_batch = data.batch(BATCH_SIZE)
    return data_batch

def load_model_func(model_path):
  """Loads a saved TensorFlow/Keras model.

  Args:
    model_path: Path to the saved model file.

  Returns:
    The loaded model.
  """

  try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
    return model
  except Exception as e:
    print(f"Error loading model: {e}")
    return None

def process_face(face_image, img_size=IMG_SIZE):
    """
    Takes a face image and preprocesses it to the required format
    """
    face_image = tf.convert_to_tensor(face_image, dtype=tf.float32)
    # face_image = tf.expand_dims(face_image, axis=0)
    face_image = face_image[tf.newaxis, ...]
    face_image = tf.image.resize(face_image, [img_size, img_size])
    face_image = face_image / 255.0  # Normalize the image to [0, 1]
    return face_image

def predict_photo(image_path, face_recognizer, unique_students, save_folder_path, confidence_threshold=50):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Open the original image for visualization
    img_pil = Image.open(image_path)
    img = np.array(img_pil)

    # Initialize MTCNN for face detection
    detector = MTCNN()

    # Detect faces in the image
    faces = detector.detect_faces(img)
    recognized_faces = []

    for face in faces:
        # Extract face bounding box coordinates
        x, y, w, h = face['box']
        x1, y1 = x + w, y + h

        margin = int(0.1 * min(w, h))  # Calculate 10% of the smaller dimension (width or height)

        # Adjust coordinates to increase bounding box by 10%
        x_adjusted = max(0, x - margin)
        y_adjusted = max(0, y - margin)
        x1_adjusted = min(img.shape[1], x1 + margin)
        y1_adjusted = min(img.shape[0], y1 + margin)
        
        # Crop the face from the original image
        roi_color = img[y_adjusted:y1_adjusted, x_adjusted:x1_adjusted]

        # Preprocess the face (assuming process_face function is already defined)
        roi_color_resized = process_face(roi_color)

        # Make prediction
        predictions = face_recognizer.predict(roi_color_resized)
        predicted_ind = np.argmax(predictions[0])
        predicted_label = unique_students[predicted_ind]
        confidence = predictions[0][predicted_ind] * 100

        # Determine label text
        if confidence > confidence_threshold:
            label_text = str(predicted_label)
        else:
            label_text = 'Unknown'

        # Draw the bounding box and label on the image
        cv2.rectangle(img, (x_adjusted, y_adjusted), (x1_adjusted, y1_adjusted), (0, 0, 255), 2)
        cv2.putText(img, f"{label_text}: {confidence:.2f}%", (x1_adjusted, y1_adjusted-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        recognized_faces.append((label_text, confidence))

    # Add the count of detected faces to the top left of the image
    face_count_text = f"Count: {len(faces)}"
    cv2.putText(img, face_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save the processed image
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(save_folder_path, output_filename)
    cv2.imwrite(output_path, img)

    return recognized_faces, len(faces)

def predict_robustly(image_path, face_recognizer, unique_students, save_folder_path, confidence_threshold=25):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Open the original image for visualization
    img_pil = Image.open(image_path)
    img = np.array(img_pil)

    # Initialize MTCNN for face detection
    detector = MTCNN()

    # Detect faces in the image
    faces = detector.detect_faces(img)
    recognized_faces = []

    # Track assigned labels to ensure uniqueness
    assigned_labels = set()

    for face in faces:
        # Extract face bounding box coordinates
        x, y, w, h = face['box']
        x1, y1 = x + w, y + h
        
        margin = int(0.1 * min(w, h))  # Calculate 10% of the smaller dimension (width or height)

        # Adjust coordinates to increase bounding box by 10%
        x_adjusted = max(0, x - margin)
        y_adjusted = max(0, y - margin)
        x1_adjusted = min(img.shape[1], x1 + margin)
        y1_adjusted = min(img.shape[0], y1 + margin)
        
        # Crop the face from the original image
        roi_color = img[y_adjusted:y1_adjusted, x_adjusted:x1_adjusted]

        # Preprocess the face (assuming process_face function is already defined)
        roi_color_resized = process_face(roi_color)

        # Make prediction
        predictions = face_recognizer.predict(roi_color_resized)
        predicted_ind = np.argmax(predictions[0])
        label = unique_students[predicted_ind]
        confidence = predictions[0][predicted_ind] * 100

        # Determine label text
        if confidence > confidence_threshold:
            label_text = label
        else:
            label_text = 'Unknown'

        # Check if the label is already assigned
        if label_text not in assigned_labels:
            assigned_labels.add(label_text)
        else:
            label_text = 'Unknown'  # Set label to Unknown if label is not unique

        # Draw the bounding box and label on the image
        cv2.rectangle(img, (x_adjusted, y_adjusted), (x1_adjusted, y1_adjusted), (0, 0, 255), 2)
        cv2.putText(img, f"{label_text}: {confidence:.2f}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        recognized_faces.append((label_text, confidence))

    # Add the count of detected faces to the top left of the image
    face_count_text = f"Count: {len(faces)}"
    cv2.putText(img, face_count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Save the processed image
    output_filename = os.path.basename(image_path)
    output_path = os.path.join(save_folder_path, output_filename)
    cv2.imwrite(output_path, img)

    return recognized_faces, len(faces)

def predict_from_folder(custom_folder_path, model):

    final_faces = []
    test_images_dir = custom_folder_path

    # Get current timestamp for folder naming
    now = datetime.datetime.now()
    timestamp = now.strftime("%H-%M-%S-%d-%m-%Y")
    save_folder_path = f"C_{timestamp}"
    subfolder_path = os.path.join("Captured", save_folder_path)
    os.makedirs(subfolder_path, exist_ok=True)

    total_faces = 0
    
    # model = load_model_func("Models/D_V2\CNN-GAP-Adam_20240719_235915.h5")

    for filename in os.listdir(test_images_dir):
        try:
            file_path = os.path.join(test_images_dir, filename)
            if os.path.isfile(file_path):
                for ext in ['jpg', 'jpeg', 'png']:
                    if filename.lower().endswith(ext):
                        print("Reading", file_path)
                        face_details, num_faces = predict_robustly(
                                                        image_path=file_path,  
                                                        face_recognizer=model,
                                                        unique_students=unique_students,
                                                        save_folder_path=subfolder_path
                                                    )
                        total_faces += num_faces
                        final_faces.append(face_details)
                        break
                else:
                    print(f"Skipping {filename}: File format not supported (must be jpg, jpeg, or png)")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

app = Flask(__name__)

# Load your CNN model
model = load_model_func("Models/D_V2\CNN-GAP-Adam_20240719_235915.h5")

@app.route('/predictPhoto', methods=['POST'])
def predict():
    """Endpoint to handle prediction requests."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_path = os.path.join("temp", file.filename)
    file.save(file_path)

    # Make prediction
    prediction = predict_photo(file_path, model, unique_students, "Captured")
    response = {
        'prediction': prediction.tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)