import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st 

st.header('Fashion Recommendation System')

# Load image features and filenames
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Function to extract features from the uploaded image
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)  # Normalize the result
    return norm_result

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# Fit the NearestNeighbors model on the image features
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')

# Ensure n_neighbors doesn't exceed the number of samples in the dataset
n_neighbors = min(6, len(Image_features))  # Adjust n_neighbors to the number of samples in the dataset
neighbors.set_params(n_neighbors=n_neighbors)

neighbors.fit(Image_features)

# Streamlit file uploader
upload_file = st.file_uploader("Upload Image")
if upload_file is not None:
    # Save the uploaded file
    with open(os.path.join('upload', upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())
    
    st.subheader('Uploaded Image')
    st.image(upload_file)
    
    # Extract features from the uploaded image
    input_img_features = extract_features_from_images(upload_file, model)
    
    # Debug: print the shape of the extracted feature vector
    st.write(f"Shape of the extracted feature vector: {input_img_features.shape}")
    
    # Find the nearest neighbors of the uploaded image
    distance, indices = neighbors.kneighbors([input_img_features])

    # Debug: print the indices and distances
    st.write(f"Indices of the recommended images: {indices}")
    st.write(f"Distances to the recommended images: {distance}")
    
    # Display recommended images
    st.subheader('Recommended Images')
    col1, col2, col3 = st.columns(3)
    
    # Debug: Show which images are being recommended
    st.write(f"Recommended image filenames: {filenames[indices[0][1]]}, {filenames[indices[0][2]]}, {filenames[indices[0][3]]}")
    
    with col1:
        st.image(filenames[indices[0][1]])  # Display the first recommended image
    with col2:
        st.image(filenames[indices[0][2]])  # Display the second recommended image
    with col3:
        st.image(filenames[indices[0][3]])  # Display the third recommended image
