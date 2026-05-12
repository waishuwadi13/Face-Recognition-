# Face Recognition System using Eigenfaces and Cosine Similarity

## Overview
This project is a Python-based Face Recognition System that identifies a person from an input image using the Eigenfaces method and Cosine Similarity. The system is trained using neutral face images and tested using smiling face images. It compares the test image with trained facial feature vectors and returns the matched person's name along with the similarity score.

---

# Technologies Used

- Python
- NumPy
- PIL (Python Imaging Library)
- CSV
- PCA (Principal Component Analysis)
- SVD (Singular Value Decomposition)

---

# Project Workflow

The project works in two phases:

1. Training Phase
2. Testing / Recognition Phase

---

# Training Phase (`training_face1.py`)

## Step 1: Load and Preprocess Images
- Read all training images from the dataset folder.
- Convert images to grayscale.
- Resize images to 64×64.
- Normalize pixel values between 0 and 1.
- Flatten images into 1D vectors.

## Step 2: Create Mean Face
- Calculate the average face from all training images.
- Save the mean face using NumPy.

## Step 3: Normalize Images
- Subtract the mean face from all training images.
- This removes common facial information and keeps important facial variations.

## Step 4: Apply PCA using SVD
- Perform Singular Value Decomposition (SVD) on normalized images.
- Extract top Eigenfaces for dimensionality reduction.

## Step 5: Generate Face Signature Vectors
- Project normalized images onto Eigenface space.
- Create compact facial feature vectors.
- Normalize signature vectors for accurate comparison.

## Step 6: Store Training Data
- Save:
  - Person names
  - Face signature vectors
- Store data in CSV format for future matching.

---

# Testing Phase (`testing_face.py`)

## Step 1: Load Saved Training Data
- Load:
  - Face signature vectors
  - Person names
  - Mean face
  - Eigenfaces

## Step 2: Preprocess Test Image
- Convert image to grayscale.
- Resize to 64×64.
- Normalize pixel values.
- Flatten image into vector form.

## Step 3: Generate Test Signature
- Subtract mean face from test image.
- Project test image onto Eigenface space.
- Generate normalized feature vector.

## Step 4: Compare Faces
- Use Cosine Similarity to compare:
  - Test image vector
  - Training image vectors

## Step 5: Identify Person
- Find the highest similarity score.
- If similarity score is greater than threshold:
  - Display matched person's name
  - Display similarity score
- Otherwise:
  - Display "Person not found"

---

# Output

Example Output:

text
Matched person : person1
similarity score : 0.9234


or

text
person not found
similarity score : 0.5123


---

# Features

- Face recognition using Eigenfaces
- PCA-based dimensionality reduction
- Cosine similarity matching
- Grayscale image preprocessing
- Efficient facial feature extraction
- CSV-based storage of facial signatures

---

# Dataset Structure

```text
mydataset1/
│
├── training/
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
│
├── testing/
│   └── test.jpg
```

