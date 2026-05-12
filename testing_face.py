import numpy as np
from PIL import Image
import csv

Threshold = 0.80

CSV_file = "face_signature_vector.csv"
Mean_file = "mean_face.npy"
Eigen_file = "eigenfaces.npy"

def load_image(path):
    img  = Image.open(path).convert("L")
    img = img.resize((64,64))
    img = np.array(img,dtype =np.float32)/255.0
    return img.flatten()

def load_training():
    names =[]
    signatures =[]

    with open(CSV_file ,"r") as f:
         reader = csv.reader(f)
         next(reader)
         for row in reader:
             names.append(row[0])
             signatures.append(np.array(row[1:], dtype = np.float32))
    
    return np.array(signatures),names

def cosine_similarity(a,b):
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))


def check(test_image_path):
    signatures,names = load_training()
    mean_face = np.load(Mean_file)
    eigenfaces = np.load(Eigen_file)

    test_img = load_image(test_image_path)
    test_centere = test_img - mean_face

    test_sig = np.dot(test_centere,eigenfaces)
    test_sig /= np.linalg.norm(test_sig)

    best_score = -1
    best_match = None
 
    for train_sig,name in zip(signatures,names):
        score = cosine_similarity(test_sig,train_sig)
        if score > best_score:
           best_score = score
           best_match = name
  
    if best_score >= Threshold :
       print(f"Matched person : {best_match}")
       print(f"similarity score : {best_score:.4f}")
    
    else:
        print("person not found")
        print(f"similarity score : {best_score:.4f}")

if __name__ == "__main__":
     check("mydataset1/testing/test.jpg") 
