import numpy as np
from PIL import Image
import csv
import os

k = 10
Training_dataset = "mydataset1/training"

CSV_file = "face_signature_vector.csv"
Mean_file = "mean_face.npy"
Eigen_file = "eigenfaces.npy"

def load_image(path):
    img = Image.open(path).convert("L")
    img = img.resize((64,64))
    img = np.array(img,dtype =np.float32)/255.0
    return img.flatten()

def train():
    images = []
    names = []
  
    for file in os.listdir(Training_dataset):
        if file.lower().endswith((".png",".jpg",".jpeg")):
           path = os.path.join(Training_dataset,file)
           images.append(load_image(path))
           names.append(os.path.splitext(file)[0])
#    print(names)    
    images = np.array(images)
 #   print(images)
 #   print(images.shape)
 
    mean_face = np.mean(images,axis=0)
    np.save(Mean_file,mean_face)
    mean = np.load("mean_face.npy")
#    print(mean)
#    print(mean.shape)

    normalized = images - mean_face
 
    U,S,Vt = np.linalg.svd(normalized ,full_matrices=False)

    eigenfaces = Vt[:k].T
    np.save(Eigen_file,eigenfaces)

    signatures = np.dot(normalized,eigenfaces)
    signatures = signatures/np.linalg.norm(signatures,axis = 1,keepdims=True)
 #   print(signatures)
 #   print(signatures.shape)

    with open(CSV_file,"w",newline = "") as f:
         writer = csv.writer(f)
         writer.writerow(["person_name"] + [f"c{i}" for i in range(k)]) 
         for name, train_sig in zip(names,signatures):
             writer.writerow([name] + train_sig.tolist())

  
if __name__ == "__main__":
    train() 



 
