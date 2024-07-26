# Bird or Drone?

[https://www.allaboutbirds.org/news/wp-content/uploads/2020/04/RBGull-Vyn-FI-1280x720.jpg](https://www.allaboutbirds.org/news/wp-content/uploads/2020/04/RBGull-Vyn-FI-1280x720.jpg)![image](https://github.com/user-attachments/assets/4b532215-e12a-4f5d-9ac2-3eeaee07a8d1)

![image](https://github.com/user-attachments/assets/b16a877b-41c7-48af-a02e-b7f0f27eb8d6)



Using NVIDIA's Jetson Nano for its nueral-core processing and VS Code for its program, I made a convulitional neural network model that is able to differentiate between birds and drones. It can be deployed to any device via simple file transfer and differentiate between birds and drones from a simple video-frame-grab Python program. Its purpose is to help the U.S. Government identify and destroy reconnaissance from enemies more quickly and efficiently.

## The Algorithm

The training for the model occured via the constant modifying of weights on a confidence-prediction algorithm for the drones and birds. To run it, do the following:

## Running this project

YouTube Tutorial -- https://youtu.be/S_KyYnYmsHw

Download into your device via a terminal the folders from this link:

https://drive.google.com/drive/folders/1v6nPyS2BPKYyRJAVdyaNRX3azlSeuOa5?usp=sharing


1. In your terminal, change directories into your Python file model's directory.
2. Run this in terminal:

NET=models/drone_or_bird_model
DATASET=data/drone_or_bird_data

imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt 

$DATASET/path/to/image(s)/{image_file}
