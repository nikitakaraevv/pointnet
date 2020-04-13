# PointNet
PyTorch implementation of "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation" https://arxiv.org/abs/1612.00593

## Classification dataset
This code implements object classification on [ModelNet10](https://modelnet.cs.princeton.edu) dataset.

As in the original paper, we sample 1024 points on the objects surface depending on the area of the current face. Then we normalize the object to a unit sphere and add Gaussian noise. This is an example of input to the neural network that represents a chair:

<img src="images/chair.gif" alt="matching points" width="400"/> 

You can download the dataset by following [this link](https://drive.google.com/open?id=12Mv19pQ84VO8Av50hUXTixSxd5NDjeEB)

## Classification performance

| Class (Accuracy) | Overall | Bathtub | Bed| Chair|Desk|Dresser|Monitor|Night stand|Sofa|Table|Toilet|
| :---: |:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ModelNet10 | 82.0% | 93.4% | 92.0% | 97.2% | 81.5% | 71.0% | 89.4% | 56.0% |86.9%| 93.4% |95.9%|



## Part segmentation dataset
The dataset includes 2609 point clouds representing different airplanes, where every point has its coordinates in 3D space and a label of an airplane’s part the point belongs to. As all images have different number of points and PyTorch library functions require images of the same size to form a PyTorch tensor, we sample uniformly 2000 points from every point cloud.

You can download the dataset by following [this link](https://drive.google.com/drive/u/1/folders/1Z5XA4uJpA86ky0qV1AVgA_G1_ETkq9En)

## Part segmentation performance
The resulting accuracy on the validation dataset is 88%. In the original paper part segmentation results corresponding to category of objects (airplanes) is 83.4%.

<img src="images/airplane.gif" alt="matching points" width="400"/> 

## Authors
* [Nikita Karaev](https://github.com/nikitakaraevv)
* [Irina Nikulina](https://github.com/washburn125)
