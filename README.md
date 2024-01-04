# Project Title
Spatial Propagation Learning and Segmentation for 3D Medical Images
## Introduction
This project developed a neural network model for efficient 3D medical image segmentation, which includes an optical flow propagation model and an affinity matrix model. 
In the optical flow model, we constructed a shared convolutional neural network, establishing the foundation for two Feature Pyramid Networks (FPNs) for feature extraction. 
Additionally, we developed a propagation model using optical flow algorithms to extend segmentation information across adjacent slices. 
This method enabled the efficient annotation of entire 3D images, starting from a single segmented slice.

In the affinity matrix model, to enhance accuracy, we integrated a spatial affinity matrix network, refining segmentation results.
