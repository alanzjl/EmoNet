# EmoNet

Built by ***Jialiang Zhao***
    
Most data is labelled by ***Jiawei Zhang***
    
Other data contributors: (From EmoWeb)
    
***Jiawen Wu, Chenyang Ling, Hao Lu, Jialiang Zhao, (don't know your name)***
    
----------------------
    
#### Platform: 
Python 3.6, Google Tensorflow 1.1, Numpy, Jupyter
    
#### Inspired By:
    
Beijing Institute of Technology, *Pattern Recognition* by ***Prof. Qi Gao***
    
Stanford, CS231n *Convolutional Neural Networks in Visual Recognition* by ***Prof. Fei Fei Li***
    
#### Network Structure:
    
| Layer Name | Input Size | Output Size | Comment|
| ------ | ------ | ------ | ------ |
|**Input**| 144 x 3| - | - |
|**Augmentation (Affine)**| 144 x 3 | 32 x 32 x 3 | vectorized and augmented|
|**ConvLayer32**| 32 x 32 x 3 | 28 x 28 x 32 | Nonlinear Activation: RELU |
|**2x2 Maxpool**| 28 x 28 x 32 | 28 x 28 x 32 | - |
|**ConvLayer64 1**| 28 x 28 x 32 | 24 x 24 x 64 | Nonlinear Activation: RELU |
|**ConvLayer64 2**| 24 x 24 x 64 | 20 x 20 x 64 | Nonlinear Activation: RELU |
|**2x2 Maxpool**| 20 x 20 x 64 | 20 x 20 x 64 | - |
|**ConvLayer128**| 20 x 20 x 64 | 16 x 16 x 128 | Nonlinear Activation: RELU |
|**2x2 Maxpool**| 16 x 16 x 128 | 16 x 16 x 128 | - |
|**ConvLayer256**| 16 x 16 x 128 | 12 x 12 x 256 | Nonlinear Activation: RELU |
|**2x2 Maxpool**| 12 x 12 x 256 | 6 x 6 x 256 | reduced pooling, for efficiency in FCNN |
|**Affine**| 6 x 6 x 256 = 9216 | 1024 | Nonlinear Activation: RELU | 
|**Dropout**| - | - | reduce overfitting | 
|**Affine**| 1024 | 1024 | Nonlinear Activation: RELU | 
|**Dropout**| - | - | reduce overfitting | 
|**Output Affine**| 1024 | 5 | - |
    
#### Total Trainable Parameters: 
150 MB *(float32)*
    
#### Training Cost: 
About ***30 hr*** CPU (aborted) and ***4 hr*** GPU
    
#### Runtime Cost: 
***5.44 ms +- 103 us*** With GPU (NV Quadro M2000M, 4GB)
    
***20.32 ms +- 96 us*** With CPU (Intel Xeon v5 1505M, 2 of 4 cores used)\n
