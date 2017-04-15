# Autoregressive_CNN_Predictor

Simplest Model attempted (X2X) from paper: https://arxiv.org/abs/1703.07684v2 on prediction of semantic segmentation of visual scenes . The paper usees generative adversarial method to train convolution net to predict next frames in a visual scene. 

The model contains a generator CNN and a discriminator CNN. 

Generative Network: Contains multi-scale (coarse and fine scale) sub-networks, each sub-network contains convolutional layers interleaved with relu layer. The final layer generated predicted frames. 
Discrimiator Network: Contains multi-scale sub-networks, with convolutional layers, relu layers and fully connected layer. 

Adverserial Training: Generator network predicts next frame. This frame is mixed with ground truth frames and fed to discriminator network. The discriminator network learns to distinguish between generated and real frames. Generate is trained to against the performance of discrimiator by reducing the liklihood of successful classification by discriminator. 

Dataset contains 10,000 sequences each of length 20 showing 2 digits moving in a 64 x 64 frame.http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy 
