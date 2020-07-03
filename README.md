# Codes for Cross-domain representation learning by domain-migration generative adversarial network for sketch based image retrieval published on JVCIR.



##Link

The paper's link is [https://www.sciencedirect.com/science/article/pii/S1047320320300857](https://www.sciencedirect.com/science/article/pii/S1047320320300857). 

## Environment

The core code uses python3.6, Tensorflow 1.8 and [pretrained Inception-v3 model](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained).

##Introduction

*network.py* is the proposed network that we name CDRL.

*data_ut_half.py* contains the code for processing the dataset.

*main_half.py* is used to train the model and generate features.

*retrieval.py* realized the function of retrieval and testing.



