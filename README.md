# DLCNet for Plant Leaf Disease Detection and Classification

## Abstract

**An optimized deeper lightweight convolutional neural network architecture for early plant leaves disease detection and classification is proposed in this paper.Plants are highly susceptible to leaf diseases varying from plant to plant. Symptoms of most of the disease are early recognisable and non-differentiable in early stages. Current research presents a solution through deep learning so the plants health can be monitored and if required cure will be given, it will lead to an increase in the quality as well as the better crop production.Proposed model used point-wise and separable convolution block for reducing number of trainable parameters and more connectivity between layers for alleviating the vanishing gradient problem,strengthen feature propagation and motivating feature reusing. Proposed architecture achieved better accuracy on plants tested as well as able to reduce number of trainable parameters compared to Densenet, Resnet, VGG, Alexnet, Xception-network and many more on several metric like number of trainable parameters and classification metrics.**

## Dataset Used

### Dataset Information

![leaf_dataset](https://user-images.githubusercontent.com/46646804/172684024-933e4472-74ac-4746-8544-558019876598.png)

**Link for the dataset** <br/> <br/>
Citrus Dataset: https://www.kaggle.com/datasets/dtrilsbeek/citrus-leaves-prepared <br/>
Grapes Dataset: https://www.kaggle.com/datasets/sakashgundale/grapes-images <br/>
Cucumber Dataset: https://www.kaggle.com/datasets/kareem3egm/cucumber-plant-diseases-dataset  <br/>
Tomato Dataset: https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf <br/>

### Image Augmentation Technique

1. Central Crop
2. Random Left Right Flip
3. Random Up Down Flip
4. Random Contrast
5. Random Saturation
6. Random Brightness

## Proposed Methods

### Convolution Operation Used

![1](https://user-images.githubusercontent.com/46646804/172694081-35006616-e71f-4cdb-919d-ce924f4c39fc.jpg)
![2](https://user-images.githubusercontent.com/46646804/172694106-de3d0f77-463b-43ce-b399-a299d019ec91.jpg)




### Proposed Optimizer

Current state-of-art methods are using Adam as optimizer for most of the plant disease detection classification tasks. This paper uses Ada-grad optimizer with some modification. Momentum term is added in it. Now optimizer used have adaptive learning rate as well as momentum term they will lead to slowing of learning rate  as well as moving in the direction of novel weights (reduces oscillations while training) which are ideal for classification. This will results in reducing of work giving better training.

![3](https://user-images.githubusercontent.com/46646804/172694756-e12de493-5906-41c6-ab80-3bc147b9d1b0.jpg)
![4](https://user-images.githubusercontent.com/46646804/172694776-c65a7064-3879-4327-9793-4f12a6e50a76.jpg)


where L is loss calculated by the model while training, \beta lies between 0 , 1 its value used is 0.99 it is the coefficient for momentum basically it takes weighted average of the weights into the account if \beta value is 0 means there is no momentum term, \eta is learning rate its value used is 0.001 and \epsilon is small positive value used to avoid divisibility by zero.

### DLC-Net Architecture and Explanation

***Proposed Model solves these two problems:***

1. Reduces the number of trainable parameters as well as number of multiplication operations without compromising the evaluation metrics.

2. Resolves the vanishing gradient problem as much as it can.

Reducing the number of trainable parameters will results in much faster training of model as well as much faster getting results or class while testing as well as vanishing gradient problem affects a lot while training which will results in slower updating of weight and less likely to achieve ideal weights are mandatory for better classification.

***Layer wise information:***

1. Collective Block, it tries to extract deeper features which are helpful in classification. In this block we replaces 3 * 3 mask with 1 * 3 and 3 * 1 mask as it is discussed above that it will results in reducing number of trainable parameters as well as number of matrix multiplications without compromising the evaluation metrics and in this block we are frequently using batch normalization layer for normalizing the feature vector, it helps in better learning of model. In this block, different masks or filters are used for different feature extraction and because of large number of features it tries to learn as much features as it learns as well as connectivity between layers ensures that it minimizes vanishing gradient in each collective block.

2. Passage layer, it is used for minimizing vanishing gradient problem across each collective block as is provides extra smaller path for gradient to travel while back propagation. It contains layers in such a way that it ensures the output feature vector for both are both have same dimensions and are compatible for concatenation.

3. Dense layer, it matches the number of classes so that there will be neuron corresponding to each class which will be activated according to the class it belongs to.

4. Different pooling layers are also used for reducing the size of feature vector which is coming as input to next layer. Batch Normalization layer is used for normalizing the feature vector and ReLU is the activation function which is used except in the Dense layer, In Dense layer activation used is softmax to give the probability for each present classes. Rest layers are used for better flow of learning while training the proposed model.

![proposed_architecture](https://user-images.githubusercontent.com/46646804/172687269-5c8f395b-8e09-45f8-ad1b-d591020fe626.png)


### Implementation Flow

Firstly we collect the data and then partition the data into training and testing set accordingly to number of images present in the dataset.The ratio of training images and testing images differs on the basis of total number of images of each class present in the dataset.Then normalize the image as then comparing the model on different distribution we will train the models with simple images, augmented images and generated images using generative adversial network depends on the necessity. After that we select the model which performs better in environmental conditions and that will be our final model.

![flowchart](https://user-images.githubusercontent.com/46646804/172687431-5e4af9fd-9b58-49e3-919b-21b6cdf58fb0.png)

### Heat-Map for dataset

![heat_map](https://user-images.githubusercontent.com/46646804/172687677-b8209a75-7a81-4c5a-bac9-79c1f011ab4f.png)







