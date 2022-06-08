# DLCNet for Plant Leaf Disease Detection and Classification

## Abstract

**An optimized deeper lightweight convolutional neural network architecture for early plant leaves disease detection and classification is proposed in this paper.Plants are highly susceptible to leaf diseases varying from plant to plant. Symptoms of most of the disease are early recognisable and non-differentiable in early stages. Current research presents a solution through deep learning so the plants health can be monitored and if required cure will be given, it will lead to an increase in the quality as well as the better crop production.Proposed model used point-wise and separable convolution block for reducing number of trainable parameters and more connectivity between layers for alleviating the vanishing gradient problem,strengthen feature propagation and motivating feature reusing. Proposed architecture achieved better accuracy on plants tested as well as able to reduce number of trainable parameters compared to Densenet, Resnet, VGG, Alexnet, Xception-network and many more on several metric like number of trainable parameters and classification metrics.**

## Dataset Used

### Dataset Information

![leaf_dataset](https://user-images.githubusercontent.com/46646804/172684024-933e4472-74ac-4746-8544-558019876598.png)

**Link for the dataset**
Citrus Dataset: https://www.kaggle.com/datasets/dtrilsbeek/citrus-leaves-prepared
Grapes Dataset: https://www.kaggle.com/datasets/sakashgundale/grapes-images
Cucumber Dataset: https://www.kaggle.com/datasets/kareem3egm/cucumber-plant-diseases-dataset
Tomato Dataset: https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf

### Image Augmentation Technique

1. Central Crop
2. Random Left Right Flip
3. Random Up Down Flip
4. Random Contrast
5. Random Saturation
6. Random Brightness

## Proposed Methods

### Convolution Operation Used
In place of Traditional convolution layer of 3 * 3 kernel we are using 1 * 3 then 3 * 1 convolution operation and this will results in reducing number of trainable parameters as well as number of multiplications, Below stated how it reduces the both.

Lets suppose we have  N * M * K  as out input for our traditional convolutional layer and we are using 3 * 3 kernel with f filters and stride as 1 then 

total number of multiplication performed are = \n
        3 * 3 * K * (N-2) * (M-2) * f
  
total number of trainable parameters are = 
        3 * 3 * K
 
but if we use 1 * 1 * K kernel size then we use 3 * 3 kernel with same f filters and stride as 1 then 
total number of multiplications performed are = 
        K * N * M  + 3 * 3 * (N-2) * (M-2) * f 

total number of trainable parameters are = 
        K + 3 * 3
        
by using this lot of matrix multiplication are reduced as well as number of trainable parameters are also reduced and it is also found in practical that it will not compromise with the performance of the model.

We will further reduce the computation by substituting 3 * 3 with two different kernels one of size 1 * 3 and another of size 3 * 1. This type of convolution is known as depth-wise convolution layer. This will let us to further reduce the total number of multiplications and number of parameters by a lot.
total number of multiplications performed are = 

        K * N * M + 2 * 3 * (N-2) * (M-2) * f
  
total number of trainable parameters are = 

        K + 2 * 3 

Earlier total number of multiplications performed in model in conventional conventional convolution are:
     \sum_{i=1}^{i_c} ({n_{ic}}*{c_{c}}) + \sum_{i=1}^{i_p} ({p_{p}}) + (3*3*K*(N-2)*(M-2)*f) 
     
Collective block calculations with conventional convolution 
     {c_{c}} =  3*3*K*(N-2)*(M-2)*f 

After using optimization steps, number of multiplications performed are
     \sum_{i=1}^{i_c} (n_{ic}*c_{used}) + \sum_{i=1}^{i_p} ({p_{p}}) + (3*3*K*(N-2)*(M-2)*f)  

 Collective block calculations with used approach 
    {c_{used}} =  K*N*M + 2*3*(N-2)*(M-2)*f  
 
 Passage block calculations 
    {p_{p}} =  K*N*M*f  
 
 Same happens with Number of trainable parameters with conventional convolution are:
     {\sum_{i=1}^{i_c} ({n_{ic}}*{3 * 3 * K}}) + \sum_{i=1}^{i_p} (K) + (3*3*K)
 
 After using optimization steps, number of trainable parameters are
     {\sum_{i=1}^{i_c} ({n_{ic}}*{K + 2*3}}) + \sum_{i=1}^{i_p} (K) + (3*3*K)
 
 In above Equations {i_c} is number of collective block used, {i_p}  is number of passage layers are used and n_{ic} is number of repetition in that collective block.





