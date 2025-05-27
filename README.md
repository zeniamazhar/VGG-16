# Fine-tuning BERT For Gender Classification Using CelebA Dataset From Kaggle

## Introduction 
The main goal here is to perform binary gender classification on a subset of CelebA dataset containing facial images of celebrities, by using transfer learning with a pretrained VGG-16 model. Moreover, 2 learning rates will be experimented to see which one performs better (0.001 vs. 0.0001), and 2 different fine-tuning strategies will be explored: (1) freeze all convolutional layers and train only the classifier head, and (2) freeze all weights, but fine-tune the last convolutional block with the classifier head. In all cases, the number of epochs will be fixed to 10.

## Methodology
### 1. Visualizing/Understanding The Dataset
#### a. 5 random images with their labels
Matplotlib was used to visualize 5 images that were randomly sampled from the dataset. I looped over the indices of the randomly selected images and showed their images as well as their labels (“female” if the Male column has the value of -1, and “male” if it has the value of 1). This was done by enumerating the sampled data to access the rows and get the filenames of each of the sampled images to be able to open them from the images.

#### b. Counts of each gender
Matplotlib was used to visualize the total number of images with males and females. These counts were generated using the value_counts() method and sorted using the sort_index() method. A bar chart was generated to show the number of images of each gender. There were 17320 images of females and 12680 images of males.

#### c. Null values in each column
The isnull() method was used with the sum() method to get the total number of null values present in the dataset for each of the columns, in order to see if any preprocessing step needs to be included to deal with these values. There were 0 null values for all of the columns, so this step wasn’t needed.

#### d. Attribute counts
The number of instances of each of the attributes being present in the images were plotted using Matplotlib.

#### e. Correlations between attributes and gender 
A plot was generated to visualize the correlation between being male and having a certain attribute, where negative correlations show that the attribute is present less in males than in females, and positive correlations show that the attribute is present more in males and less in females.

### 2. Preparing The Data
After using train_test_split from the scikit-learn library to get an 80% train, 10% validation and 10% test split, transform.Compose() from torchvision.transforms was used to transform the images in these ways: (1) Resize images to 224 x 224, (2) convert the images to tensors, (3) normalize them, and (4) do a horizontal flip. A Dataset class was written to take the images and return their images with their labels (label of - 1 was changed to 0 for the binary classification). DataLoader objects were then made by using DataLoader from torch.utils.data, to ensure the training data and the validation data is loaded efficiently, and the batch_size was set to 64. Generally speaking, a batch size of 32, 64 or 128 works well during training, so 64 was selected for the sake of ensuring there’s enough GPU memory and the training isn’t too slow either. 

### 3. Transfer learning with VGG-16
First, the VGG-16 model was loaded with pretrained being set to true. The last layer was modified so that it would be a linear activation layer with one output. Sigmoid activation wasn’t applied instead of linear here, because the binary cross entropy loss function (to be used in the next stage) applies the sigmoid activation itself.

### 4. Fine-Tuning and Training The Model
A function called get_model was written in order to make a model with the fine-tuning strategy specified as its parameter (1 or 2). The strategy can either be strategy 1 (freeze all convolutional layers and only optimize the classifier head) or strategy 2 (freeze all convolutional blocks except for the last one, and train this last block with the classifier head). The next function, train_model is made to take the following parameters: the model (made with get_model()), the train_loader, val_loader, optimizer (can be torch.optim.SGD() or torch.optim.Adam()), the criterion for calculating the loss (in our case this was BCEWithLogitsLoss()), and the number of epochs to be trained (set to 10 by default). As mentioned earlier, the learning rates 0.001 and 0.0001 need to be experimented. Moreover, the two fine-tuning strategies need to be tried, where in one strategy, all the convolutional layers need to be frozen while only training the classifier layers (strategy 1). The other strategy (strategy 2) involves freezing all convolutional layers except for the last convolutional layer block (block 5, which starts from index 24 in vgg.features). So, in total, 4 separate models were trained with all combinations of learning rates and fine-tuning strategies. The loss function was chosen to be BCEWithLogitsLoss(), which computes binary cross entropy loss while internally applying the sigmoid function to the model’s raw outputs (logits). This allows us to use raw logits during training while still evaluating how well the predicted probabilities align with the true binary labels. For the optimizer, which has the task of updating the weights according to the gradients of the loss calculated, was chosen to be the torch.optim.Adam(), where for each batch of training data, optimizer.zero_grad() was used to clear all previous gradients, then loss.backward() was used to backpropagate to compute the gradients, and then optimizer.step() was used to update the weights according to the gradients that were computed, and this was done for 10 epochs. The accuracies and losses were tracked by using separate lists, and are returned at the end of the function. The model was validated after every epoch, with the same methods of calculating the loss and accuracies as in the training stage. 
Another function, plot_loss_curves() was defined in order to use the losses computed in the previous function to generate curves for the training and validation losses for each of the 4 models trained. 

#### 5. Testing The Classifier On Test Set
The evaluate_model_on_test() function was written in order to test the classifier on the test set. The function takes the model, the test_loader and the device as parameters, and uses BCEWithLogitsLoss() to calculate the losses seen in the test set predictions . The accuracy is calculated by counting the number of predictions that were made correctly and dividing this by the total number of predictions made.

## Results
Firstly, 5 randomly selected images were displayed along with their gender labels in order to get an idea on what the images look like and how they relate to their labels (Figure 1).


### Figure 1. 5 randomly sampled images and their target labels

The gender distributions were computed and shown in a bar chart (Figure 2). It can be seen that there are more images with the gender of the celebrity in the image being female than male, where 17320 images were of females and 12680 images were of males.



### Figure 2. Gender distributions in the CelebA30k dataset.

Moreover, a plot showing the number of images with each of the attributes present in them was generated (Figure 3). It can be seen that the ‘No_Beard’ attribute was seen in the highest number of images (nearly 25000), while the attribute seen in the lowest number of images was ‘Bald’ (1000 or so).


### Figure 3. Plot showing the number of images with each attribute.

Additionally, correlations were calculated between the presence of each of the attributes and the gender being male (Figure 4). The blue lines represent the negative correlations, showing the attributes that have a negative correlation with being male. The red lines show the correlations that are positive between being male and the attribute. It can be seen that these are quite consistent. For example, wearing lipstick is expected to be more positively correlated with being female than with being male, and this is seen here as well. On the other hand, attributes like having  a mustache are more positively correlated with being male, meaning they are most likely to be seen in males than in females, which is also expected.

### Figure 4. Correlations between ‘Male’ in the CelebA30k dataset and presence of each attribute.

The 4 different models made with the different combinations of learning rates (0.001 or 0.0001) and the strategy used to train the models (strategy 1 or strategy 2) generated different losses and accuracies at the training and validation stages in each epoch, which can be seen in the figures 5-8, below. The plots showing the training and validation losses can be seen in figures 9-12, which show the overall patterns observed for these. The different combinations of strategies and learning rates were used to generate 4 deep learning models, with a fixed 10 epochs during the training phase for each one. Strategy 1 is the one which involves freezing all the convolutional layers and only optimizing the classifier head during training, while strategy 2 represents the strategy involving freezing all the convolutional layers except for the last convolutional block (block 5, which starts from index 24 in vgg.features). Generally, it can be seen that in 3 out of 4 models, as the number of epochs increases, the training accuracy increases (due to optimizations in the weights of the neurons), but at some point the training accuracy increases such that it is more than the validation accuracy. This can especially be seen in epoch 10 of training in 3 out of 4 deep learning models. This shows that the more epochs are used to optimize the model, the higher the chance of the model overfitting the data. This is more so the case when the learning rate is 0.0001 than when it is 0.001. For example, when strategy 1 is used and the learning rate is set to 0.001, the difference between the training accuracy and the validation accuracy in the 10th epoch is around 2.85% (Figure 5), but when the learning rate is set to 0.0001 (and the strategy is kept as 1), the difference between the training accuracy and the validation accuracy is 3.71%. The same is true for strategy 2, where the difference between the validation and training accuracies is 0% when the learning rate is 0.001, but it is 3.17% when the learning rate is set to 0.0001, though the model trained using strategy 2 with the learning rate set to 0.001 has other issues which will be discussed later. The same pattern is observed with the validation and training losses, where a higher loss is seen for the validation stage than for training, and the differences seen between the learning rates and the strategies is proportional to the ones seen and discussed for the accuracies. The second strategy involves optimizing not just the classifier head, but also the last convolutional block, which may have made it so more epochs of training would be needed for the models trained with the second strategy to result in the same level of overfit as seen with those trained with the first strategy. 
An interesting point to note is that when strategy 2 is used with a learning rate of 0.001, the accuracy actually decreases when we go from epoch 1 (training accuracy: 0.6953) to epoch 2 (training accuracy: 0.5773). This is probably because the learning rate of 0.001 is too high which makes the initial weights (pretrained) get destabilized. This is also supported by the fact that the training accuracy stays stuck at 0.5773 across multiple epochs, as the learning rate being too high makes the accuracy oscillate rather than reaching a global maximum. It becomes clear where the issue lies when we look at the confusion matrix for this model (Figure 11): The model only predicts class 1 (this is most likely the female class) and never predicts class 0 (this is most likely the male class), which may have been because of the data being higher for females, making the likelihood of any given image being that of a female (especially during training) to be higher than that of being male (Figure 2), which may be why the final accuracy is a bit higher than 50%, with a value of 57.73%. Since more weights are being trained in strategy 2 than in strategy 1, the model is much more sensitive to hyperparameters since more of the weights are being updated in order to get a more optimized version of the model, and more epochs would be needed to train all the weights properly. When the learning rate is set to a lower value of 0.0001, it can be seen that the model learns the data well, resulting in a final training accuracy of 99.54% and a validation accuracy of 96.37% (Figure 8), making it the best performing model out of the 4 models. However, as mentioned earlier, this model (and also the other models, other than the one trained using strategy 2 and a learning rate of 0.001) may have overfit the data to some extent, causing the training accuracy to be higher than the validation accuracy. Hence, it’s possible that with more epochs of training and with a learning rate that lies somewhere between 0.001 and 0.0001, the model will be able to learn from the data in a better way and be able to generalize to new data well. 
When it comes to the performance of the models on the test data, it can be seen that the performance on the validation data is nearly proportional to the performance on the test set: The model with the highest validation accuracy at the 10th epoch (which was strategy 2 with learning rate of 0.0001) was the one with the highest accuracy on the test set (96.37%), and the second best (test accuracy: 95.1%) was the one trained with strategy 1 and a learning rate of 0.0001, and the third best (test accuracy: 94.8%) is the one which was trained with strategy 1 and learning rate of 0.001, while the one trained with strategy 2 and learning rate of 0.001 performed the worst (test accuracy: 57.73%). 

### Figure 5. Data about all 10 epochs of training the model using strategy 1 with learning rate = 0.001.

### Figure 6. Data about all 10 epochs of training the model using strategy 2 with learning rate = 0.001.


### Figure 7. Data about all 10 epochs of training the model using strategy 1 with learning rate = 0.0001.

### Figure 8. Data about all 10 epochs of training the model using strategy 2 with learning rate = 0.0001.

### Figure 9. Loss curves for the training (Train) and validation (Val) stages for each of the models.  

### Figure 10. Confusion matrix for the predictions made on the test set by the model trained with strategy 1 and learning rate of 0.001

### Figure 11. Confusion matrix for the predictions made on the test set by the model trained with strategy 2 and learning rate of 0.001

### Figure 12. Confusion matrix for the predictions made on the test set by the model trained with strategy 1 and learning rate of 0.0001

### Figure 13. Confusion matrix for the predictions made on the test set by the model trained with strategy 2 and learning rate of 0.0001



## Discussion 
If we purely base our decision on which model is the best off of the test accuracies, then the best model is the one where strategy 2 was used with the learning rate set as 0.0001. This is also consistent with the idea of overfitting discussed earlier, where out of the models trained with a learning rate of 0.0001, the model with the lowest overfit was the model that was trained with the second strategy and with the learning rate set as 0.0001. This may have been because the number of epochs needed to reach the same amount of overfit as the one trained with strategy 1 is higher due to the number of weights that need to be trained being much higher in strategy 2 than in strategy 1. This answer may have been different if fewer epochs were used to train the models, as it would decrease the amount of overfit seen in the models on the data. 
## Conclusion
In conclusion, in this assignment, the aim of training deep neural network models by using transfer learning of a pretrained VGG-16 model with the task of identifying the gender of the celebrity images from the CelebA dataset, was achieved successfully. All 4 models, each trained with a different combination of learning rate (either 0.001 or 0.0001 for each model) and training strategy (strategy 1: freezing all convolutional blocks and optimizing the classifier head alone, or strategy 2: freezing all convolutional blocks except for the last one, resulting in optimizing the classifier head along with the last convolutional block) The model that performed the best on the test set was the one that used the 2nd strategy (freezing all convolutional convolutional blocks except for the last one and optimizing this last block along with the classifier head) with the learning rate being set to 0.0001. This answer may have been different if fewer epochs were used to train the models, as it would decrease the amount of overfit seen in the models on the data. Moreover, the models trained with the first strategy being trained on fewer epochs than the ones trained with the second strategy may have been the right decision, since the number of weights to be optimized were fewer here than in the model trained with the second strategy, since the second strategy involved optimizing the weights in last convolutional block in addition to the classifier head, which should require more epochs of training.

			
		
