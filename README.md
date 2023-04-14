# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 

## Hyperparameter Tuning
For this project, I chose to apply transfer learning on the Pretrained Resnet50 model provided by the torchvision library. Resnet50 is a convolutional neural network with a total of 50 convolutional and fully connected layers. The model has about 25 million trainable parameters. The provided model is trained on the ImageNet dataset so it has learned to find some relations and insights from training on a large number of images, so using transfer learning, we can transfer the knowledge from the pretrained model and use it to enhance the model for the task of dogbreed classification without needing too much data.

Hyperparameter	Type	             Range
Learning Rate	    Continous	    interval: [0.001, 0.1]
Batch Size	           Categorical	    Values : [32, 64, 128]
Epochs	                 Categorical	  Values: [1, 2]


- Include a screenshot of completed training jobs

![Image1](https://user-images.githubusercontent.com/107848751/231981245-1c8e2823-7c06-4605-969a-9a700e5a304c.png)


- Logs metrics during the training process

![Image2](https://user-images.githubusercontent.com/107848751/231981375-87c8110a-1ad5-4c71-9a3f-fc3d12929bd8.png)


- First Job

![Image3](https://user-images.githubusercontent.com/107848751/231981450-0390ffc4-b288-42fb-8ab9-d65b6a598731.png)


- Second Job

![Image4](https://user-images.githubusercontent.com/107848751/231981524-135334c4-5564-41e0-86e3-981645abab4b.png)


- Third Job

![Image5](https://user-images.githubusercontent.com/107848751/231981582-a3d513f3-ae6d-41f3-a12b-b93361c76eaf.png)


- The Best Hyparameter

![Image6](https://user-images.githubusercontent.com/107848751/231981640-2e455c5e-841f-486a-8313-151cc9542551.png)





## Debugging and Profiling
Model debugging is useful for capturing the values of the tensors as they flow through the model during the training & evaluation phases. In addition to saving & monitoring the tensors, sagemaker provides some prebuilt rules for analizing the tensors and extracting insights that are useful for understanding the process of training & evaluating the model.

I chose the to monitor the Loss Not Decreasing Rule during debugging the model which monitors if the loss isn't decreasing at an adequate rate.

Model Profiling is useful for capturing system metrics such as bottlenecks, CPU utilization, GPU utilization and so on. I used the ProfilerReport rule to generate a profiler report with statistics about the training run.

### Results
Insights from the Plot

* The training loss decreases with the number of steps.
* The training loss is a bit noisy, may be this means that the training might have required a larger batch size.
* The validation loss seems to be almost constant and it is very low compared to the training loss from the beginning which might be a sign of overfitting.
* What to be applied if the plot was erronous Inorder to avoid overfitting we might try the following solutions:

* Maybe I need to use a smaller model compared to the resnet50 like the resnet18 for example.
* Maybe I need to apply regularization to avoid overfitting over the dataset.
* Maybe I need more data for my model..


## Model Deployment

### Overview of Endpoint
The deployed model is a resnet50 model pretrained on the ImageNet dataset and finetuned using the dog breed classification dataset.

The model takes an image of size (3, 224, 224) as an input and outputs 133 values representing the 133 possible dog breeds availabe in the dataset.

The model doesn't apply softmax or log softmax (they are applied only inside the nn.crossentropy loss during training).

The model's output label can be found by taking the maximum over the 133 output values and finding its correponding index.

The model was finetuned for 1 epoch using a batch size of 128 and learning rate ~0.05.


![Image7](https://user-images.githubusercontent.com/107848751/231981719-caca10ba-6492-4a76-a611-7049b67b867f.png)



### Instructions to query the model
* Provide the path of a local image to the Image.open() function from the PIL library to load the image as a PIL image.

* Preprocess the image to prepare the tensor input for the resnet50 network. First the image is resized to (3x256x256) then a center crop is applied to make the image size (3x224x224), the image is then converted to a tensor with values from 0.0 to 1.0 and finally it is normalized by some common known values fro the mean and the standard deviation.

* A request is then sent to the endpoint having the image as its payload

