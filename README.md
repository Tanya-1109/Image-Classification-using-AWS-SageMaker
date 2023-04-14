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

![Photo1]("https://d-clkas2kj3gz9.studio.us-east-1.sagemaker.aws/jupyter/default/files/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/Photos/Image1.png?_xsrf=2%7C9254bf6e%7Cc2b46b35c80f3fb2f83f41c82884806d%7C1681381479")


- Logs metrics during the training process

![Photo2]("https://d-clkas2kj3gz9.studio.us-east-1.sagemaker.aws/jupyter/default/files/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/Photos/Image2.png?_xsrf=2%7C9254bf6e%7Cc2b46b35c80f3fb2f83f41c82884806d%7C1681381479")

- First Job

![Photo3]("https://d-clkas2kj3gz9.studio.us-east-1.sagemaker.aws/jupyter/default/files/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/Photos/Image3.png?_xsrf=2%7C9254bf6e%7Cc2b46b35c80f3fb2f83f41c82884806d%7C1681381479")

- Second Job

![Photo4]("https://d-clkas2kj3gz9.studio.us-east-1.sagemaker.aws/jupyter/default/files/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/Photos/Image4.png?_xsrf=2%7C9254bf6e%7Cc2b46b35c80f3fb2f83f41c82884806d%7C1681381479")

- Third Job

![Photo5]("https://d-clkas2kj3gz9.studio.us-east-1.sagemaker.aws/jupyter/default/files/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/Photos/Image5.png?_xsrf=2%7C9254bf6e%7Cc2b46b35c80f3fb2f83f41c82884806d%7C1681381479")

- The Best Hyparameter

![Photo6]("https://d-clkas2kj3gz9.studio.us-east-1.sagemaker.aws/jupyter/default/files/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/Photos/Image6.png?_xsrf=2%7C9254bf6e%7Cc2b46b35c80f3fb2f83f41c82884806d%7C1681381479")




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


![Photo7]("https://d-clkas2kj3gz9.studio.us-east-1.sagemaker.aws/jupyter/default/files/CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/Photos/Image6.png?_xsrf=2%7C9254bf6e%7Cc2b46b35c80f3fb2f83f41c82884806d%7C1681381479")


### Instructions to query the model
* Provide the path of a local image to the Image.open() function from the PIL library to load the image as a PIL image.

* Preprocess the image to prepare the tensor input for the resnet50 network. First the image is resized to (3x256x256) then a center crop is applied to make the image size (3x224x224), the image is then converted to a tensor with values from 0.0 to 1.0 and finally it is normalized by some common known values fro the mean and the standard deviation.

* A request is then sent to the endpoint having the image as its payload

