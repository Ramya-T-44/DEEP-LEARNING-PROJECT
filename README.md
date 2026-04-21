# DEEP-LEARNING-PROJECT
*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : TADEPALLLI R N V SAI RAMYA

*INTERN ID* : CTIS6700

*DOMAIN* : DATA SCIENCE

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTHOSH

**DESCRIPTION**

# Task 2: Deep Learning Model using PyTorch

Task 2 focused on implementing a deep learning model using PyTorch to perform image classification on the MNIST dataset. The MNIST dataset consists of handwritten digits ranging from 0 to 9 and is widely used as a benchmark dataset for testing machine learning models.

The task began with loading the dataset using the torchvision library. The dataset was automatically downloaded and transformed into tensors, which are required for PyTorch models. Data normalization was applied to scale pixel values, ensuring faster and more stable training.

A Convolutional Neural Network (CNN) architecture was designed for this task. CNNs are particularly effective for image-related tasks as they can capture spatial features using convolutional layers. The model included convolutional layers, activation functions (ReLU), pooling layers, and fully connected layers. Each layer plays a specific role in extracting and learning patterns from the input images.

The training process involved defining a loss function and an optimizer. CrossEntropyLoss was used as the loss function since this is a classification problem. The Adam optimizer was used to update model weights efficiently. The model was trained over multiple epochs, where each epoch represents one complete pass through the dataset.

During training, the loss value was monitored to observe how well the model was learning. A decreasing loss indicates that the model is improving. After training, the model was evaluated on a test dataset to measure its performance on unseen data. The model achieved an accuracy of around 98%, which demonstrates the effectiveness of CNNs for image classification tasks.

Additionally, predictions were generated for sample images to verify the model’s performance visually. This step helps in understanding how the model interprets input data and produces output.

This task provided practical exposure to deep learning concepts such as neural networks, backpropagation, and optimization. It also highlighted the importance of data preprocessing and model architecture design. Deep learning models like CNNs are widely used in applications such as image recognition, medical imaging, and autonomous vehicles.

Overall, Task 2 enhanced understanding of how deep learning models are built and trained using PyTorch, making it a valuable experience for real-world AI applications.

** OUTPUT** 
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/36cadace-e9c5-4ac8-831b-12d3a51e0629" />
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/1503aaec-e2b4-47ac-8e9d-346a85a36c4e" />
<img width="640" height="480" alt="Image" src="https://github.com/user-attachments/assets/dedb26ea-0533-4027-b6ca-ea52b152b3f1" />
