# Shark Identification and Classification for Conservation

Our App is available here: https://shark-id.streamlit.app/

This work is part of LeWagon's project weeks - Data Science batch #1271.

Project overview 🚀
The aim of this project was to develop an app that could accurately identify the species of a shark by analyzing an input image. To accomplish this, we utilized a Convolutional Neural Network (CNN) that was trained on the Kaggle Shark Species Dataset. Once the network had undergone training, we packaged it into a Docker container for deployment as a web application through Google Cloud Run. The end result is a user-friendly app that can swiftly determine the shark species from a given image.

Dataset
The original dataset is available here: https://www.kaggle.com/datasets/larusso94/shark-species . It contains 14 different species of sharks, each class containing 73 to 135 images, which makes a total of 1549 images.

To address a compatibility issue with the EfficientNet model, we specifically installed TensorFlow version 2.9.1. This was necessary because the EfficientNet model required an older version of TensorFlow to save the model in the h5 format successfully.

Data Preprocessing 🚨
To prepare the shark image data for training the model, the following preprocessing steps were performed:

Class Labels:
After loading the dataset each shark species is assigned a numerical label. The class labels used in this project are as follows:

'basking': 0
'blue': 1
'hammerhead': 2
'mako': 3
'sand tiger': 4
'tiger': 5
'white': 6
'blacktip': 7
'bull': 8
'lemon': 9
'nurse': 10
'thresher': 11
'whale': 12
'whitetip': 13

Image Resizing and Color Conversion:
Each image is resized to a dimension of 224x224 pixels to ensure consistency.
The images are converted from BGR to RGB color format using OpenCV.

Creating Input and Target Arrays:
The resized images are converted into NumPy arrays and stored in the 'imgs' list.
Corresponding labels for the images are stored in the 'labels' list.
The labels are one-hot encoded using the 'to_categorical' function.

Data Shuffling:
The data is shuffled using a random permutation to ensure randomness in the training, validation, and test sets.

Train-Validation-Test Split:
The data is split into three sets: train, validation, and test sets.
Approximately 1/6th of the data is used for testing, and 20% of the data is allocated for validation.
The remaining data is used for training the model.

Model 🧠
After thorough testing of different models, we concluded that the most suitable option for our project is the EfficientNet B3 model, which comes pre-trained. To ensure the model's performance, we made the initial layers untrainable. We then added a flattening layer on top of the pre-trained model, followed by three dense layers, two dropout layers, and a prediction layer. The prediction layer consists of a fully connected layer with 14 outputs, representing each species class.

flatten_layer = layers.Flatten()
    dense_layer1 = layers.Dense(500, activation='relu', kernel_regularizer=regularizers.L1(0.0001))
    dropout_layer1 = layers.Dropout(0.3)
    dense_layer2 = layers.Dense(400, activation='relu', kernel_regularizer=regularizers.L1(0.0001))
    dropout_layer2 = layers.Dropout(0.3)
    dense_layer3 = layers.Dense(300, activation='relu', kernel_regularizer=regularizers.L1(0.0001))
    dropout_layer3 = layers.Dropout(0.3)
    prediction_layer = layers.Dense(14, activation='softmax')

To train the model, we utilized the Kaggle Shark Species Dataset, conducting 85 epochs with a batch size of 16. The Adam optimizer was employed with a learning rate of 1e-4, and early stopping was implemented with a patience of 20.

Metrics ⏱️
In evaluating our model, we utilized accuracy as the primary metric. Accuracy is determined by the ratio of correct predictions to the total number of predictions. Our model achieved an accuracy of 76%. For compiling the model, we utilized categorical_crossentropy as the loss function.

Improving the accuracy of the model 💪
To enhance the model's accuracy, we implemented data augmentation techniques. By applying transformations to the training dataset images, we augmented the size and diversity of the dataset. The following data augmentation configuration was utilized:

datagen = ImageDataGenerator(
    featurewise_center = False,
    featurewise_std_normalization = False,
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    zoom_range = (0.8, 1.2),
    preprocessing_function=preprocess_input
    )

datagen.fit(X_train)
datagen

By leveraging data augmentation, we were able to introduce variations in the training data, leading to a more robust and accurate model.

Results 📊
The model achieved an overall accuracy of 76% on the test dataset. When examining the individual class performance, the F1-score varied across different shark species.

We obtained impressive F1-scores for the Hammerhead, Basking, and Thresher classes, indicating precise and accurate classification. However, the Santiger, Bull, and Lemon classes had lower F1-scores, suggesting room for improvement in their classification accuracy. These results underscore the variations in the model's performance across different shark species, with certain classes achieving high accuracy (such as Basking and Hammerhead), while others present opportunities for enhancement (such as Santiger).

F1 scores for all 14 classes ordered in ascending order:

Sand tiger: 0.57
Bull: 0.57
Lemon: 0.60
Tiger: 0.63
Nurse: 0.67
Whitetip: 0.72
Blacktip: 0.72
White: 0.72
Blue: 0.79
Mako: 0.82
Whale: 0.87
Thresher: 0.91
Basking: 0.94
Hammerhead: 1.00

To improve the classification accuracy of the Santiger, Bull, and Lemon classes, we could consider increasing the training data. Additionally, fine-tuning the model's architecture or exploring attention mechanisms may help address the variations in performance across different shark species and further enhance overall accuracy.

Links
Web interface is based on streamlit and located in a separate repository: https://github.com/nikkernoodle/shark-id-front.
