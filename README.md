# Food101 CNN Classification

## Project Overview
This project involves building a Convolutional Neural Network (CNN) model to classify images from the Food101 dataset. The dataset contains 101 food categories, and the task is to accurately classify an image into one of these categories.

## Dataset
The dataset used for this project is the **Food101** dataset. It consists of 101 food categories with 1,000 images per class.

- **Dataset Source**: [Food101 Dataset on Kaggle](https://www.kaggle.com/dansbecker/food-101)
- **Number of Classes**: 101
- **Number of Images**: 101,000

## Project Structure
- **Food101 CNN Classification.ipynb**: This Jupyter notebook contains the complete code for the project, including data preprocessing, model creation, training, evaluation, and predictions.

## Model Architecture
The model is built using a Convolutional Neural Network (CNN). The architecture includes the following layers:

- **Input Layer**: Preprocessed images from the Food101 dataset.
- **Convolutional Layers**: Several layers with ReLU activations and MaxPooling.
- **Dense Layers**: Fully connected layers with dropout for regularization.
- **Output Layer**: Softmax layer with 101 outputs corresponding to the food categories.

## Training
The model is trained using the following configuration:

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: [Specify the number of epochs]
- **Batch Size**: [Specify the batch size]

## Evaluation
The model's performance is evaluated using accuracy and loss metrics on the test set. The notebook includes visualizations of the training process and confusion matrix to show the classification performance across different categories.

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/food101-cnn-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd food101-cnn-classification
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter Notebook:
    ```bash
    jupyter notebook Food101\ CNN\ Classification.ipynb
    ```

## Results
The final model achieved an accuracy of [Insert accuracy here] on the test dataset. Below are sample predictions made by the model:

- **Image 1**: Predicted - [Class], Actual - [Class]
- **Image 2**: Predicted - [Class], Actual - [Class]

## Conclusion
This project demonstrates the application of CNNs in classifying food images into 101 categories. The model performs well on the test set, showing the effectiveness of deep learning techniques in image classification tasks.

## Future Work
- **Data Augmentation**: Implementing more advanced data augmentation techniques to improve model robustness.
- **Transfer Learning**: Experimenting with pre-trained models to improve accuracy.
- **Hyperparameter Tuning**: Fine-tuning model hyperparameters for better performance.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
