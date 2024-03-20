# README

## Overview
This code is designed for building and evaluating a Random Forest Classifier model for intrusion detection using the KDD Cup 1999 dataset. The dataset contains a large number of network connection records, including both normal and attack connections. The model aims to classify these connections into different attack types.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn
- numpy

## Usage
1. Ensure all the required libraries are installed. You can install them using pip:
   ```
   pip install pandas scikit-learn matplotlib seaborn numpy
   ```

2. Clone or download this repository to your local machine.

3. Run the script `intrusion_detection.py`. This script contains the code for building and evaluating the Random Forest Classifier model.

4. Upon execution, the script will perform the following steps:
   - Load the KDD Cup 1999 dataset.
   - Prepare numerical features for training the model.
   - Split the dataset into training and testing sets.
   - Train a Random Forest Classifier model on the training data.
   - Evaluate the model's performance using accuracy score and classification report.
   - Visualize the class distribution of attack types in the dataset.
   - Visualize the confusion matrix to understand the model's performance further.

## Files
- `main.py`: Contains the main code for building and evaluating the Random Forest Classifier model.
- `KDDTrain+.txt`: Dataset file containing network connection records.

## Notes
- This code assumes that the dataset file `KDDTrain+.txt` is located in the same directory as the script. If the dataset is stored elsewhere, please provide the correct path while loading the dataset.
- Ensure that the dataset file is in the correct format compatible with the code. Adjustments may be necessary if the dataset format differs.
- Feel free to modify the code to experiment with different machine learning models, parameters, or datasets for intrusion detection tasks.

## Acknowledgments
- This code is inspired by the KDD Cup 1999 dataset and the need for effective intrusion detection mechanisms in network security.
