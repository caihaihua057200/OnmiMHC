## Introduction
OnmiMHC integrates large-scale mass spectrometry data with other relevant data types to achieve superior performance in MHC-I and MHC-II prediction tasks. By combining 1D-CNN-LSTM and 2D-CNN models, OnmiMHC captures both temporal and spatial features of sequences, enhancing the accuracy of peptide-MHC binding predictions.

## Requirements
Ensure the installation of the following dependencies:
- pandas version: 1.3.5
- numpy version: 1.21.6
- torch version: 1.13.1+cu116
- matplotlib version: 3.5.3

## Usage
To run predictions using the OnmiMHC model, you can use the provided scripts. For example:
- python predict.py --input your_input_file.csv --output your_output_file.csv

## Training the Model

## Model Architecture
OnmiMHC employs two encoding methods: BLOSUM62 and one-hot encoding. The architecture integrates 1D-CNN-LSTM and 2D-CNN models to extract both temporal and spatial features from the sequences. Additionally, the CBAM attention mechanism is applied to enhance feature representation.

## Key Components
- 1D-CNN-LSTM: Captures temporal information and local features.
- 2D-CNN: Extracts planar local features by rearranging sequences into a 2D matrix.
- CBAM: Enhances feature representation through channel and spatial attention mechanisms.

## Training Procedure
Training of OnmiMHC is divided into three steps:
- Pre-training: Train the OnmiMHC model using the BA dataset.
- Fine-tuning: Fine-tune the pre-trained model using the EL dataset.
- Final Training: Combine preprocessed datasets to train the final model.

## Pre-training
- For MHC-I tasks, we use five-fold BA datasets from NetMHCpan-4.1. 
- For MHC-II tasks, we use five-fold BA datasets from NetMHCIIpan-4.0. 

## Data Preprocessing and Label Generation
We preprocess MS ELs-SA and MS ELs-MA datasets using the pre-trained model to predict peptide-MHC binding scores, converting multi-allele data to single-allele data. 
This ensures high-quality data representation.

## Final Training
We preprocess the BA dataset by changing labels based on IC50 values and then train the final model using cross-entropy loss and backpropagation. The model is evaluated using 5-fold cross-validation.

Data Preprocessing
Data preprocessing involves:

## Removing peptides with missing values.
Eliminating records where the post-mutation peptide sequence remains unchanged.
Deduplicating the data.
These steps ensure the accuracy and reliability of the analysis.

## Contributing
We welcome contributions to OnmiMHC. If you have any suggestions or improvements, please open an issue or submit a pull request.
