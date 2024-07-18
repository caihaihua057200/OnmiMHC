## Introduction
OnmiMHC integrates large-scale mass spectrometry data with other relevant data types to achieve superior performance in MHC-I and MHC-II prediction tasks. By combining 1D-CNN-LSTM and 2D-CNN models, OnmiMHC captures both temporal and spatial features of sequences, enhancing the accuracy of peptide-MHC binding predictions.

## Requirements
Ensure the installation of the following dependencies:
- pandas version: 1.3.5
- numpy version: 1.21.6
- torch version: 1.13.1+cu116
- matplotlib version: 3.5.3

## Usage
First, download the model weights from [this link](https://drive.google.com/drive/folders/13NZmHObr3VvkZD59yxFaWjxFVe_wj6ID?usp=sharing). This link contains the model weights for both MHC-I and MHC-II, and additionally, the candidate peptide pool file for UCEC for MHC-I.

For MHC-I tasks:
- Place the `MHC-I` weights folder into the `MHC-I` directory from the GitHub repository.
- Run the following command:
  ```bash
  python OnmiMHC-I.py IEDB.csv ./test/IEDB.csv
  ```
## Model Architecture
OnmiMHC employs two encoding methods: BLOSUM62 and one-hot encoding. The architecture integrates 1D-CNN-LSTM and 2D-CNN models to extract both temporal and spatial features from the sequences. Additionally, the CBAM attention mechanism is applied to enhance feature representation.

## Key Components
- 1D-CNN-LSTM: Captures temporal information and local features.
- 2D-CNN: Extracts planar local features by rearranging sequences into a 2D matrix.
- CBAM: Enhances feature representation through channel and spatial attention mechanisms.

## Contributing
We welcome contributions to OnmiMHC. If you have any suggestions or improvements, please open an issue or submit a pull request.
