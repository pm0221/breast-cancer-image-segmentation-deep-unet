# Breast Cancer Image Segmentation Using Deep U-Net

## Project Overview
This project uses a Deep U-Net-style convolutional neural network to segment regions in breast ultrasound images. The model is trained on breast ultrasound images and their corresponding masks from the BUSI (Breast Ultrasound Images) dataset.

The implementation is provided in the Jupyter Notebook `breast_cancer_image_segmentation_deep_unet.ipynb`.

## Objectives
- Load breast ultrasound images and their segmentation masks.
- Resize and normalize images and masks for model input.
- Split the dataset into training, validation, and testing sets.
- Build and train a Deep U-Net-style segmentation model.
- Visualize training performance and compare predicted masks with ground-truth masks.
- Evaluate segmentation predictions using IoU, precision, recall, F1 score, and Dice coefficient.

## Dataset
The notebook downloads the **Breast Ultrasound Images Dataset (BUSI)** using `kagglehub`:

```python
kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
```

It expects the downloaded data to contain the folder `Dataset_BUSI_with_GT`, with three class folders:
- `benign`
- `malignant`
- `normal`

The preprocessing function loads PNG images and files ending in `_mask.png`. It excludes additional mask variants ending in `_mask_1.png` and `_mask_2.png`.

## Project Workflow

### 1. Image and Mask Preprocessing
- Load ultrasound images and grayscale masks from the dataset folders.
- Resize images and masks to **256 × 256** pixels.
- Convert images from BGR to RGB.
- Normalize pixel values by dividing by 255.
- Convert the processed data into NumPy arrays.

### 2. Dataset Combination and Splitting
Images and masks from the benign, malignant, and normal folders are combined. The notebook shuffles image-mask pairs together and splits them into:
- **Training:** 90%
- **Validation:** 7.5%
- **Testing:** remaining 2.5%

### 3. Visualize Image-Mask Pairings
The notebook displays sample benign ultrasound images alongside their corresponding ground-truth masks.

### 4. Deep U-Net Model Architecture
The model is built using Keras functional API and contains encoder and decoder stages.

**Encoder (downsampling):**
- Convolution layers with 64, 128, 256, 512, 1024, and 2048 filters.
- Batch normalization and ReLU activation.
- Max pooling for spatial downsampling.
- Dropout with a rate of 0.15 at selected stages.

**Decoder (upsampling):**
- Convolution, batch normalization, and ReLU activation.
- Upsampling by a factor of 2.
- Filter sizes of 1024, 512, 256, 128, 64, and 32.
- Dropout with a rate of 0.15 at selected stages.

The final layer is a 1-filter convolution with sigmoid activation, producing a single-channel segmentation mask.

### 5. Training Configuration
- **Loss function:** Dice loss
- **Optimizer:** Adam
- **Metric during training:** Accuracy
- **Epochs:** Up to 50
- **Batch size:** 16
- **Callbacks:** Early stopping and learning-rate reduction on validation-loss plateau
- **Random seed:** TensorFlow seed set to 1

Early stopping monitors validation loss with a patience of 20 and restores the best weights. The learning-rate callback reduces the learning rate by a factor of 0.1 after 10 epochs without validation-loss improvement.

### 6. Performance Visualization
The notebook plots validation accuracy and validation loss against the training epochs.

### 7. Prediction Visualization
For sample test images, the notebook displays:
- Original ultrasound image
- Ground-truth mask
- Predicted segmentation mask

### 8. Evaluation Metrics
The notebook thresholds predicted probabilities at 0.5 to create binary masks, then calculates:
- **Mean Intersection over Union (Mean IoU)**
- **Precision**
- **Recall**
- **F1 score**
- **Dice coefficient**

The actual metric values are printed when the notebook is executed; they are not specified in this README.

## Technologies and Libraries
- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- OpenCV (`cv2`)
- Matplotlib
- ImageIO
- KaggleHub

## Installation

Install the required packages:

```bash
pip install tensorflow numpy pandas opencv-python matplotlib imageio kagglehub
```

The notebook also uses Keras, which is commonly installed with TensorFlow. If your environment requires it separately, install it with:

```bash
pip install keras
```

## How to Run
1. Open the project in an environment that supports Jupyter Notebook, such as JupyterLab or Google Colab.
2. Install the required dependencies.
3. Ensure KaggleHub can download the BUSI dataset in your environment.
4. Open `breast_cancer_image_segmentation_deep_unet.ipynb`.
5. Run all cells in order.
6. Review the training plots, predicted masks, and printed evaluation metrics.

## Project Structure

```text
Breast_Cancer_Image_Segmentation/
├── breast_cancer_image_segmentation_deep_unet.ipynb
└── README.md
```

The dataset is downloaded by the notebook and is not included in this project folder by default.

## Notes and Limitations
- The notebook uses one selected mask per image and does not load the `_mask_1.png` or `_mask_2.png` variants.
- The split is randomized; the notebook does not set a NumPy random seed before shuffling.
- The notebook's model is a Deep U-Net-style encoder-decoder, but it does not include skip connections between corresponding encoder and decoder stages.
- The notebook's final metric values depend on the downloaded dataset and training run.
- The notebook's own notes identify a possible weakness in performance on images without cancer and suggest adding more non-cancerous examples/masks and tuning segmentation thresholds in future work.
- This project performs image segmentation; it does not provide a clinical diagnosis.

## Future Scope
- Add more non-cancerous images and corresponding masks.
- Investigate a lower segmentation threshold or other threshold-selection strategies.
- Tune the model architecture and training parameters.
- Reassess performance on normal images and evaluate generalization on additional data.

## Author
Add your name and academic details here.
