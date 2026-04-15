# phagemorph: Classifying phage TEM images based on morphotype

`phagemorph` classifies bacteriophage transmission electron microscopy (TEM)
images by morphotype. The notebooks in this repository preprocess TEM figures,
train a small PyTorch convolutional neural network, evaluate one-vs-rest ROC
curves, and provide a helper for predicting the morphotype of a new image.

The current classifier targets three morphotype labels:

- `sipho`
- `podo`
- `myo`

## Repository Contents

| Path | Purpose |
| --- | --- |
| `morphotype_classifier.ipynb` | Main notebook for cropped Phagebase images. Includes preprocessing, model training, ROC evaluation, prediction, and layer/filter visualisation. |
| `early_stopping_classifier.ipynb` | Variant of the main workflow with a train/validation/test split and early stopping. |
| `all_images_classifier.ipynb` | Comparison notebook to show how the model performs on raw data. |
| `environment.yaml` | Conda environment for running the notebooks. |

## Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yaml
conda activate phagemorph
```

Start JupyterLab:

```bash
jupyter lab
```

The environment includes Python 3.10, PyTorch, torchvision, scikit-learn,
pandas, NumPy, Pillow, OpenCV, Matplotlib, seaborn, tqdm, and Jupyter.

## Data

The notebooks expect the source datasets and related metadata to exist locally. You can download example data from [PhageBase](https://www.phagebase.com/). Update the path cells
near the top of each notebook if your data is stored somewhere else.

During preprocessing, images are converted to grayscale, padded to a square
without distortion, resized to `265 x 265`, and written into class-specific
folders under the relevant project directory.

## Workflow

1. Load and standardise the metadata.
2. Index source images and resolve labels from metadata.
3. Preprocess each image into a square grayscale `265 x 265` PNG.
4. Train a CNN on `sipho`, `podo`, and `myo` labels.
5. Evaluate the model with a classification report, confusion matrix, and
   one-vs-rest ROC curves.
6. Save the trained checkpoint as `tem_morphotype_cnn.pt`.
7. Use the notebook prediction helper on new TEM images.

The model is a compact PyTorch CNN with four convolutional blocks, batch
normalisation, max pooling, dropout, and a dense classifier. Training uses a
fixed random seed (`42`) and light image augmentation.

## Predicting a New Image

After running a training notebook, use its helper function from a notebook cell:

```python
pred_label, probabilities = predict_morphotype("/path/to/your_tem_image.png")

print(pred_label)
probabilities
```

In `morphotype_classifier.ipynb`, the helper also supports explanation plots:

```python
pred_label, probabilities = predict_morphotype(
    "/path/to/your_tem_image.png",
    show_explanations=True,
)
```

## Outputs

The notebooks generate:

- processed image folders in `processed_265/`
- trained model checkpoints named `tem_morphotype_cnn.pt`
- training accuracy/loss plots
- classification reports and confusion matrices
- one-vs-rest ROC curves
- optional learned-filter and layer-activation visualisations

## Notes

- TEM images are treated as grayscale intensity images.
- Padding happens before resizing so rectangular figures are not stretched.
- Performance can vary between runs because the dataset is small.
- The notebooks prefer metadata-derived labels and include filename-normalisation
  fallbacks for duplicate-like suffixes such as `_2` or `copy`.
- Useful next improvements include cross-validation, class-balanced sampling,
  stronger augmentation, and transfer learning.

