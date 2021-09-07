Convolutional regression network to associate CT images with body weight. Includes trained model based on EfficientNetB2 architecture adapted for regression output. Requires python libraries Tensorflow (tested on v2.5), SimpleITK, pandas, numpy, scipy, pydicom, & matplotlib. All should be able to be installed via pip.

Usage: Place input dicom files into 'input' directory. It should be able to parse multiple scans as a batch based on series UID and handle nested subdirectories. The batch output will be written to 'predicted_weights.csv' which includes one row per series with patient name, weight, date, and series uid. Additionally, screenshots of the pseudo-scout images used for prediction, predicted weight, and overlaid saliency map will be output to the 'prediction_screens' folder with one image per series.

Execute with:
python predict_weights.py
