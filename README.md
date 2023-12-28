
## Automated Blast Cell Detection For Acute Lymphoblastic Leukemia Using a Stacking Ensemble of Convolutional Neural Networks
Wayne Matthew A. Dayata, Sabrinah Yonell C. Yap, Christine D. Bandalan  |  November 2023

## About the project:
Acute Lymphoblastic Leukemia (ALL), caused by the continuous multiplication of malignant overproduction and immature white blood cells (WBCs), is one of the most common types of leukemia among children and adults with a high mortality rate mainly because of its late detection and diagnosis. This thereby drives the need to create systems that aid pathologists in the morphological analysis and detection of ALL blast cells, reducing error rates and increasing the likelihood of survival among ALL patients as a result. Five Convolutional Neural Network (CNN) models: ConvNeXtTiny, MobileNetV2, EfficientNetV2B3, InceptionV3, and DenseNet-121 were integrated into a stacking ensemble to distinguish ALL cells from healthy cells accurately. The results show that the proposed ensemble was able to better support clinical decisions to detect ALL in patients than the individual CNN models do.

### Dataset:
ALL Challenge dataset of ISBI 2019 (C-NMC 2019) | The Cancer Imaging Archive
Download here: https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758223

### Experiments and results:
- Experiments for the development of the CNN models and stacking ensemble configurations are found in the `Experiments` folder.
- Results for the experiments are placed in the `Results` folder showing the test set prediction results and summary statistics from each CNN model and the meta model of the proposed stacking ensemble. 

### Model files:
Download here: https://tinyurl.com/ALLClassifierModels

### Set-up instructions:
1. Open command prompt and navigate to the repository.
2. Ensure the following libraries are present: `numpy`, `tk` (Tkinter), `Pillow`, `tensorflow`, and `keras`. 
	- If any of them are missing, type `pip install <package_name>`.
3. Install Pyinstaller.
	- `pip install pyinstaller`
5. Create the executable file for the application.
	- `pyinstaller --onefile ALLClassifier.py`.
	- The `--onefile` flag creates a single file bundled executable.
6. Ensure the [model files](#model-files) are downloaded and placed inside `Models/` subdirectory.
7. Run the `ALLClassifier.exe` application. 
	 - Refer to the user manual for the instructions in using the application. 
	 - Test the application using the [dataset](#dataset) provided above.
