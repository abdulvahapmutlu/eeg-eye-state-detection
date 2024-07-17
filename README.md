# EEG Eye State Detection

This project aims to classify eye states (open or closed) based on EEG signals. Multiple machine learning models are trained and evaluated to achieve high accuracy in predicting the eye state.

## Dataset

The dataset used in this project is an EEG Eye State dataset, which contains EEG signals recorded from a subject. The target variable is `eyeDetection`, indicating whether the eyes are open (0) or closed (1).
Dataset Link : Dataset link: [EEG Eye State](https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State)

Adding the Dataset
Since the dataset is located at 'C:/Users/offic/OneDrive/Masaüstü/datasets/eeg_eye/eeg_eye.csv', make sure to place it in the data/ directory of your repository as eeg_eye.csv.

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/eeg-eye-state-detection.git
   cd eeg-eye-state-detection
   
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   
3. Install the required packages:
   ```
   pip install -r requirements.txt

4. Run the Jupyter notebook to execute the code and see the analysis:   
   ```
   jupyter notebook notebooks/EEG_Eye_State_Detection.ipynb

## Models and Hyperparameters

The following machine learning models were used:
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting

Hyperparameter tuning was performed using GridSearchCV. The best parameters for each model are reported below:

- **Logistic Regression**:
  - Best Parameters: `{'C': 100, 'penalty': 'l2', 'solver': 'liblinear'}`
  - Accuracy: 0.61

- **Random Forest**:
  - Best Parameters: `{'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}`
  - Accuracy: 0.93

- **SVM**:
  - Best Parameters: `{'C': 100, 'gamma': 1, 'kernel': 'rbf'}`
  - Accuracy: 0.87

- **Gradient Boosting**:
  - Best Parameters: `{'learning_rate': 0.3, 'max_depth': 6, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 500}`
  - Accuracy: 0.95
 
## Results

Confusion matrices for each model are saved in the results/ folder.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
