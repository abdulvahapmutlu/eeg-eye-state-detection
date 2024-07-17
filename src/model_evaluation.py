import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Load preprocessed data
X_test = joblib.load('X_test.pkl')
y_test = joblib.load('y_test.pkl')

# Load models
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
random_forest_model = joblib.load('random_forest_model.pkl')
svm_model = joblib.load('svm_model.pkl')
gradient_boosting_model = joblib.load('gradient_boosting_model.pkl')

models = {
    'Logistic Regression': logistic_regression_model,
    'Random Forest': random_forest_model,
    'SVM': svm_model,
    'Gradient Boosting': gradient_boosting_model
}

# Evaluate models
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f'Accuracy for {model_name}: {accuracy:.2f}')
    print(f'Confusion Matrix for {model_name}:\n{conf_matrix}')
    print(f'Classification Report for {model_name}:\n{class_report}')
    
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'../results/{model_name.replace(" ", "_").lower()}_results.png')
    plt.show()
