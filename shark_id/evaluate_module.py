import numpy as np #.argmax
import pandas as pd
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, balanced_accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#labels for dataframe, classification report and confusion matrix
labels=['Basking', 'Blue', 'Hammerhead', 'Mako', 'Sand Tiger', 'Tiger', 'White', 'Blacktip', 'Bull', 'Lemon', 'Nurse', 'Thresher', 'Whale', 'Whitetip']

def create_predictions(model, X_test, y_test):
    #create y_pred predictions and y_text_argmax for metrics
    predictions = model.predict(X_test)

    y_pred = []
    y_test_argmax = []

    for pred in predictions:
        y_pred.append(np.argmax(pred))

    for test in y_test:
        y_test_argmax.append(np.argmax(test))

    return y_pred, y_test_argmax


def model_scoring_metrics(y_test_argmax, y_pred):
    #macro f1
    f1 = f1_score(y_test_argmax, y_pred, average='macro')

    #weighted f1
    f1_score_weighted = f1_score(y_test_argmax, y_pred, average='weighted')

    #macro precision
    precision = precision_score(y_test_argmax, y_pred, average='macro')

    #weighted precision
    precision_score_weighted = precision_score(y_test_argmax, y_pred, average='weighted')

    #macro recall
    recall = recall_score(y_test_argmax,y_pred, average='macro')

    #weighted recall
    recall_weighted = recall_score(y_test_argmax,y_pred, average='weighted')

    #macro accuracy
    accuracy = accuracy_score(y_test_argmax, y_pred)

    #weighted accuracy
    accuracy_bal = balanced_accuracy_score(y_test_argmax, y_pred)


    print(f"Model Scoring Metrics: \n F1 score: {round(f1,4)} \n Weighted F1 score: {round(f1_score_weighted,4)} \n Precision: {round(precision,4)} \n Weighted precision: {round(precision_score_weighted,4)} \n Recall: {round(recall,4)} \n Weighted recall: {round(recall_weighted,4)} \n Accuracy: {round(accuracy,4)} \n Weighted accuracy: {round(accuracy_bal,4)}")

def create_class_reports(y_test_argmax, y_pred):

    #classification report - separate accuracy, average and weighted average separate
    report = classification_report(y_test_argmax, y_pred, target_names=labels, output_dict=True)

    print(f"Scoring Metrics (Unsorted) \n {report}")

    #classification report - sorted by f1 score
    report_sorted_f1 = classification_report(y_test_argmax, y_pred, target_names=labels, output_dict=True)

    #Convert the report to a pandas DataFrame and move model accuracy to bottom of df
    accuracy = round(report_sorted_f1['accuracy'],4)
    del report_sorted_f1['accuracy']
    df = pd.DataFrame(report_sorted_f1).transpose()
    df = df.sort_values(by='f1-score', ascending=False)
    df.reset_index(inplace=True)
    print(f"Scoring Metrics Sorted by F1 Score \n {df} \n Model Accuracy: {round(accuracy, 4)*100}%")

def create_confusion_matrix(y_test_argmax, y_pred):
    cm = confusion_matrix(y_test_argmax, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(ax=ax)
    ax.set_xticklabels(labels, rotation=45)

    plt.show()
