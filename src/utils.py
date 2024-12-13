import logging as log

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import cross_val_score

kfold = 2

# %%
def evaluation(model,X,Y,kfold):
    # Cross Validation to test and anticipate overfitting problem
    scores1 = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    scores2 = cross_val_score(model, X, Y, cv=kfold, scoring='precision')
    scores3 = cross_val_score(model, X, Y, cv=kfold, scoring='recall')
    # The mean score and standard deviation of the score estimate
    log.info("Cross Validation Accuracy: %0.5f (+/- %0.2f)" % (scores1.mean(), scores1.std()))
    log.info("Cross Validation Precision: %0.5f (+/- %0.2f)" % (scores2.mean(), scores2.std()))
    log.info("Cross Validation Recall: %0.5f (+/- %0.2f)" % (scores3.mean(), scores3.std()))
    return  


# %%
def compute_roc(Y, y_pred, plot=True):
    fpr = dict()
    tpr = dict()
    auc_score = dict()
    fpr, tpr, _ = roc_curve(Y, y_pred)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
        plt.legend(loc="upper right")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.title("ROC Curve")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.show()
    return fpr, tpr, auc_score


# %%
def feature_importance(model,features,selection=False) : 
    feature_importances = pd.DataFrame(model.feature_importances_  )
    feature_importances = feature_importances.T
    feature_importances.columns = [features]
    
    sns.set(rc={'figure.figsize':(13,12)})
    fig = sns.barplot(data=feature_importances, orient='h', order=feature_importances.mean().sort_values(ascending=False).index)
    fig.set(title = 'Feature importance', xlabel = 'features', ylabel = 'features_importance' )
    
    if selection: #Selection of features with min 2% of feature importance
        n_features = feature_importances[feature_importances.loc[:,] > 0.02].dropna(axis='columns')
        n_features = n_features.columns.get_level_values(0)    
        log.info("Selected features")
        log.info(n_features)
        
    return fig


# %%
def model_fit(model,X,Y,features, performCV=True,roc=False, printFeatureImportance=False):
    
    #Fitting the model on the data_set
    model.fit(X[features],Y)
        
    #Predict training set:
    predictions = model.predict(X[features])
    predprob = model.predict_proba(X[features])[:,1]
    
    # Create and print confusion matrix    
    cfm = confusion_matrix(Y,predictions)
    log.info("\nModel Confusion matrix")
    log.info(cfm)
    
    #Print model report:
    log.info("\nModel Report")
    log.info("Accuracy : %.4g" % accuracy_score(Y.values, predictions))
    
    #Perform cross-validation: evaluate using 10-fold cross validation 
    #kfold = StratifiedKFold(n_splits=10, shuffle=True)
    if performCV:
        evaluation(model,X[features],Y,kfold)
    if roc: 
        compute_roc(Y, predictions, plot=True)
          
    #Print Feature Importance:
    if printFeatureImportance:
        feature_importance(model,features,selection=False) 
        

