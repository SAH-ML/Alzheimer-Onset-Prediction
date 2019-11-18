import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, cross_val_predict
from vecstack import stacking, StackingTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, f1_score, precision_score, roc_auc_score, roc_curve
import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import pickle

def my_metric(y_true, y_preds):
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    
    return [accuracy_score(y_true, y_preds), tn/(tn+fp), tp/(tp+fn)]


# Read the data into X and Y
fp = open("E:\Dr Asif and Tabrej Khan\Alz Train (Final)_orig.csv")
header = [i.strip() for i in fp.readline().split(',')]
print(header)
lines = fp.readlines()
print(lines)
X, Y = list(), list()
print(X)
print(Y)
for line in lines:
    line = [float(i.strip()) for i in line.split(',')]
    X.append(line[:-1])
    Y.append(int(line[-1]))

X, Y = np.array(X), np.array(Y)


# Create training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, test_size=0.1)
X, Y = X_train, Y_train

# Perform Feature ranking with recursive feature elimination (5 features only)
selector = RFE(LinearSVC(random_state=0, class_weight='balanced', tol=1.0, max_iter=1000), 5, step=3)
X, X_test = selector.fit_transform(X, Y), selector.transform(X_test)
print(X.shape)
print(selector)
print(X_test)
# Features Selected by above step
#['Clusterin_Apo_J', 'Cystatin_C', 'FAS', 'NrCAM', 'tau']

# Perform Stacking
models = [('xgb', XGBClassifier(random_state=0)), ('svc', LinearSVC(random_state=0, class_weight='balanced', tol=1.0, max_iter=1000))]
stack = StackingTransformer(models, regression=False, verbose=0)
stack = stack.fit(X, Y)
pickle.dump(stack, open('stacker.pkl','wb'))
X, X_test = np.concatenate((X, stack.transform(X)), axis=1), np.concatenate((X_test, stack.transform(X_test)), axis=1)

# Let's test the effect of different threshold values and record it
# in order to take the best threshold valuess
threshold, final_scores = [-0.8], list()
for i in range(21):
    scores, roc_auc = list(), 0
    kfolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for train, test in kfolds.split(X, Y):
        x_train, x_test = X[train], X[test]
        y_train, y_test = Y[train], Y[test]

        model = LinearSVC(random_state=0, tol=1.0, class_weight='balanced')
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        preds_proba = model.decision_function(x_test)
        preds = [1 if i>=threshold[-1] else 0 for i in preds_proba]
        roc_auc+= roc_auc_score(y_test, preds_proba)
        scores.append(my_metric(y_test, preds))
    scores, roc_auc = [i/len(scores) for i in np.sum(scores, axis=0)], roc_auc/len(scores)
    final_scores.append(scores+[roc_auc])
    print("threshold: "+str(threshold[-1]))
    threshold.append(threshold[-1]+0.05)

# Set the best threshold value here
best_thresh = 0.0

print("\n\nTEST:\n-----")
# Check out our score on the test set we split earlier
model = LinearSVC(random_state=0, tol=1.0, class_weight='balanced')#
model.fit(X, Y)
pickle.dump(model, open('model.pkl','wb'))
print("Accuracy, Specificity, Sensitivity: ", my_metric(Y_test, model.predict(X_test)))
preds_proba = model.decision_function(X_test)
print("ROC AUC: ", roc_auc_score(Y_test, preds_proba))
