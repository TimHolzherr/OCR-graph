
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import logistic
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import convert_data
#sklearn.linear_model.LogisticRegression() # C for regularisation (small value = more regularisation)


def main():
    # Used to train the logistig model
    words = convert_data.read_PA3Data()
    letters = [l for word in words for l in word]
    X, Y = zip(*[(np.array(x.flat), l) for x,l in letters])
    # seprate into test and train
    Xtrain = np.array(X[:600])
    Ytrain = np.array(Y[:600])
    Xtest = X[600:]
    Ytest = Y[600:]

    #Simple logistic regression
    logreg = linear_model.LogisticRegression()
    logreg.fit(Xtrain , Ytrain)


    # Cross Validation to find Generalisation Error
    k_fold = cross_validation.KFold(n=len(Xtrain), n_folds=7)
    insample = []
    outofsample = []
    for train_ind, test_ind in k_fold:
        logreg = linear_model.LogisticRegression(C=0.1)
        logreg.fit(Xtrain[train_ind] , Ytrain[train_ind])
        outofsample.append(logreg.score(Xtrain[test_ind], Ytrain[test_ind]))
        insample.append(logreg.score(Xtrain[train_ind], Ytrain[train_ind]))
    print(insample)
    print(outofsample)
    print(sum(insample) / len(insample))
    print(sum(outofsample) / len(outofsample))



    # Grid Search Gross Validation
    logreg = linear_model.LogisticRegression()
    Cs = np.logspace(0, 2, 20)
    clf = GridSearchCV(estimator=logreg, param_grid=dict(C=Cs), cv=7)
    clf.fit(Xtrain, Ytrain)
    print(clf.best_estimator_.C)
    print(clf.best_score_)


if __name__ == '__main__':
    main()
