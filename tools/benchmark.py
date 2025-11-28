def benchmark(X_train, X_test, Y_train, Y_test, search='random', ml='rf', criterion='gini', nb_method='Bernoulli', kernel='liner', scoring='accuracy', average='weighted', n_iter=10, cv=5):
    '''Use RandomSearchCV() or GridSearchCV() to find the best parameters for a given ML classifier
    (Random Forest, Naive Bayesian or Support Vector Machines)'''
    
    import numpy as np
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    from scipy import stats

    # Set the the scoring function
    if scoring == 'precision':
        scorer = make_scorer(precision_score, average=average)
    elif scoring == 'recall':
        scorer = make_scorer(recall_score, average=average)
    elif scoring == 'f1':
        scorer = make_scorer(f1_score, average=average)
    else:
        scorer = make_scorer(accuracy_score)
    
    # Generate the model and define the hyperparmeters to optimise
    if 'rf' in ml.lower():
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(criterion=criterion)
        if search == 'random':
            param = {'n_estimators': stats.randint(25,500), 'max_depth': stats.randint(1,10), 'min_samples_split': stats.randint(2,10), 'min_samples_leaf': stats.randint(1,10)}
        else:
            param = {'n_estimators': [25, 50, 100, 200, 400], 'max_depth': np.arange(1,10,2), 'min_samples_split': np.arange(2,11,2), 'min_samples_leaf': np.arange(1,11,2)}
    if 'nb' in ml.lower():
        if nb_method.lower() == 'bernoulli':
            from sklearn.naive_bayes import BernoulliNB
            model = BernoulliNB()
        elif nb_method.lower() == 'categorical':
            from sklearn.naive_bayes import CategoricalNB
            model = CategoricalNB()
        elif nb_method.lower() == 'complement':
            from sklearn.naive_bayes import ComplementNB
            model = ComplementNB()
        elif nb_method.lower() == 'gaussian':
            from sklearn.naive_bayes import GaussianNB
            model = GaussianNB()
        else:
            return print('ERROR! NB method incorrect or not defined!')
        if search == 'random':
            if nb_method.lower() != 'gaussian':
                param = {'alpha': stats.uniform(1e-10,1)}
            else:
                param = {'var_smoothing': stats.uniform(1e-10,1)}
        else:
            if nb_method.lower() != 'gaussian':
                param = {'alpha': np.logspace(-10,1,6)}
            else:
                param = {'var_smoothing': np.logspace(-10,1,6)}
    if 'svm' in ml.lower():
        from sklearn.svm import SVC
        model = SVC(kernel=kernel)
        if search == 'random':
            if kernel != 'poly':
                param = {'C': stats.uniform(0.01,1000), 'gamma': stats.uniform(0.0001,1)}
            else:
                param = {'C': stats.uniform(0.01,1000), 'gamma': stats.uniform(0.0001,1), 'degree': stats.randint(2,5)}
        else:
            if kernel != 'poly':
                param = {'C': np.logspace(-2,3,5), 'gamma': np.logspace(-4,0,5)}
            else:
                param = {'C': np.logspace(-2,3,5), 'gamma': np.logspace(-4,0,5), 'degree': np.arange(2,6,1)}
    
    if search == 'grid':
        model_search = GridSearchCV(model, param_grid = param, cv=cv, scoring=scorer)
    else:
        model_search = RandomizedSearchCV(model, param_distributions = param, n_iter=n_iter, cv=cv, scoring=scorer)
    model_search.fit(X_train, Y_train)
    
    # Create a variable for the best model
    best_model = model_search.best_estimator_
        
    # Generate predictions with the best model
    Y_pred = best_model.predict(X_test)

    accu = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred, average=average)
    rec = recall_score(Y_test, Y_pred, average=average)
    f1 = f1_score(Y_test, Y_pred, average=average)
    
    return accu, prec, rec, f1, model_search.best_params_