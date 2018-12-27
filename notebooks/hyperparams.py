space ={
    'max_depth':     hp.randint('best_max_depth', 10),    
    'n_estimators':  hp.randint('best_n_estimators', 500),
    'learning_rate': hp.uniform('best_learning_rate', 0.05, 0.3),
    'log_transform': hp.choice('best_log_transform', [True, False]),
}

feats = history_lags + pred_future_linear_lags
targets = future_lags

def build_objective_func(train, test, validation, vertical_validation, get_hyperparams):
    def objective_func(space):
        hyperparams = get_hyperparams(space)
        xgb_params = { 'max_depth': 5 + hyperparams[0],
                      'n_estimators': 40 + hyperparams[1],
                      'learning_rate': hyperparams[2],
                      'colsample_bytree': .7,
                      'subsample': 0.7,
                      'random_state ': 2013,}
        
        log_transform = hyperparams[3]        
        local_train = train.copy()
        local_test = test.copy()
        local_validation = validation.copy()
        local_vertical_validation = vertical_validation.copy()
        models = h.build_prediction_models(local_train, local_test, feats, targets, xgb_params, log_transform)
        for (index,target) in enumerate(targets):
            model = models[target]
            model_feats = feats
            #model_feats = feats[0:-3*len(targets)+3*index]
            
            if log_transform:                
                y_log_pred = model.predict(local_validation[feats].values)
                y_pred = np.exp(y_log_pred) - 1
            else:
                y_pred = model.predict(local_validation[model_feats].values)
                
            y_pred[y_pred<0]=0
            local_validation['pred_' + target] = y_pred
        h.decompose_prediction_to_vertical(local_validation, local_vertical_validation, features_desc, add_real_future_values=True)    
            
        #if mean is None:
        #    print("Hyperparams: {} FAIL".format(hyperparams, mean, std))
        #    return {'loss':0, 'status': STATUS_FAIL  }
        #print("Hyperparams: {} mean: {} std: {}".format(hyperparams, mean, std))
        scores = []
        for i in range(0,101,5):
            weighted_prediction = (i* local_vertical_validation.gradient_xgb_r_8_last + (100-i)*local_vertical_validation.dummy_last)/100
            scores.append(rmsle(local_vertical_validation.real_values.values, weighted_prediction))
        print("Hyperparams: {} score: {}".format(hyperparams, np.min(scores)))
        return {'loss':np.min(scores), 'status': STATUS_OK } #mo¿e wariancja lepsza?
    
    return objective_func

warnings.filterwarnings('ignore')
objective = build_objective_func(                                 
                                 df_xgb_train,
                                 df_xgb_test,
                                 df_ts_featurized_validation,
                                 proper_validation,
                                 lambda i_space:(i_space['max_depth'],                                                 
                                                 i_space['n_estimators'],
                                                 i_space['learning_rate'],
                                                 i_space['log_transform']))
trials = Trials()
best_params_sarima = fmin(fn=objective,
                    space=space,
                    algo=partial(tpe.suggest, n_startup_jobs=3),
                    max_evals=100, #iloœæ prób
                    trials=trials)

warnings.filterwarnings('default')

best_params = (best_params_sarima['best_max_depth'],
              best_params_sarima['best_n_estimators'],
              best_params_sarima['best_learning_rate'],
              best_params_sarima['best_log_transform']
             )

print(best_params)