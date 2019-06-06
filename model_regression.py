# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:51:32 2018

@author: twope
"""

from sklearn.model_selection import GroupKFold
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
y_target = train_set['feature_revenue']
valid_target = valid_pd['feature_revenue']

params = {
        'boosting_type': 'gbdt', 
        'objective': 'regression',
        'learning_rate': 0.01, 
        'num_leaves': 50, 
        'max_depth': 6,
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
         }
data_train = lgb.Dataset(train_set[train_features],y_target,silent=True)
cv_results = lgb.cv(params,data_train,
                    num_boost_round=1000, 
                    nfold=5, 
                    stratified=False, 
                    shuffle=True, 
                    metrics='rmse',
                    early_stopping_rounds=100, 
                    verbose_eval=50, 
                    show_stdv=True, seed=0)
len(cv_results['rmse-mean'])

#调参 max_depth 和 num_leaves
model_lgb = lgb.LGBMRegressor(objective = 'regression',
                              metric='rmse', 
                              learning_rate = 0.01,
                              bagging_fraction = 0.8,
                              feature_fraction = 0.8,
                              n_estimators = 400)
param_lr ={'max_depth': [6,7,8],
           'num_leaves':[50,100,150,200,250]}

gsearch1 = GridSearchCV(estimator=model_lgb, 
                        param_grid=param_lr, 
                        scoring='neg_mean_squared_error', 
                        cv=5, 
                        verbose=1, 
                        n_jobs=4)
gsearch1.fit(train_set[train_features],y_target)

[gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_]#8,50

#调整bagging_fraction he feature_fraction
params_frac = {'bagging_fraction':[0.6,0.7,0.8,0.9],
               'feature_fraction':[0.6,0.7,0.8,0.9]}

model_lgb2 = lgb.LGBMRegressor(objective = 'regression',
                               metric='rmse', 
                               learning_rate = 0.01,
                               max_depth = 8,
                               num_leaves = 50,
                               n_estimators = 387)

gsearch2 =GridSearchCV(estimator = model_lgb2,
                       param_grid = params_frac,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       verbose=1,
                       n_jobs=4)
gsearch2.fit(train_set[train_features],y_target)
[gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_]#0.6 0.6

params_reg={
            'reg_lambda': [0,0.01,0.05,0.3,0.5]#L2
           }
model_lgb3 = lgb.LGBMRegressor(objective = 'regression',
                               metric='rmse', 
                               learning_rate = 0.01,
                               max_depth = 8,
                               num_leaves = 50,
                               bagging_fraction = 0.6,
                               feature_fraction = 0.6,
                               n_estimators = 400)

gsearch3 =GridSearchCV(estimator = model_lgb3,
                       param_grid = params_reg,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       verbose=1,
                       n_jobs=4)
gsearch3.fit(train_set[train_features],y_target)
[gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_]#0.05


params = {
        'boosting_type': 'gbdt', 
        'objective': 'regression',
        'learning_rate': 0.01, 
        'num_leaves': 50, 
        'max_depth': 8,
        'bagging_fraction':0.6,     
        'feature_fraction':0.6,
         }
data_train = lgb.Dataset(train_set[train_features],y_target,silent=True)
cv_results = lgb.cv(params,data_train,
                    num_boost_round=1000, 
                    nfold=5, 
                    stratified=False, 
                    shuffle=True, 
                    metrics='rmse',
                    early_stopping_rounds=100, 
                    verbose_eval=50, 
                    show_stdv=True, seed=0)
len(cv_results['rmse-mean'])#310

params['n_estimators'] = 500
model_lgb_final = lgb.train(params,data_train,num_boost_round = 500)
pred_y = model_lgb_final.predict(valid_pd[train_features],num_iteration=310)
print('The rmse of prediction is:', mean_squared_error(valid_target,pred_y) ** 0.5)

lgb.plot_importance(model_lgb_final,figsize = (30,15))


'''''''''''''''''''''''''''''''''''''
pre_revenue = model_lgb_final.predict(test_df[train_features],num_iteration = 310)
len(pre_revenue)
test_df.info()
GA_predict_revenue = pd.DataFrame({'fullVisitorId':test_df['fullVisitorId'].values,
                                   'PredictedLogRevenue':pre_revenue})
GA_predict_revenue.to_csv('D:/DATA/python files/kaggle_kernel/google_revenue_prediction/predictions.csv',index = False)





