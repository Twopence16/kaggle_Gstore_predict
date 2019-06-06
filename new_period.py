# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:52:30 2018

@author: twope
"""

train_period_4 = train_df[(train_df['date']<=pd.datetime(2017,11,15)) & (train_df['date']>=pd.datetime(2017,6,1))] 
train_predict_preiod_4 = train_df[(train_df['date']<=pd.datetime(2017,12,31)) & (train_df['date']>=pd.datetime(2017,11,16))]

train_period_5 = train_df[(train_df['date']<=pd.datetime(2018,1,15)) & (train_df['date']>=pd.datetime(2017,8,1))] 
train_predict_preiod_5 = train_df[(train_df['date']<=pd.datetime(2018,2,28)) & (train_df['date']>=pd.datetime(2018,1,16))]

def future_revenue(df):
    period_future = df.groupby('fullVisitorId')['totals_transactionRevenue'].sum()
    period_future = period_future.reset_index()
    period_future.rename(columns = {'totals_transactionRevenue':'feature_revenue'},
                         inplace = True)
    period_future['feature_revenue'] = np.log1p(period_future['feature_revenue'])
    return period_future

period4_future_revenue = future_revenue(train_predict_preiod_4)
period5_future_revenue = future_revenue(train_predict_preiod_5)


def fill(train_period,target_period):
    
    train_period['totals_totalTransactionRevenue'] = train_period['totals_totalTransactionRevenue'].fillna(0).astype('float64')
    target_period['totals_totalTransactionRevenue'] =target_period['totals_totalTransactionRevenue'].fillna(0).astype('float64')
    train_period['totals_transactionRevenue'] = train_period['totals_transactionRevenue'].fillna(0).astype('float64')
    target_period['totals_transactionRevenue'] = target_period['totals_transactionRevenue'].fillna(0).astype('float64')
    revenue_train = train_period.groupby('fullVisitorId')['totals_transactionRevenue'].sum().values
    revenue_target = target_period.groupby('fullVisitorId')['totals_transactionRevenue'].sum().values
    target_pd = target_period.groupby('fullVisitorId').mean().reset_index()
    target_pd['totals_transactionRevenue'] = revenue_target
    train_pd = train_period.groupby('fullVisitorId').mean().reset_index()
    train_pd['totals_transactionRevenue'] = revenue_train
    #target_pd=target_period
    #Find the visitors those back puchased in future period
    train_visitors = train_pd['fullVisitorId'].unique()
    train_predict_visitors = target_pd['fullVisitorId'].unique()
    same_visitors = np.intersect1d(train_visitors, train_predict_visitors)
    back_user = target_pd[(target_pd['fullVisitorId'].isin(same_visitors)) & (target_pd['totals_transactionRevenue'] > 0)]
    back_user = back_user[['fullVisitorId','totals_transactionRevenue']]
    print(f'numbers of back users is {len(same_visitors)}')
    print('we have',len(back_user['fullVisitorId'].value_counts()),'visitors back to purchase at target periods')
    
    train_pd['classfication_target'] = train_pd['fullVisitorId'].map(lambda x: 1 if x in list(back_user['fullVisitorId']) else 0)
    train_pd['totals_totalTransactionRevenue'] = np.log1p(train_pd['totals_totalTransactionRevenue'])
    train_pd['totals_transactionRevenue'] = np.log1p(train_pd['totals_transactionRevenue'])
    print (train_pd.shape)
    return train_pd

train_pd_4= fill(train_period_4,train_predict_preiod_4)#对应 train_period_4
train_pd_5= fill(train_period_5,train_predict_preiod_5)#对应 train_period_5


#把符合长度的feature_revenue做出来
train_pd_4['feature_revenue'] = 0
train_pd_4 = pd.merge(train_pd_4,period4_future_revenue,
                      on = 'fullVisitorId',
                      how = 'left')
train_pd_4.info()
train_pd_4['feature_revenue_y'].fillna(0,inplace = True)
train_pd_4.drop('feature_revenue_x',axis = 1,inplace = True)
train_pd_4.rename(columns = {'feature_revenue_y':'feature_revenue'},
                  inplace = True)


train_pd_5['feature_revenue'] = 0
train_pd_5 = pd.merge(train_pd_5,period5_future_revenue,
                      on = 'fullVisitorId',
                      how = 'left')
train_pd_5.info()
train_pd_5['feature_revenue_y'].fillna(0,inplace = True)
train_pd_5.drop('feature_revenue_x',axis = 1,inplace = True)
train_pd_5.rename(columns = {'feature_revenue_y':'feature_revenue'},
                  inplace = True)

y_target2 = train_set2['feature_revenue']
valid_target = valid_pd['feature_revenue']

train_set2 = pd.concat([train_pd_4,train_pd_5], axis=0)
excluded_features = [
    'date','fullVisitorId', 'sessionId','classfication_target','feature_revenue',
    'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits','next_session_1','next_session_2',
     'day.1','day.10','day.11', 'day.12', 'day.13', 'day.14', 'day.15', 'day.16', 'day.17','day.18','day.19', 'day.2','day.20','day.21','day.22', 'day.23', 'day.24', 'day.25', 'day.26','day.27', 'day.28', 'day.29','day.3','day.30','day.31','day.4','day.5','day.6','day.7','day.8','day.9',
 'month.1','month.10','month.11','month.12','month.2','month.3','month.4','month.5','month.6',
 'month.7','month.8', 'month.9','weekday.0','weekday.1','weekday.2','weekday.3','weekday.4','weekday.5','weekday.6'
]
train_features = [_f for _f in train_set.columns if _f not in excluded_features ]

params = {
        'boosting_type': 'gbdt', 
        'objective': 'regression',
        'learning_rate': 0.01, 
        'num_leaves': 50, 
        'max_depth': 6,
        'subsample': 0.8, 
        'colsample_bytree': 0.8, 
         }
data_train = lgb.Dataset(train_set2[train_features],y_target2,silent=True)
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
                              n_estimators = 1000)
param_lr ={'max_depth': [6,7,8],
           'num_leaves':[50,100,150,200]}

gsearch1 = GridSearchCV(estimator=model_lgb, 
                        param_grid=param_lr, 
                        scoring='neg_mean_squared_error', 
                        cv=5, 
                        verbose=1, 
                        n_jobs=4)
gsearch1.fit(train_set2[train_features],y_target2)

[gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_]#8,50

#调整bagging_fraction he feature_fraction
params_frac = {'bagging_fraction':[0.6,0.7,0.8,0.9],
               'feature_fraction':[0.6,0.7,0.8,0.9]}

model_lgb2 = lgb.LGBMRegressor(objective = 'regression',
                               metric='rmse', 
                               learning_rate = 0.01,
                               max_depth = 8,
                               num_leaves = 50,
                               n_estimators = 1000)

gsearch2 =GridSearchCV(estimator = model_lgb2,
                       param_grid = params_frac,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       verbose=1,
                       n_jobs=4)
gsearch2.fit(train_set[train_features],y_target)
[gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_]#0.6 0.7

params_reg={
            'reg_lambda': [0,0.01,0.05,0.3,0.5]#L2
           }
model_lgb3 = lgb.LGBMRegressor(objective = 'regression',
                               metric='rmse', 
                               learning_rate = 0.01,
                               max_depth = 8,
                               num_leaves = 50,
                               bagging_fraction = 0.6,
                               feature_fraction = 0.7,
                               n_estimators = 1000)

gsearch3 =GridSearchCV(estimator = model_lgb3,
                       param_grid = params_reg,
                       scoring='neg_mean_squared_error',
                       cv=5,
                       verbose=1,
                       n_jobs=4)
gsearch3.fit(train_set[train_features],y_target)
[gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_]#0.5


params = {
        'boosting_type': 'gbdt', 
        'objective': 'regression',
        'learning_rate': 0.01, 
        'num_leaves': 50, 
        'max_depth': 8,
        'bagging_fraction':0.6,     
        'feature_fraction':0.7,
        'reg_lambda'：0.5
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

params['n_estimators'] = 1000
model_lgb_final = lgb.train(params,data_train,num_boost_round = 500)
pred_y2 = model_lgb_final.predict(valid_pd[train_features],num_iteration=310)
print('The rmse of prediction is:', mean_squared_error(valid_target,pred_y2) ** 0.5)
print(mean_squared_error(valid_target,[0]*len(pred_y2))**0.5)

pre_revenue = model_lgb_final.predict(test_df[train_features],num_iteration = 1000)
len(pre_revenue)
GA_predict_revenue = pd.DataFrame({'fullVisitorId':test_df['fullVisitorId'].values,
                                   'PredictedLogRevenue':pre_revenue})
GA_predict_revenue.loc[GA_predict_revenue['PredictedLogRevenue']<0.05,'PredictedLogRevenue'] = 0
GA_predict_revenue.to_csv('D:/DATA/python files/kaggle_kernel/google_revenue_prediction/predictions.csv',index = False)
