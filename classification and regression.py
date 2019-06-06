# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:21:37 2018

@author: twope
"""
def add_target(train_period,target_period):
    
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