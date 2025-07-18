#%%
exec(open("Boosting_Modeling_CatBoost-TSS.py").read())
###############################################################

#### CatBoost with Time Series 5-fold CV Regression Result ####
# Time taken: 150m 46.9s
# Searched CV comb: Fitting 5 folds for each of 6 candidates, totalling 30 fits

# CatBoost 최적 하이퍼 파라미터:  {'colsample_bylevel': 1.0, 'l2_leaf_reg': 3, 'learning_rate': 0.1, 'max_depth': 9, 'n_estimators': 1000, 'subsample': 0.7}
# Stopped by overfitting detector  (20 iterations wait)

# bestTest = 0.2300888523
# bestIteration = 41

# Shrink model to first 42 iterations.

# |   Set   |      RMSE      |      MAE      |      MAPE      |
# |---------|----------------|---------------|----------------|
# |  Train  |15778.137598    |8211.585140    |0.140434        |
# |  Valid  |29762.405555    |15549.942221   |0.189765        |
# |  Test   |29209.026750    |16053.838783   |0.162300        |

###############################################################


#%%
exec(open("Boosting_Modeling_LightGBM-TSS.py").read())
###############################################################

#### LightGBM with Time Series 5-fold CV Regression Result ####
# Time taken: 83m 7.6s
# Searched CV comb: Fitting 5 folds for each of 12 candidates, totalling 60 fits

# LightGBM 최적 하이퍼 파라미터:  {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 9, 'min_child_samples': 10, 'n_estimators': 1000, 'reg_alpha': 0, 'reg_lambda': 0.1, 'subsample': 0.7}
# Stopped by overfitting detector  (20 iterations wait)

# Early stopping, best iteration is:
# [48]	valid_0's rmse: 0.244429	valid_0's l2: 0.0597455
# Shrink model to first 42 iterations.

# |   Set   |      RMSE      |      MAE      |      MAPE      |
# |---------|----------------|---------------|----------------|
# |  Train  |16520.962723    |8467.132612    |0.144209        |
# |  Valid  |34496.052437    |17500.072098   |0.204564        |
# |  Test   |35467.563694    |19458.163407   |0.192276        |

###############################################################


#%%
exec(open("Boosting_Modeling_XGBoost-TSS.py").read())
###############################################################

#### XGBoost with Time Series 5-fold CV Regression Result ####
# Time taken: 10m 25.5s
# Searched CV comb: Fitting 5 folds for each of 12 candidates, totalling 60 fits

# XGBoost 최적 하이퍼 파라미터:  {'colsample_bytree': 1.0, 'gamma': 0, 'learning_rate': 0.1, 'max_depth': 9, 'min_child_weight': 5, 'n_estimators': 1000, 'subsample': 1.0}

# [48]	validation_0-rmse:0.22134

# |   Set   |      RMSE      |      MAE      |      MAPE      |
# |---------|----------------|---------------|----------------|
# |  Train  |13187.704005    |7127.851414    |0.122785        |
# |  Valid  |25693.876491    |14726.108607   |0.175882        |
# |  Test   |27692.264768    |15965.169735   |0.162499        |

###############################################################


#%%
exec(open("Bagging_Modeling_RF-TSS.py").read())
###############################################################

#### RandomForest with Time Series 5-fold CV Regression Result ####
# Time taken: 41m 2.7s
# Searched CV comb: Fitting 5 folds for each of 12 candidates, totalling 60 fits

# RandomForest 최적 하이퍼 파라미터:  {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'n_estimators': 200}

# |   Set   |      RMSE      |      MAE      |      MAPE      |
# |---------|----------------|---------------|----------------|
# |  Train  |6951.943013     |3321.762270    |0.057424        |
# |  Test   |20178.523618    |9821.548001	 |0.092682        |

###############################################################



#%%
exec(open("Boosting_Modeling_GB-TSS.py").read())
###############################################################

#### GradientBoost with Time Series 5-fold CV Regression Result ####
# V2
# Time taken: 40m 19.8s
# Searched CV comb: Fitting 5 folds for each of 72 candidates, totalling 360 fits

# HistGradientBoostingRegressor 최적 하이퍼 파라미터:  {'learning_rate': 0.1, 'max_depth': 15, 'max_iter': 1200, 'min_samples_leaf': 10}

# |   Set   |      RMSE      |      MAE      |      MAPE      |
# |---------|----------------|---------------|----------------|
# |  Train  |7151.832596     |3687.817040	 |0.063581        |
# |  Test   |18680.396672    |9939.297323    |0.089568        |

# V3
# Time taken: 46m 28.6s
# HistGradientBoostingRegressor 최적 하이퍼 파라미터:  {'learning_rate': 0.07, 'max_depth': 15, 'max_iter': 2000, 'min_samples_leaf': 10}

# |   Set   |      RMSE      |      MAE      |      MAPE      |
# |---------|----------------|---------------|----------------|
# |  Train  |6721.680362     |3466.038984	 |0.059923        |
# |  Test   |17828.432856    |9586.932307    |0.087905        |

# V4
# Time taken: 86m 20.7s
# HistGradientBoostingRegressor 최적 하이퍼 파라미터:  {'learning_rate': 0.1, 'max_depth': 13, 'max_iter': 3500, 'min_samples_leaf': 8}
# |   Set   |      RMSE      |      MAE      |      MAPE      |
# |---------|----------------|---------------|----------------|
# |  Train  |5376.82689      |2751.201446	 |0.047727        |
# |  Test   |15046.453018    |8160.695610    |0.076428        |



###############################################################


#%%
exec(open("Stacking_and_Voting_with_all_models-TSS.py").read())
###########################################################################

######### Stacking and Voting with all Models Regression Result ###########
# Time taken: 30m 41s


############################ Stacking Result ##############################
# Time taken: 24m 41.9s

# |   Set   |      RMSE      |      MAE      |      MAPE      |
# |---------|----------------|---------------|----------------|
# |  Train  |8388.895813     |4454.908733    |0.069975        |
# |  Test   |15606.410418    |8307.640410    |0.084930        |

############################# Voting Result ###############################
# Time taken: 5m 33.7s

# |   Set   |      RMSE      |      MAE      |      MAPE      |
# |---------|----------------|---------------|----------------|
# |  Train  |5861.14801      |2965.027641    |0.051734        |
# |  Test   |16916.411992    |8584.747387    |0.077578        |


###########################################################################


#%%
exec(open("Stacking_and_Voting_with_selected_models-TSS.py").read())
###########################################################################

####### Stacking and Voting with selected Models Regression Result ########
# Time taken: 10m 55.0s


############################ Stacking Result ##############################
# Time taken: 24m 41.9s

# |   Set   |      RMSE      |      MAE      |      MAPE      |
# |---------|----------------|---------------|----------------|
# |  Train  |8933.403787    |4623.366065   |0.076155        |
# |  Test   |17222.978410    |9335.462332   |0.094886        |



############################# Voting Result ###############################
# Time taken: 4m 34s

# |   Set   |      RMSE      |      MAE      |      MAPE      |
# |---------|----------------|---------------|----------------|
# |  Train  |5861.14801      |2965.027641    |0.051734        |
# |  Test   |16916.411992    |8584.747387    |0.077578        |


###########################################################################
