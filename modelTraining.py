# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss, roc_curve, auc, confusion_matrix
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import optuna
from scipy.stats import t
import pickle

decimal_num = 3
format_string = "." + str(decimal_num) + "f"
split_random_state = 3511
rf_random_state = 42
boruta_random_state = 42
is_internal_standardize = False
is_model_training = True

def interval(score, n, confidence_level):
    std_error = np.sqrt((score * (1 - score)) / n)
    t_value = t.ppf((1 + confidence_level) / 2, n - 1)
    auc_lower = score - t_value * std_error
    auc_upper = score + t_value * std_error
    if auc_lower < 0:
        auc_lower = 0.0
    if auc_upper > 1:
        auc_upper = 1.0
    return auc_lower, auc_upper

def metrics_cal(TN, FP, FN, TP, fpr, tpr, performance_all):
    sample_num = TP + FP + TN + FN
    confidence_level = 0.95
    accurate = (TP + TN) / (TP + FP + TN + FN)
    accuracy_lower, accuracy_upper = interval(accurate, sample_num, confidence_level)
    performance_all["Accurate"].append(f"{accurate:{format_string}} ({accuracy_lower:{format_string}}-{accuracy_upper:{format_string}})")
    sensitivity = TP / (TP + FN)
    sensitivity_lower, sensitivity_upper = interval(sensitivity, sample_num, confidence_level)
    performance_all["Sensitivity"].append(f"{sensitivity:{format_string}} ({sensitivity_lower:{format_string}}-{sensitivity_upper:{format_string}})")
    specificity = TN / (TN + FP)
    specificity_lower, specificity_upper = interval(specificity, sample_num, confidence_level)
    performance_all["Specificity"].append(f"{specificity:{format_string}} ({specificity_lower:{format_string}}-{specificity_upper:{format_string}})")
    npv = TN / (FN + TN)
    npv_lower, npv_upper = interval(npv, sample_num, confidence_level)
    performance_all["NPV"].append(f"{npv:{format_string}} ({npv_lower:{format_string}}-{npv_upper:{format_string}})")
    ppv = TP / (TP + FP)
    ppv_lower, ppv_upper = interval(ppv, sample_num, confidence_level)
    performance_all["PPV"].append(f"{ppv:{format_string}} ({ppv_lower:{format_string}}-{ppv_upper:{format_string}})")
    roc_auc = auc(fpr, tpr)
    auc_lower, auc_upper = interval(roc_auc, sample_num, confidence_level)
    performance_all["AUC"].append(f"{roc_auc:{format_string}} ({auc_lower:{format_string}}-{auc_upper:{format_string}})")
    return roc_auc, auc_lower, auc_upper, performance_all

def xgb_objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 2, 40),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2),
        'subsample': trial.suggest_float('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': trial.suggest_int('random_state', 1, 500)
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_val)
    loss = log_loss(y_val, y_pred)
    return loss

def modelTraining(inFileDir, outFileDir, inFileList, data_split_random_state, rf_random_state, boruta_random_state,
                  performance_all):
    for e in inFileList:
        inFilePath = os.path.join(inFileDir, e)
        model_path = os.path.join(outFileDir, e.replace('.xlsx', '.dat'))
        df = pd.read_excel(inFilePath)
        feature_columns = [col for col in df.columns if col != 'ID' and col != 'Label']
        X = df[feature_columns]
        y = df['Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                            random_state=data_split_random_state)
        df_train = pd.concat([X_train, y_train], axis=1)
        df_train['ID'] = df['ID'][df.index.isin(X_train.index)]

        df_test = pd.concat([X_test, y_test], axis=1)
        df_test['ID'] = df['ID'][df.index.isin(X_test.index)]

        rf = RandomForestClassifier(n_estimators=100, random_state=rf_random_state)
        boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, max_iter=100, random_state=boruta_random_state)
        boruta_selector.fit(X_train.values, df_train['Label'].values)
        selected_features = X_train.columns[boruta_selector.support_]
        selected_features1 = X_train.columns[boruta_selector.support_weak_]
        X_train_selected1 = X_train[selected_features]
        X_train_selected2 = X_train[selected_features1]
        X_train_selected = pd.concat([X_train_selected1, X_train_selected2], axis=1)
        if len(X_train_selected.columns) == 0:
            continue
        X_test_selected1 = X_test[selected_features]
        X_test_selected2 = X_test[selected_features1]
        X_test_selected = pd.concat([X_test_selected1, X_test_selected2], axis=1)

        y_train_selected = df_train['Label']
        y_test_selected = df_test['Label']

        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: xgb_objective(trial, X_train_selected, y_train_selected, X_test_selected,
                                                   y_test_selected), n_trials=100, show_progress_bar=True)
        best_params = study.best_params
        performance_all['model_parameters'].append(best_params) # for train data
        performance_all['model_parameters'].append(best_params) # for test data
        model = xgb.XGBClassifier(max_depth=best_params['max_depth'],
                                  learning_rate=best_params['learning_rate'],
                                  subsample=best_params['subsample'],
                                  colsample_bytree=best_params['colsample_bytree'],
                                  min_child_weight=best_params['min_child_weight'],
                                  random_state=best_params['random_state']
                                  )

        model.fit(X_train_selected, y_train_selected)
        y_train_pred_pro = model.predict_proba(X_train_selected)[:, 1]
        y_test_pred_pro = model.predict_proba(X_test_selected)[:, 1]

        fpr_train, tpr_train, thresholds_train = roc_curve(y_train_selected, y_train_pred_pro)
        pickle.dump(model, open(model_path, "wb"))
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test_selected, y_test_pred_pro)
        youden_index_train = tpr_train - fpr_train
        best_threshold_train = thresholds_train[np.argmax(youden_index_train)]

        y_train_pred = (y_train_pred_pro >= best_threshold_train).astype(int)
        performance_all['features'].append(list(X_train_selected.columns))
        performance_all['threshold'].append(best_threshold_train)
        performance_all['random_state'].append(data_split_random_state)
        temp_name_list = e.split("_")
        performance_all['mask'].append(temp_name_list[0])
        performance_all['model'].append('XGBoost')
        performance_all['isTraining'].append('1')
        TN_train, FP_train, FN_train, TP_train = confusion_matrix(y_train_selected, y_train_pred).ravel()
        auc_train, auc_lower_train, auc_upper_train, performance_all = metrics_cal(TN_train, FP_train, FN_train,
                                                                                         TP_train, fpr_train, tpr_train,
                                                                                         performance_all)

        y_test_pred = (y_test_pred_pro >= best_threshold_train).astype(int)
        performance_all['features'].append(list(X_test_selected.columns))
        performance_all['threshold'].append(best_threshold_train)
        performance_all['random_state'].append(data_split_random_state)
        performance_all['mask'].append(temp_name_list[0])
        performance_all['model'].append('XGBoost')
        performance_all['isTraining'].append('0')
        TN_test, FP_test, FN_test, TP_test = confusion_matrix(y_test_selected, y_test_pred).ravel()
        auc_test, auc_lower_test, auc_upper_test, performance_all = metrics_cal(TN_test, FP_test, FN_test,
                                                                                     TP_test, fpr_test, tpr_test,
                                                                                     performance_all)
    return performance_all

def featureStandardize(inFileDir, inFileList, outFileList):
    for i in range(0, len(inFileList)):
        inFilePath = os.path.join(inFileDir, inFileList[i])
        df = pd.read_excel(inFilePath)
        feature_columns = [col for col in df.columns if col != 'ID' and col != 'Label']
        X = df[feature_columns]
        y = df['Label']
        zscore_scaler = preprocessing.StandardScaler()
        zscore_scaler.fit(X)
        X_standard = pd.DataFrame(zscore_scaler.transform(X), columns=X.columns)
        df_stand = pd.concat([df['ID'], X_standard, y], axis=1)
        df_stand.to_excel(os.path.join(inFileDir, outFileList[i]), index=False)

def main():
    featuresDir = './features'
    resultDir = './results'
    outFilePath = os.path.join(resultDir, './performance_all.xlsx')
    inFileList = ['radiomics_features.xlsx', 'expand8_features.xlsx', 'expand17_features.xlsx',
                  'expand25_features.xlsx', 'expand33_features.xlsx']
    inFileStandList = ['radiomics_features_stand.xlsx', 'expand8_features_stand.xlsx', 'expand17_features_stand.xlsx',
                  'expand25_features_stand.xlsx', 'expand33_features_stand.xlsx']
    performance_all = {'random_state': [], 'mask': [], 'model': [], 'model_parameters':[], 'features':[],
                       'threshold': [], 'isTraining': [], 'Accurate': [], 'Sensitivity': [],
                       'Specificity': [], 'NPV': [], 'PPV': [], 'AUC': []}
    if is_internal_standardize:
        featureStandardize(featuresDir, inFileList, inFileStandList)
    if is_model_training:
        performance_all = modelTraining(featuresDir, resultDir, inFileStandList, split_random_state, rf_random_state,
                                        boruta_random_state, performance_all)

    df_final = pd.DataFrame(performance_all)
    if len(df_final) > 0:
        df_final.to_excel(outFilePath, index=False)

if __name__ == "__main__":
    main()
    print('Over!!')