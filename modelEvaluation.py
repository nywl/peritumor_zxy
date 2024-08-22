from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
from sklearn.metrics import roc_curve, auc, confusion_matrix
import modelTraining

is_external_standardize = False # 是否对外部测试集标准化处理
data_split_random_state = 3511


# 根据标准化后内部数据集的均值和方差对非标准化的数据集进行标准化处理
def featureStandardize(inFileDir, internal_file_list, external_file_list):
    for i in range(0, len(internal_file_list)):
        internal_file = os.path.join(inFileDir, internal_file_list[i])
        external_file = os.path.join(inFileDir, external_file_list[i])
        df = pd.read_excel(internal_file)
        feature_columns = [col for col in df.columns if col != 'ID' and col != 'Label']
        X = df[feature_columns]  # 特征值
        zscore_scaler = preprocessing.StandardScaler()
        zscore_scaler.fit(X)

        # 外部测试集排除部分个体后进行标准化处理
        df1 = pd.read_excel(external_file)
        feature_columns1 = [col for col in df1.columns if col != 'ID' and col != 'Label']
        X1 = df1[feature_columns1]
        X1_standard = pd.DataFrame(zscore_scaler.transform(X1), columns=X1.columns)
        X1_standard['ID'] = df1['ID'].reset_index(drop=True)
        X1_standard['Label'] = df1['Label'].reset_index(drop=True)
        result_file = external_file.replace('.xlsx', '_stand.xlsx')
        X1_standard.to_excel(result_file, index=False)

def main():
    inFileDir = "./features"
    resultFileDir = "./results"
    result_file = "final_performance.xlsx"
    train_performance_file = "performance_all.xlsx"
    result_file_path = os.path.join(resultFileDir, result_file)
    individual_file_path = os.path.join( resultFileDir, "individual_pred.xlsx")
    train_performance_path = os.path.join(resultFileDir, train_performance_file)
    raw_file = "radiomics_features.xlsx"
    expanded_file_list = ["expand8_features.xlsx", "expand17_features.xlsx",
                          "expand25_features.xlsx", "expand33_features.xlsx"]
    external_file_list = ["external_radiomics_features.xlsx", "external_expand8_features.xlsx",
                           "external_expand17_features.xlsx", "external_expand25_features.xlsx",
                           "external_expand33_features.xlsx"]
    performance_all = {'random_state': [], 'mask': [], 'model': [], 'model_parameters': [], 'features': [],
                       'threshold': [], 'isTraining': [], 'Accurate': [], 'Sensitivity': [],
                       'Specificity': [], 'NPV': [], 'PPV': [], 'AUC': []}
    individual_pred = pd.DataFrame()
    train_performance = pd.read_excel(train_performance_path)

    if is_external_standardize:
        internal_file_list = [raw_file] + expanded_file_list
        featureStandardize(inFileDir, internal_file_list, external_file_list)

    inFileStandList = ["radiomics_features_stand.xlsx", "expand8_features_stand.xlsx",
                       "expand17_features_stand.xlsx",
                       "expand25_features_stand.xlsx", "expand33_features_stand.xlsx"]
    inFileStandList1 = ["external_radiomics_features_stand.xlsx", "external_expand8_features_stand.xlsx",
                       "external_expand17_features_stand.xlsx",
                       "external_expand25_features_stand.xlsx", "external_expand33_features_stand.xlsx"]

    train_performance_xgboost = train_performance[train_performance['model'] == 'XGBoost']
    for i in range(0, len(inFileStandList)):
        print('循环', i)
        inFilePath = os.path.join(inFileDir, inFileStandList[i])
        df = pd.read_excel(inFilePath)

        temp_name_list = inFileStandList[i].split("_")
        modal_temp = temp_name_list[0]
        df_temp = train_performance_xgboost[train_performance_xgboost['mask'] == modal_temp]
        if len(df_temp) == 0:
            continue

        feature_columns = [item.strip(" '\n\r") for item in str(df_temp.iloc[0]['features'])[1:-1].split(",")]
        X_ID = df[['ID'] + feature_columns]
        y = df['Label']
        X_train_ID, X_test_ID, y_train, y_test = train_test_split(X_ID, y, test_size=0.2, stratify=y,
                                                            random_state=data_split_random_state)
        X_train = X_train_ID[feature_columns]
        X_test = X_test_ID[feature_columns]
        model_path = os.path.join(resultFileDir, inFileStandList[i].replace('.xlsx', '.dat'))
        model = pickle.load(open(model_path, "rb"))

        inFilePath1 = os.path.join(inFileDir, inFileStandList1[i])
        df1 = pd.read_excel(inFilePath1)
        X1_ID = df1[['ID'] + feature_columns]
        X1 = X1_ID[feature_columns]
        y1 = df1['Label']

        y_train_pred_pro = model.predict_proba(X_train)[:, 1] # train
        y_test_pred_pro = model.predict_proba(X_test)[:, 1] # valid
        y_external_pred_pro = model.predict_proba(X1)[:, 1] # test


        fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred_pro)
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_pred_pro)
        fpr_external, tpr_external, thresholds_external = roc_curve(y1, y_external_pred_pro)
        best_threshold_train = float(train_performance_xgboost.iloc[i*2]['threshold'])


        y_train_pred = (y_train_pred_pro >= best_threshold_train).astype(int)
        individual_train = pd.concat([X_train_ID['ID'], y_train], axis=1)
        individual_train.reset_index(drop=True, inplace=True)
        individual_train_pred = pd.DataFrame(y_train_pred, columns=['Pred'])
        individual_train_pro = pd.DataFrame(y_train_pred_pro, columns=['PredPro'])
        individual_train = pd.concat([individual_train, individual_train_pred, individual_train_pro], axis=1)
        individual_train['Group'] = 'Train'
        individual_train['Mask'] = temp_name_list[0]
        individual_pred = pd.concat([individual_pred, individual_train], ignore_index=True)
        performance_all['threshold'].append(best_threshold_train)
        performance_all['random_state'].append(data_split_random_state)
        if temp_name_list[1] == 'only':
            performance_all['mask'].append(temp_name_list[0] + '_' + temp_name_list[1])
        else:
            performance_all['mask'].append(temp_name_list[0])
        performance_all['model'].append('XGBoost')
        performance_all['model_parameters'].append(model.get_params())
        performance_all['features'].append(feature_columns)
        performance_all['isTraining'].append('1')
        TN_train, FP_train, FN_train, TP_train = confusion_matrix(y_train, y_train_pred).ravel()
        auc_train, auc_lower_train, auc_upper_train, performance_all = modelTraining.metrics_cal(TN_train, FP_train,
                                                           FN_train,TP_train, fpr_train, tpr_train, performance_all)


        y_test_pred = (y_test_pred_pro >= best_threshold_train).astype(int)
        individual_test = pd.concat([X_test_ID['ID'], y_test], axis=1)
        individual_test.reset_index(drop=True, inplace=True)
        individual_test_pred = pd.DataFrame(y_test_pred, columns=['Pred'])
        individual_test_pro = pd.DataFrame(y_test_pred_pro, columns=['PredPro'])
        individual_test = pd.concat([individual_test, individual_test_pred, individual_test_pro], axis=1)
        individual_test['Group'] = 'Validation'
        individual_test['Mask'] = temp_name_list[0]
        individual_pred = pd.concat([individual_pred, individual_test], ignore_index=True)
        performance_all['threshold'].append(best_threshold_train)
        performance_all['random_state'].append(data_split_random_state)
        if temp_name_list[1] == 'only':
            performance_all['mask'].append(temp_name_list[0] + '_' + temp_name_list[1])
        else:
            performance_all['mask'].append(temp_name_list[0])
        performance_all['model'].append('XGBoost')
        performance_all['model_parameters'].append(model.get_params())
        performance_all['features'].append(feature_columns)
        performance_all['isTraining'].append('0') # 2表示外部测试集
        TN_test, FP_test, FN_test, TP_test = confusion_matrix(y_test, y_test_pred).ravel()
        auc_test, auc_lower_test, auc_upper_test, performance_all = modelTraining.metrics_cal(TN_test, FP_test, FN_test,
                                                                                TP_test, fpr_test, tpr_test,
                                                                                performance_all)

        y_external_pred = (y_external_pred_pro >= best_threshold_train).astype(int)
        individual_external = pd.concat([X1_ID['ID'], y1], axis=1)
        individual_external.reset_index(drop=True, inplace=True)
        individual_external_pred = pd.DataFrame(y_external_pred, columns=['Pred'])
        individual_external_pro = pd.DataFrame(y_external_pred_pro, columns=['PredPro'])
        individual_external = pd.concat([individual_external, individual_external_pred, individual_external_pro], axis=1)
        individual_external['Group'] = 'External_Test'
        individual_external['Mask'] = 'external_' + temp_name_list[0]
        individual_pred = pd.concat([individual_pred, individual_external], ignore_index=True)

        performance_all['threshold'].append(best_threshold_train)
        performance_all['random_state'].append(-1)
        performance_all['mask'].append(temp_name_list[0])
        performance_all['model'].append('XGBoost')
        performance_all['model_parameters'].append(model.get_params())
        performance_all['features'].append(feature_columns)
        performance_all['isTraining'].append('2')
        TN_external, FP_external, FN_external, TP_external = confusion_matrix(y1, y_external_pred).ravel()
        auc_external, auc_lower_external, auc_upper_external, performance_all = modelTraining.metrics_cal\
            (TN_external, FP_external,FN_external, TP_external, fpr_external, tpr_external, performance_all)

    if len(performance_all) > 0:
        pd.DataFrame(performance_all).to_excel(result_file_path, index=False)
        individual_pred.to_excel(individual_file_path, index=False)


if __name__ == "__main__":
    main()
    print('Over!!')