from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

GB1 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=200, max_depth=5, random_state=42)
RF1 = RandomForestClassifier(n_estimators=35, class_weight='balanced', max_depth=5, random_state=42)
DT1 = DecisionTreeClassifier(class_weight={0: 1, 1: 3}, max_depth=6, random_state=42)
SVM1 = SVC(class_weight={0: 1, 1: 3}, probability=True, max_iter=3000, random_state=42)

models = [('Random Forest', RF1), ('Decision Tree', DT1), ('Gradient Boosting', GB1), ('SVM',SVM1)]

SoftVoting = VotingClassifier(
    estimators = models,
    voting ='soft',
    weights = [0.74, 0.67, 0.66, 0.62])

feature_names = ['original_gldm_DependenceNonUniformityNormalized',
        'log-sigma-2-mm-3D_glcm_Imc2',
        'log-sigma-3-mm-3D_glszm_SmallAreaLowGrayLevelEmphasis',
        'log-sigma-3-mm-3D_glcm_Idmn',
        'log-sigma-3-mm-3D_gldm_DependenceVariance',
        'log-sigma-1-mm-3D_firstorder_InterquartileRange',
        'log-sigma-2-mm-3D_gldm_SmallDependenceLowGrayLevelEmphasis',
        'log-sigma-2-mm-3D_firstorder_Skewness',
        'log-sigma-3-mm-3D_glszm_ZonePercentage',
        'log-sigma-1-mm-3D_glcm_Idmn',
        'original_firstorder_Energy','primarybin', 'histologytbin', 'IT+/-30 corrected', 'CLTA-4', 'Intact/Cav', 'metdose','idl', 'metvol', 'v12']

if __name__ == "__main__":
    df = pd.read_excel("BRAIN_Radiation.xlsx")
    X = df.loc[:, 'original_shape_Elongation':]
    y = df['rn (endpoint)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    X_train = X_train[feature_names]
    X_test = X_test[feature_names]

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_resampled_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    SoftVoting.fit(X_train, y_train)