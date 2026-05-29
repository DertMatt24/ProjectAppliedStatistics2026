
from sklearn.linear_model import LinearRegression


from src.PatientsDSsetup import PatientsDSsetup
from src.linear_model.FeatureSelection import FeatureSelection

from sklearn.model_selection import KFold, cross_val_score




# age - Age of the patient [Years]
# sex - sex of the patient [F: Female, M: Male]
# height - Height of the patient [Centimeters]
# weight - Weight of the patient [Kilograms]
# pulse - Pulse of the patient [Beats per minute]
# BPsys/BPdia - Systolic blood pressure / diastolic blood pressure of the patient [Millimeters of mercury/Millimeters of mercury]
# ODI - Oxygen desaturation index of the patient for the night. For some healthy patients will be none [Continuous range 0-…]
# NAp - Number of cases apnoe of the patient for the night. For healthy patients is none [Range 0-…]
# NHyp - Number of cases hyperapnoe of the patient for the night. For healthy patients is none [Range 0-…]
# AI - Apnoe index of the patient for the night. Calculated by the formula: NAp / TimeOfRecordInHours. For healthy patients is none [Continuous range 0-…]
# HI - Hyperapnoe index of the patient for the night. Calculated by the formula: NHyp / TimeOfRecordInHours. For healthy patients is none [Continuous range 0-…]
# AHI - Apnoe-hyperapnoe index of the patient for the night. Calculated by the formula: (NAp+NHyp) / TimeOfRecordInHours. For healthy patients is none [Continuous range 0-…]

def prep(df):
    cols_to_fix = ['BMI', 'age', 'pulse', 'AHI', 'HI', 'AI', 'NHyp', 'NAp', 'ODI', 'weight', 'height']
    for col in cols_to_fix:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.').astype(float)

    df['sex'] = df['sex'].map({'M': 1, 'F': 0})

    #df_grouped = df.groupby('user_id').mean(numeric_only=True)

    df = df.fillna(0)

    return df

def k_fold(model, X, y, k):
    kf = KFold(n_splits=k, shuffle=True#, random_state=42
               )
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

    return kf, scores

def sel_features(X, bool_sel):
    result = []
    for i in range(len(bool_sel)):
        if bool_sel[i]:
            result.append(X.columns[i])

    return result

def rfe_k_fold(X, y, model, k):
    kf, scores = k_fold(model, X, y, k)

    print(f"Scores for each fold: {scores}")
    print(f"The 'Honest' R2 Score: {scores.mean():.4f}")
    print(f"How much the score varies: {scores.std():.4f}")

    rfe = FeatureSelection.recursive_feature_elimination_kf(model, cv=kf)
    rfe.fit(X, y)

    print(f"Selected features: {sel_features(X, rfe.support_)}")



def main():
    p = PatientsDSsetup.load_dataframe()
    df = PatientsDSsetup.add_BMI(p)

    df_prep = prep(df)
    #df_prep = df_prep[df_prep.sex == 0].copy()

    # selecting predictors and to predict
    X = df_prep[['BMI', 'age', 'sex','ODI','pulse']].copy()
    # AHI, HI, AI, NHyp, NAp
    y = df_prep['AHI']


    model = LinearRegression()
    #model = RandomForestRegressor()

    print("Mutual Information")
    mi_scores = FeatureSelection.computing_mi_scores(X, y)



    print("RFE with cv k_fold")
    rfe_k_fold(X, y, model, 5)

    print("Backward elimination")
    feature, model_sel = FeatureSelection.backward_elimination(X, y)
    FeatureSelection.be_score(X, y, model_sel)
    print(feature)

    print("Forward selection")
    selected_features, s = FeatureSelection.forward_selection(model, X, y, 5, 3)
    print(selected_features)
    FeatureSelection.fs_score(model, X, y, selected_features, 5)
    ##################################################################################


if __name__ == "__main__":
    main()