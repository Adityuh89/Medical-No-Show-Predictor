import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import 
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline as SklearnPipeline  # use sklearn only now

df= pd.read_csv(r'c:\Users\Aditya\Downloads\medical-appointments-no-show-en.csv')
RANDOM_STATE = 42
CV_FOLDS = 5
print(df.columns)
#features in my dataset USEFULL
#comodity
print(df.head())
#usefull-specialty appotime gender  disability age under12 over 60, 
print(df.isnull().sum()/len(df))


#sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
#plt.show()

df['appointment_date'] = pd.to_datetime(df['appointment_date'], errors='coerce')
df['entry_service_date'] = pd.to_datetime(df['entry_service_date'], errors='coerce')
df['appointment_time'] = pd.to_datetime(df['appointment_time'], errors='coerce')


df['waiting_days'] = (df['appointment_date'] - df['entry_service_date']).dt.days 
df['appointment_hour'] = df['appointment_time'].dt.hour


df = df[df['waiting_days'] >= 0]

df.drop(columns=[
    'appointment_date',
    'date_of_birth',
    'entry_service_date',
    'appointment_time',
    
    'no_show_reason',
    'under_12_years_old',
    'over_60_years_old'
], inplace=True, errors='ignore')

y=df['no_show']
y = y.map({'no': 0, 'yes': 1})

features = [
    'age',
    'gender',
    'disability',
    'patient_needs_companion',
    'specialty',
    'max_temp_day',     
    'max_rain_day',      
    'rainy_day_before',
    'appointment_shift',
    'waiting_days',
    'appointment_hour',
    'average_temp_day',
    'average_rain_day',
    'storm_day_before',
    'rain_intensity',
    'heat_intensity'
]

# Select features
X = df[features]

# Separate column types
numerical_cols = [
    'age',
    'patient_needs_companion',
    'waiting_days',
    'appointment_hour',
    'average_temp_day',
    'average_rain_day',
    'storm_day_before',
    'max_temp_day',      
    'max_rain_day',     
    'rainy_day_before', 
]

categorical_cols = [
    'gender',
    'disability',
    'specialty',
    'appointment_shift',
    'rain_intensity',
    'heat_intensity'
]

X_train, X_val, y_train, y_val = train_test_split(
 X, y, random_state=RANDOM_STATE, stratify=y)

numerical_transformer=SklearnPipeline([
    ('imputer', SimpleImputer(strategy='median'))
])
categorical_transformer=SklearnPipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor=ColumnTransformer([
    ('num',numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])
scale = (y_train == 0).sum() / (y_train == 1).sum()
models={
    'randomforest': RandomForestClassifier(random_state=RANDOM_STATE,class_weight='balanced'),
    'xgboost': XGBClassifier(random_state=RANDOM_STATE, scale_pos_weight=scale,eval_metric='logloss',use_label_encoder=False)
}


param_grid={
    'randomforest':{
        'model__n_estimators':[100, 200, 300],
        'model__max_depth':[5, 10,],
    },
    'xgboost':{
            'model__n_estimators':[100,200,300],
            'model__max_depth':[3, 5],
            'model__learning_rate':[0.01,0.03,0.05],
            'model__min_child_weight': [1, 3, 5, 7]
    }
}
skf=StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE) 


for name, model_obj in models.items():
    
    pipeline = SklearnPipeline([
        ('preprocessor', preprocessor),
        ('model', model_obj)
    ])
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid[name],
        cv=skf,
        scoring='f1',   
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n Model: {name}")
    print(f"Best Params: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.2%}")
    
    final_best_model = grid_search.best_estimator_
   
    predictions = final_best_model.predict(X_val)
    
    
    acc = accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    recall = recall_score(y_val, predictions)
    
    print(f"Validation Accuracy: {acc:.2%}")
    print(f"Validation F1 Score: {f1:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(confusion_matrix(y_val, predictions))

    # ── Threshold Tuning ─────────────────────────────────────────────────────
    print(f"\n--- Threshold Tuning for {name} ---")
    proba = final_best_model.predict_proba(X_val)[:, 1]

    best_f1 = 0
    best_threshold = 0.5

    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
        preds = (proba >= threshold).astype(int)
        f1_t    = f1_score(y_val, preds, zero_division=0)
        rec_t   = recall_score(y_val, preds, zero_division=0)
        prec_t  = precision_score(y_val, preds, zero_division=0)
        print(f"Threshold {threshold}: F1={f1_t:.2%}  Recall={rec_t:.2%}  Precision={prec_t:.2%}")

        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = threshold

    print(f"\n🏆 Best Threshold for {name}: {best_threshold} (F1: {best_f1:.2%})")
    print(f"Final predictions at best threshold:")
    best_preds = (proba >= best_threshold).astype(int)
    print(confusion_matrix(y_val, best_preds))
feature_names = final_best_model.named_steps['preprocessor'].get_feature_names_out()
importances = final_best_model.named_steps['model'].feature_importances_

feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feat_imp.head(20))
import joblib
joblib.dump(final_best_model, 'noshow_model.pkl')
print(" Model saved!")