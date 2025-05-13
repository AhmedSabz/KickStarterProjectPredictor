import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from pandas import set_option
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import explained_variance_score
from sklearn import svm

from pandas import read_csv
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns

from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm, datasets
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
#eda find how many startups were unsuccesful and succesful
#Ahmed Sabzwari SVM
df = pd.read_csv("kickstarter_data_with_features.csv",low_memory=False,index_col=0)

df = df[df['state'].isin(['failed', 'canceled', 'successful'])].copy()

#divide it into two classes
df['state_binary'] =df['state'].replace({
    'failed': 'unsuccessful',
    'canceled': 'unsuccessful',
    'successful': 'successful',
    'live' : 'successful',
    'suspended' : 'unsuccessful'
})
df['state_encoded'] = LabelEncoder().fit_transform(df['state_binary'])
# sns.countplot(x='state_binary', data=df)
# plt.title("Distribution of Project Outcomes")
# plt.xlabel("Project Outcome")
# plt.ylabel("Count")
# plt.show()

# 4. Drop unnecessary columns
df = df.drop(columns=['source_url', 'friends', 'is_starred', 'is_backing'], errors='ignore')

# 5. Drop rows with missing numeric data 
df = df.dropna(subset=df.select_dtypes(include='number').columns)

# plt.figure(figsize=(15, 13))
# sns.heatmap(
#     df.select_dtypes(include='number').corr(),
#     annot=True,
#     cmap="coolwarm",
#     cbar=True,
#     linewidths=0.2,
#     annot_kws={"size": 10},          # bigger numbers in boxes
#     xticklabels=1, yticklabels=1     # keep all labels
# )
# plt.xticks(rotation=45, ha='right', fontsize=12)  # angle + size
# plt.yticks(fontsize=12)
# plt.title("Correlation Heatmap", fontsize=16)
# plt.tight_layout()  # fit everything within figure nicely
# plt.show()

# numeric_cols = df.select_dtypes(include='number').drop(columns=['state_encoded']).columns
# numeric_cols=df[['goal','backers_count','deadline_yr', 'state_changed_at_yr','pledged', 'launched_at_yr', 'deadline_month', 'usd_pledged', 'state_changed_at_month','launched_at_month','deadline_day','created_at_yr']]



# 6. Set up features and labels
y = df['state_encoded']
X = df.select_dtypes(include=['number']).drop(columns=['launched_at_hr','launched_at_day','name_len_clean','blurb_len','static_usd_rate','usd_pledged','created_at_yr','name_len','blurb_len_clean','launched_at_yr','state_changed_at_day','deadline_day','state_encoded','id','created_at_day', 'created_at_hr', 'deadline_hr', 'created_at_month', 'state_changed_at_hr', 'launched_at_month'])
# X=numeric_cols


standScaler=StandardScaler()

X_stand = pd.DataFrame(standScaler.fit_transform(X), columns=X.columns)


X_train, X_test, y_train, y_test = train_test_split(X_stand, y, test_size=0.2, random_state=42)

# Create an SVC classifier object 
# {0:2, 1:1} has the highest accuracy score with a linear svm
clf = svm.SVC(kernel='linear', C=10,class_weight={0:2, 1:1}) 

# Train the classifier using the training data
clf.fit(X_train, y_train)

#find feature ranking
feature_importance = np.abs(clf.coef_[0])
feature_names = X_train.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

print(importance_df)

# Predict the labels for the test data
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(clf.fit(X_train,y_train).decision_function(X_test))
# C_values = [0.001, 0.01, 0.1, 1, 10, 100]
# train_errors = []
# test_errors = []

# for C in C_values:
#     clf = SVC(kernel='linear', C=C)
#     clf.fit(X_train, y_train)

#     y_train_pred = clf.predict(X_train)
#     y_test_pred = clf.predict(X_test)

#     train_error = 1 - accuracy_score(y_train, y_train_pred)
#     test_error = 1 - accuracy_score(y_test, y_test_pred)

#     train_errors.append(train_error)
#     test_errors.append(test_error)

# # Plotting
# plt.figure(figsize=(8, 6))
# plt.plot(C_values, train_errors, label='Training Error', marker='o')
# plt.plot(C_values, test_errors, label='Test Error', marker='s')
# plt.xscale('log')
# plt.xlabel('Model Complexity (C)')
# plt.ylabel('Error')
# plt.title('Bias-C Tradeoff')
# plt.legend()
# plt.grid(True)
# plt.show()

#classification statistics results
print(classification_report(y_test, y_pred))

#ensemble techniques by Ali Fayed
df = read_csv("kickstarter_data_with_features.csv")
df = df.drop(df.columns[0], axis=1)

features = ['goal', 'backers_count', 'state_changed_at_yr', 'deadline_yr', 'pledged', 'state_changed_at_month', 'deadline_month']

model_df = df[features + ['state']].copy()
model_df = model_df.dropna(subset=model_df.select_dtypes(include='number').columns)

X = model_df.drop('state', axis=1).values
y = model_df['state'].values

print(X.shape)

scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
X = rescaledX

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

# Random Forest
max_depths = list(range(2, 8))
rf_accuracy = []
rf_precision = []
rf_recall = []

for i in max_depths:
    rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=42, max_depth=i)
    rnd_clf.fit(X_train, y_train)
    y_prob_rf = rnd_clf.predict_proba(X_test)
    y_pred_rf = rnd_clf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_rf)
    
    # Check what values are in y_test to determine the positive class
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:  # Binary classification
        # Assuming 'successful' is the positive class, but we'll check if it exists
        if 'successful' in unique_labels:
            pos_label = 'successful'
        else:
            # Use the second label (alphabetically) as positive class
            pos_label = sorted(unique_labels)[1]
            
        precision = precision_score(y_test, y_pred_rf, average='binary', pos_label=pos_label)
        recall = recall_score(y_test, y_pred_rf, average='binary', pos_label=pos_label)
    else:
        # For multi-class, use weighted average
        precision = precision_score(y_test, y_pred_rf, average='weighted')
        recall = recall_score(y_test, y_pred_rf, average='weighted')
    
    # Store metrics
    rf_accuracy.append(accuracy)
    rf_precision.append(precision)
    rf_recall.append(recall)
    
    print(f"Random Forest (Max Depth = {i}): Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    kfold = KFold(n_splits=4)
    cv_results = cross_val_score(rnd_clf, X_test, y_test, cv=kfold)
    msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
    print(msg)

from sklearn.ensemble import AdaBoostClassifier

learning_rates = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
ada_accuracy = []
ada_precision = []
ada_recall = []

for learning_rate in learning_rates:
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2), n_estimators=200, learning_rate=learning_rate, random_state=42)
    ada_clf.fit(X_train, y_train)
    y_pred_ada = ada_clf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_ada)
    
    # Check what values are in y_test to determine the positive class
    unique_labels = np.unique(y_test)
    if len(unique_labels) == 2:  # Binary classification
        # Assuming 'successful' is the positive class, but we'll check if it exists
        if 'successful' in unique_labels:
            pos_label = 'successful'
        else:
            # Use the second label (alphabetically) as positive class
            pos_label = sorted(unique_labels)[1]
            
        precision = precision_score(y_test, y_pred_ada, average='binary', pos_label=pos_label)
        recall = recall_score(y_test, y_pred_ada, average='binary', pos_label=pos_label)
    else:
        # For multi-class, use weighted average
        precision = precision_score(y_test, y_pred_ada, average='weighted')
        recall = recall_score(y_test, y_pred_ada, average='weighted')
    
    # Store metrics
    ada_accuracy.append(accuracy)
    ada_precision.append(precision)
    ada_recall.append(recall)
    
    print(f"AdaBoost (Learning Rate = {learning_rate:.2f}): Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")

    kfold = KFold(n_splits=4)
    cv_results = cross_val_score(ada_clf, X_test, y_test, cv=kfold)
    msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
    print(msg)

# Plot precision and recall for both models
plt.figure(figsize=(12, 10))

# Plot Random Forest metrics
plt.subplot(2, 1, 1)
plt.plot(max_depths, rf_precision, 'bo-', label='Precision')
plt.plot(max_depths, rf_recall, 'ro-', label='Recall')
plt.xlabel('Max Depth')
plt.ylabel('Score')
plt.title('Random Forest: Precision and Recall vs Max Depth')
plt.legend()
plt.grid(True)

# Plot AdaBoost metrics
plt.subplot(2, 1, 2)
plt.plot(learning_rates, ada_precision, 'bo-', label='Precision')
plt.plot(learning_rates, ada_recall, 'ro-', label='Recall')
plt.xlabel('Learning Rate')
plt.ylabel('Score')
plt.title('AdaBoost: Precision and Recall vs Learning Rate')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#Akshat Logistic Regression
"""FinalProject.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gOOOY2nEpLJ3t27dIO4bwN09i_d6o1UM
"""

import pandas as pd

df = pd.read_csv(
    'kickstarter_data_with_features.csv',
    on_bad_lines='skip',         # NEW way to skip bad lines
    quoting=3,                   # Ignore quote characters
    engine='python'              # Use the forgiving parser
)

            # Dimensions
print(df.columns)              # Column names
print(df.dtypes)               # Data types
print(df.head())               # First few rows

missing = df.isnull().sum()
print(missing[missing > 0])

print(df['state'].value_counts())

print(df.describe())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col} value counts:\n", df[col].value_counts().head(10))

import seaborn as sns
import matplotlib.pyplot as plt

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
corr = df[numeric_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.countplot(data=df, x='state')
plt.title('Success vs Failure Distribution')
plt.show()

# 2. Quick overview
print("Shape:", df.shape)
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

# 3. Target (“state”) distribution
print("\nState counts:\n", df['state'].value_counts())

# 4. Missing‑value check
print("\nMissing values per column:\n", df.isnull().sum())

# 1. Show all column names
print("Columns:", df.columns.tolist())

# 2. Inspect the contents of the 'state' column
print("\nUnique values in 'state':", df['state'].unique())

# 3. If you spot any other candidate for the outcome (e.g. 'status' or 'project_state'), check it too:
for col in ['status', 'project_state', 'final_state']:
    if col in df.columns:
        print(f"\nUnique values in '{col}':", df[col].unique())

# Load only the 'state' column from the full Kickstarter dataset
df_full = pd.read_csv('kickstarter_data_with_features.csv', usecols=['state'])

# 1. Show the unique state values (should be your campaign outcomes)
print("Unique campaign outcomes:", df_full['state'].unique())

# 2. Show the counts for each
print("\nOutcome counts:\n", df_full['state'].value_counts())

import pandas as pd

# 1. Load engineered features (squelch low‑memory dtype warnings)
df_feat = pd.read_csv(
    'kickstarter_data_with_features.csv',
    low_memory=False
)

# 2. Load true labels from the full dataset, rename 'state' ➔ 'label'
df_lbl = (
    pd.read_csv(
        'kickstarter_data_full.csv',
        usecols=['id','state'],
        low_memory=False
    )
    .rename(columns={'state':'label'})
)

# 3. Merge features + labels
df = df_feat.merge(df_lbl, on='id', how='inner')

# 4. Keep only finished campaigns
df = df[df['label'].isin(['successful','failed'])]

# 5. Binary target: 1 if successful, else 0
df['target'] = (df['label']=='successful').astype(int)

# 6. Drop now‑unneeded columns (textual/URL fields & the raw label)
to_drop = [
    'label',          # raw campaign outcome (we now have 'target')
    'photo','name',
    'blurb','slug',
    'urls','source_url',
    'creator','profile'
]
df.drop(columns=[c for c in to_drop if c in df.columns], inplace=True)

# Quick sanity check
print("Shape after merge/filter:", df.shape)
print("\nTarget balance:\n", df['target'].value_counts())
print("\nRemaining columns:\n", df.columns.tolist())

import pandas as pd

# Assuming df is already merged from the previous step

# 1. Drop unwanted columns
for col in ['Unnamed: 0','id','state']:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# 2. Show top 20 features with most missing values
null_counts = df.isnull().sum().sort_values(ascending=False)
print(null_counts.head(20))

import numpy as np

# Pull out feature names after preprocessing
ohe = clf.named_steps['preproc'].named_transformers_['cat'].named_steps['onehot']
cat_names = ohe.get_feature_names_out(categorical_cols)
feat_names = np.concatenate([numeric_cols, cat_names])

coefs = clf.named_steps['model'].coef_[0]
imp = sorted(zip(feat_names, coefs), key=lambda x: abs(x[1]), reverse=True)[:10]

print("Top 10 features by absolute coefficient:")
for name, coef in imp:
    print(f"  {name:30s} {coef:.3f}")

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    clf, X, y,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
print("5‑fold accuracy:", scores)
print("Mean ± std:", scores.mean(), "+/-", scores.std())

from sklearn.model_selection import cross_val_score

# 1. Drop the leaking spotlight & staff_pick flags
for leak in ['spotlight','staff_pick']:
    if leak in df_full.columns:
        df_full.drop(columns=leak, inplace=True)

# 2. Re‑define X & y
X = df_full.drop(columns='target')
y = df_full['target']

# 3. Re‑identify columns
categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()
numeric_cols     = [c for c in X.columns if c not in categorical_cols]

# 4. Rebuild the preprocessing + model pipeline (as before)
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer,     numeric_cols),
    ('cat', categorical_transformer, categorical_cols),
])

clf = Pipeline([
    ('preproc', preprocessor),
    ('model',   LogisticRegression(class_weight='balanced', max_iter=1000))
])

# 5. 5‑fold CV on accuracy
scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
print("5‑fold accuracies:", scores)
print("Mean ± std:", scores.mean(), "+/-", scores.std())

from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__penalty': ['l2'],                # you can try 'l1' with solver='saga'
    'model__C': [0.01, 0.1, 1, 10, 100],      # inverse regularization strength
}

grid = GridSearchCV(
    clf, param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# --- assume df_model already exists, with 'create_to_launch' and 'launch_to_deadline' still as strings

# 1. Convert durations to total days (float)
for col in ['create_to_launch','launch_to_deadline']:
    df_full[col] = (
        pd.to_timedelta(df_full[col], errors='coerce')  # parse to timedelta
          .dt.total_seconds()                            # seconds
        / 86400                                          # → days
    )

# (Optional) If you still have nulls after parsing, fill them
df_full[['create_to_launch','launch_to_deadline']] = df_full[['create_to_launch','launch_to_deadline']].fillna(0)

# 2. Now generate degree‑2 only‑interaction features
core = df_full[['goal','create_to_launch','launch_to_deadline']]
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
core_poly = poly.fit_transform(core)
poly_cols = poly.get_feature_names_out(core.columns)

df_poly = pd.DataFrame(core_poly, columns=poly_cols, index=df_full.index)

# 3. Merge back and drop the originals
df_model = pd.concat([df_full, df_poly], axis=1)
df_model.drop(columns=['goal','create_to_launch','launch_to_deadline'], inplace=True)

# Then continue with your cyclic encodings, pipeline building, and CV as before!

from sklearn.ensemble import HistGradientBoostingClassifier
gb_clf = Pipeline([
    ('preproc', preprocessor),
    ('model',   HistGradientBoostingClassifier(max_iter=200, random_state=42))
])

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Rebuild the preprocessing so OneHotEncoder is named properly
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer,     numeric_cols),
    ('cat', categorical_transformer, categorical_cols),
])

# Recreate the HGB pipeline
gb_clf = Pipeline([
    ('preproc', preprocessor),
    ('model',   HistGradientBoostingClassifier(max_iter=200, random_state=42))
])

# Fit & evaluate on hold‑out
gb_clf.fit(X_train, y_train)
y_pred  = gb_clf.predict(X_test)
y_proba = gb_clf.predict_proba(X_test)[:,1]

print("Test set classification report:\n", classification_report(y_test, y_pred))
print("Test set ROC‑AUC:", roc_auc_score(y_test, y_proba))
