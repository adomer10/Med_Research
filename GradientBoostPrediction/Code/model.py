import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTENC
import warnings


warnings.filterwarnings("ignore", category = FutureWarning)


def find_best_params(X_train, y_train):
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 75, 100, 200, 300, 400],
        'max_depth': [1, 2, 3, 4, 5, 7, 8, 9, 10],
        'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2],
        'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0],
        'colsample_bytree': [0.5, 0.7, 0.8, 1.0],
        'gamma': [0, 0.001, 0.01, 0.1, 0.2]
        }
    
    # Define the model
    model = xgb.XGBClassifier()
    
    # Perform grid search
    grid_search = GridSearchCV(
            estimator = model,
            param_grid = param_grid,
            cv = 5,
            scoring = 'roc_auc',
            n_jobs = -1,
            verbose = 0
            )
    
    # Fit the grid search model
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_params_


TRAIN = False

# Load the data
df = pd.read_csv("../Data/Final_Data_Dictionary_in_order.csv")

legs = ["Left", "Right"]
parts = ["High_Thigh", "Low_Thigh", "Calf", "Ankle", "Metatarsal"]

params = {
    "Left_High_Thigh": {
        'colsample_bytree': 0.7,
        'gamma': 0.2,
        'learning_rate': 0.1,
        'max_depth': 2,
        'n_estimators': 50,
        'subsample': 0.2
        },
    "Left_Low_Thigh": {
        'colsample_bytree': 0.5,
        'gamma': 0.01,
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 50,
        'subsample': 0.5
        },
    "Left_Calf": {
        'colsample_bytree': 1.0,
        'gamma': 0.2,
        'learning_rate': 0.001,
        'max_depth': 3,
        'n_estimators': 100,
        'subsample': 0.2
        },
    "Left_Ankle": {
        'colsample_bytree': 0.7,
        'gamma': 0.01,
        'learning_rate': 0.2,
        'max_depth': 3,
        'n_estimators': 50,
        'subsample': 0.5
        },
    "Left_Metatarsal": {
        'colsample_bytree': 0.8,
        'gamma': 0,
        'learning_rate': 0.01,
        'max_depth': 3,
        'n_estimators': 400,
        'subsample': 1.0
        },
    "Right_High_Thigh": {
        'colsample_bytree': 0.8,
        'gamma': 0,
        'learning_rate': 0.01,
        'max_depth': 2,
        'n_estimators': 400,
        'subsample': 0.8
        },
    "Right_Low_Thigh": {
        'colsample_bytree': 0.5,
        'gamma': 0.1,
        'learning_rate': 0.05,
        'max_depth': 5,
        'n_estimators': 75,
        'subsample': 0.8
        },
    "Right_Calf": {
        'colsample_bytree': 1.0,
        'gamma': 0,
        'learning_rate': 0.2,
        'max_depth': 1,
        'n_estimators': 75,
        'subsample': 0.4
        },
    "Right_Ankle": {
        'colsample_bytree': 1.0,
        'gamma': 0.01,
        'learning_rate': 0.01,
        'max_depth': 8,
        'n_estimators': 300,
        'subsample': 1.0
        },
    "Right_Metatarsal": {
        'colsample_bytree': 0.5,
        'gamma': 0,
        'learning_rate': 0.01,
        'max_depth': 3,
        'n_estimators': 300,
        'subsample': 0.7
        }
    }

for leg in legs:
    
    for part in parts:
        
        body_part = f"{leg}_{part}"
        print(f"Working on {body_part}")
        
        if part == "High_Thigh":
            
            temp_df = df.copy(deep = True)
            
            temp_df = temp_df[
                [f"{body_part.lower()}_grade", f"{body_part.lower()}_pressure", f"{body_part.lower()}_index",
                 f"{body_part}_Disease"]]
            
            # Turn -1 in Body_Part_Disease to 0
            temp_df[f"{body_part}_Disease"] = temp_df[f"{body_part}_Disease"].apply(lambda x: 0 if x == -1 else x)
            
            X = temp_df.drop(columns = [f"{body_part}_Disease"])
            y = temp_df[f"{body_part}_Disease"]
            
            # Use SMOTE to balance the data
            smote = SMOTENC(categorical_features = [0], random_state = 42)
            
            # Fit the SMOTE model
            X, y = smote.fit_resample(X, y)
            
            # One hot encode the grade with one hot encoding
            values = temp_df[f"{body_part.lower()}_grade"].values.reshape(-1, 1)
            grade_encoder = OneHotEncoder(categories = [['A', 'B', 'C', 'D', 'I']], sparse_output = False)
            encoded_values = grade_encoder.fit_transform(values)
            
            # Drop the grade column
            X = X.drop(f"{body_part.lower()}_grade", axis = 1)
            
            # Add the encoded values to the dataframe
            X = pd.concat([X, pd.DataFrame(encoded_values, columns = ['A', 'B', 'C', 'D', 'I'])], axis = 1)
            
            # Standardize the data
            scaler = StandardScaler()
            X[[f"{body_part.lower()}_pressure", f"{body_part.lower()}_index"]] = scaler.fit_transform(
                    X[[f"{body_part.lower()}_pressure", f"{body_part.lower()}_index"]])
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
            
            if TRAIN:
                # Find the best parameters
                best_params = find_best_params(X_train, y_train)
            
                # Update the dictionary
                params[body_part] = best_params
                
            else:
                best_params = params[body_part]
                
            # Create the model
            model = xgb.XGBClassifier(**best_params)
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Predict the test data
            y_pred = model.predict(X_test)
            
            # Print the accuracy
            print(f"Accuracy for {body_part}: {accuracy_score(y_test, y_pred)}")
            
            # Print the classification report
            print(f"Classification Report for {body_part}:\n{classification_report(y_test, y_pred)}")
            
            # Print the confusion matrix
            print(f"Confusion Matrix for {body_part}:\n{confusion_matrix(y_test, y_pred)}")
            
            # Create the prediction column
            df[f"{body_part}_Prediction"] = ''
            
            # Predict the data
            for index, row in df.iterrows():
                
                data = (row[
                    [f"{body_part.lower()}_grade", f"{body_part.lower()}_pressure", f"{body_part.lower()}_index"]]
                        .values.reshape(1, -1))
                data = pd.DataFrame(data, columns = [f"{body_part.lower()}_grade",
                                                     f"{body_part.lower()}_pressure", f"{body_part.lower()}_index"])
                
                # One hot encode the grade
                grade = grade_encoder.transform(data[f"{body_part.lower()}_grade"].values.reshape(-1, 1))
                
                # Drop the grade column
                data = data.drop(f"{body_part.lower()}_grade", axis = 1)
                
                data = pd.concat([data, pd.DataFrame(grade, columns = ['A', 'B', 'C', 'D', 'I'])], axis = 1)
                
                # Scale the data
                data[[f"{body_part.lower()}_pressure", f"{body_part.lower()}_index"]] = scaler.transform(
                        data[[f"{body_part.lower()}_pressure", f"{body_part.lower()}_index"]]
                        )
                
                # Make the prediction
                prediction = model.predict(data)
                
                # Add the prediction to the dataframe
                df.at[index, f"{body_part}_Prediction"] = prediction[0]
                
        else:
            
            body_index = parts.index(part)
            
            previous_part = parts[body_index - 1]
            
            previous_part = f"{leg}_{previous_part}"
            
            temp_df = df.copy(deep = True)
            
            temp_df = temp_df[
                [f"{body_part.lower()}_grade", f"{body_part.lower()}_pressure", f"{body_part.lower()}_index",
                 f"{body_part}_Disease", f"{previous_part}_Prediction"]]
            
            # Turn -1 in Body_Part_Disease to 0
            temp_df[f"{body_part}_Disease"] = temp_df[f"{body_part}_Disease"].apply(lambda x: 0 if x == -1 else x)
            
            # Split the data
            X = temp_df.drop(columns = [f"{body_part}_Disease"])
            y = temp_df[f"{body_part}_Disease"]
            
            # Use SMOTE to balance the data
            smote = SMOTENC(categorical_features = [0, 3], random_state = 42)
            
            # Fit the SMOTE model
            X, y = smote.fit_resample(X, y)
            
            # One hot encode the prediction
            values = temp_df[f"{previous_part}_Prediction"].values.reshape(-1, 1)
            pred_encoder = OneHotEncoder(categories = [[0, 1]], sparse_output = False)
            encoded_values = pred_encoder.fit_transform(values)
            
            # Drop the Left_Low_Thigh_Prediction column
            X = X.drop(f"{previous_part}_Prediction", axis = 1)
            
            # Add the encoded values to the dataframe
            X = pd.concat([X, pd.DataFrame(encoded_values, columns = ["0", "1"])], axis = 1)
            
            # One hot encode the grade with one hot encoding
            values = temp_df[f"{body_part.lower()}_grade"].values.reshape(-1, 1)
            grade_encoder = OneHotEncoder(categories = [['A', 'B', 'C', 'D', 'I']], sparse_output = False)
            encoded_values = grade_encoder.fit_transform(values)
            
            # Drop the grade column
            X = X.drop(f"{body_part.lower()}_grade", axis = 1)
            
            # Add the encoded values to the dataframe
            X = pd.concat([X, pd.DataFrame(encoded_values, columns = ['A', 'B', 'C', 'D', 'I'])], axis = 1)
            
            # Standardize the data
            scaler = StandardScaler()
            X[[f"{body_part.lower()}_pressure", f"{body_part.lower()}_index"]] = scaler.fit_transform(
                    X[[f"{body_part.lower()}_pressure", f"{body_part.lower()}_index"]])
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
            
            if TRAIN:
                # Find the best parameters
                best_params = find_best_params(X_train, y_train)
            
                # Update the dictionary
                params[body_part] = best_params
                
            else:
                best_params = params[body_part]
                
            # Create the model
            model = xgb.XGBClassifier(**best_params)
            
            # Fit the model
            model.fit(X_train, y_train)
            
            # Predict the test data
            y_pred = model.predict(X_test)
            
            # Print the accuracy
            print(f"Accuracy for {body_part}: {accuracy_score(y_test, y_pred)}")
            
            # Print the classification report
            print(f"Classification Report for {body_part}:\n{classification_report(y_test, y_pred)}")
            
            # Print the confusion matrix
            print(f"Confusion Matrix for {body_part}:\n{confusion_matrix(y_test, y_pred)}")
            
            # Create the prediction column
            df[f"{body_part}_Prediction"] = ''
            
            # Predict the data
            for index, row in df.iterrows():
                
                data = (row[
                    [f"{body_part.lower()}_grade", f"{body_part.lower()}_pressure", f"{body_part.lower()}_index",
                     f"{previous_part}_Prediction"]].values.reshape(1, -1))
                data = pd.DataFrame(data, columns = [f"{body_part.lower()}_grade",
                                                     f"{body_part.lower()}_pressure", f"{body_part.lower()}_index",
                                                     f"{previous_part}_Prediction"])
                
                # One hot encode the prediction
                prediction = pred_encoder.transform(data[f"{previous_part}_Prediction"].values.reshape(-1, 1))
                
                # Drop the Left_Low_Thigh_Prediction column
                data = data.drop(f"{previous_part}_Prediction", axis = 1)
                
                # Add the encoded values to the dataframe
                data = pd.concat([data, pd.DataFrame(prediction, columns = ["0", "1"])], axis = 1)
                
                # One hot encode the grade
                grade = grade_encoder.transform(data[f"{body_part.lower()}_grade"].values.reshape(-1, 1))
                
                # Drop the grade column
                data = data.drop(f"{body_part.lower()}_grade", axis = 1)
                
                data = pd.concat([data, pd.DataFrame(grade, columns = ['A', 'B', 'C', 'D', 'I'])], axis = 1)
                
                # Scale the data
                data[[f"{body_part.lower()}_pressure", f"{body_part.lower()}_index"]] = scaler.transform(
                        data[[f"{body_part.lower()}_pressure", f"{body_part.lower()}_index"]]
                        )
                
                # Make the prediction
                prediction = model.predict(data)
                
                # Add the prediction to the dataframe
                df.at[index, f"{body_part}_Prediction"] = prediction[0]
                
print(params)

df.to_csv("../Data/Final_Data_with_Predictions.csv", index = False)


