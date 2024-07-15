import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, auc
import matplotlib.pyplot as plt

def linear_regression_operations(df):
    columns_to_drop = st.multiselect("Select Columns for Target Variables:", df.columns)

    X = df.drop(columns=columns_to_drop)
    Y = df[columns_to_drop]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)

    st.subheader("Train-Test Split:")
    st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    st.write(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    st.subheader("Linear Regression Operations:")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write('MSE:', mse)

    st.write('Coefficients:')
    st.write(model.coef_)

    accuracy = r2_score(y_test, y_pred)
    st.write('Accuracy:', accuracy)

    st.subheader("Scatter Plot of Predicted vs Actual Values:")
    fig, ax = plt.subplots()
    ax.scatter(y_pred, y_test, alpha=0.5)
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("Actual values")
    ax.set_title("Scatter plot of the predicted values vs. the actual values")
    st.pyplot(fig)

    st.subheader("Ridge Regression on Validation Set:")
    from sklearn.linear_model import Ridge
    X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.25)
    ridge_model = Ridge(alpha=0.1)
    ridge_model.fit(X_train_val, y_train_val)
    y_pred_val = ridge_model.predict(X_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    st.write('MSE on validation set:', mse_val)

    st.subheader("Weight Visualizations and its histograms:")
    st.subheader("Bar Plot of Coefficients:")
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(model.coef_[0])), model.coef_[0])
    plt.xlabel('Feature index')
    plt.ylabel('Weight')
    plt.title('Weights of the linear regression model')
    st.pyplot()

    st.subheader("Histogram of Coefficients:")
    plt.figure(figsize=(10, 6))
    plt.hist(model.coef_[0])
    plt.xlabel('Weight')
    plt.ylabel('Count')
    plt.title('Histogram of the weights of the linear regression model')
    st.pyplot()

    st.subheader("Correlation Matrix of Features:")
    correlation_matrix = np.corrcoef(X_train, rowvar=False)
    plt.matshow(correlation_matrix)
    plt.colorbar()
    plt.xlabel('Feature index')
    plt.ylabel('Feature index')
    plt.title('Correlation matrix of the features')
    st.pyplot()


def knn_operations(df):
    st.subheader("K Nearest Neighbours Operations:")
    k_values = st.text_input("Enter K Values (comma-separated):", "3,")
    k_values = [int(k.strip()) for k in k_values.split(',')]
    mse_values = []
    columns_to_drop = st.multiselect("Select Columns for Target Variables:", df.columns)

    X = df.drop(columns=columns_to_drop)
    Y = df[columns_to_drop]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)

    st.subheader("Train-Test Split:")
    st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    st.write(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    for k in k_values:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_values.append(mse)
   
        st.write(f'MSE for K={k}: {mse}')
        plt.figure()
        plt.scatter(y_pred, y_test.to_numpy())
        plt.xlabel("Predicted values")
        plt.ylabel("Actual values")
        plt.title("Scatter plot of the predicted values vs. the actual values")
        st.pyplot(plt)

        plt.figure()
        plt.hist(y_pred - y_test.to_numpy())
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.title("Histogram of the residuals")
        st.pyplot(plt)

        plt.figure()
        corr_matrix = np.corrcoef(X.T)
        sns.heatmap(corr_matrix, annot=True)
        plt.title("Correlation Matrix Heatmap")
        st.pyplot(plt)

    plt.figure()
    plt.plot(k_values, mse_values, marker='o')
    plt.xlabel('K Values')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('KNN: MSE for Different K Values')
    st.pyplot(plt)

    return mse_values


def kmeans_operations(df):
    st.subheader("K Means Clustering Operations:")
    clusters = st.number_input("Enter the number of clusters:", min_value=1, step=1, value=3)

    kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=7)
    kmeans.fit(df)

    wcss = []
    for i in range(1, clusters+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=7)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)

    st.write("WCSS values:")
    st.write(wcss)

    plt.figure()
    plt.plot(range(1, clusters+1), wcss)
    plt.xlabel("Number of clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow curve")
    st.pyplot(plt)

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(df)
    cluster_labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_
    SSE = np.sum((df - kmeans.cluster_centers_[kmeans.predict(df)])**2)

    st.write("SSE:", SSE)



def naive_bayes_operations(df):
    st.subheader("Naive Bayes Operations:")

    columns_to_drop = st.multiselect("Select Columns for Target Variables:", df.columns)

    le = LabelEncoder()
    X = df.drop(columns=columns_to_drop)
    Y = df[columns_to_drop]
    y_encoded = le.fit_transform(Y)

    X_train_gnb, X_test_gnb, y_train_gnb, y_test_gnb = train_test_split(X, y_encoded, test_size=0.25, random_state=7)

    gnb = GaussianNB()
    gnb.fit(X_train_gnb, y_train_gnb)
    y_pred_gnb = gnb.predict(X_test_gnb)

    accuracy = gnb.score(X_test_gnb, y_test_gnb)
    st.write("Accuracy:", accuracy)

    matrix = confusion_matrix(y_test_gnb, y_pred_gnb)
    st.write("Confusion matrix:")
    st.write(matrix)

def svm_operations(df):
    st.subheader("Support Vector Machine Operations:")

    columns_to_drop = st.multiselect("Select Columns for Target Variables:", df.columns)

    X = df.drop(columns=columns_to_drop)
    Y = df[columns_to_drop]

    st.subheader("Train-Test Split:")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)
    y_test_np = y_test.to_numpy()
    st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    st.write(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    svm_regressor = SVR(kernel='linear')
    multi_output_svm = MultiOutputRegressor(svm_regressor)
    multi_output_svm.fit(X_train, y_train)
    y_pred = multi_output_svm.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

    for i in range(y_test.shape[1]):
        plt.figure(figsize=(8, 6))
        plt.scatter(range(len(y_test)), y_test.iloc[:, i], color='black', label='Actual')
        plt.scatter(range(len(y_pred)), y_pred[:, i], color='red', label='Predicted')
        plt.title(f'Actual vs Predicted for {y_test.columns[i]}')
        plt.xlabel('Data Points')
        plt.ylabel(f'{y_test.columns[i]}')
        plt.legend()
        st.pyplot(plt)

    threshold = 0.5
    y_test_binary = np.array([1 if val > threshold else 0 for val in y_test.values.flatten()])
    y_pred_binary = np.array([1 if val > threshold else 0 for val in y_pred.flatten()])
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

    plt.hist(y_pred - y_test_np)
    plt.xlabel("Residuals_svm")
    plt.ylabel("Frequency")
    plt.title("Histogram of the residuals in SVM")
    st.pyplot(plt)


def random_forest_operations(df):
    st.subheader("Random Forest Operations:")
    columns_to_drop = st.multiselect("Select Columns for Target Variables:", df.columns)

    X = df.drop(columns=columns_to_drop)
    Y = df[columns_to_drop]

    st.subheader("Train-Test Split:")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)
    y_test_np = y_test.to_numpy()
    st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    st.write(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

    for i in range(y_test.shape[1]):
        plt.scatter(y_test.iloc[:, i], y_pred[:, i], color='black', alpha=0.5)
        plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'k--', lw=2)
        plt.xlabel(f'Actual {y_test.columns[i]}')
        plt.ylabel(f'Predicted {y_test.columns[i]}')
        plt.title(f'Scatter plot of Actual vs Predicted for {y_test.columns[i]}')
        plt.show()
        st.pyplot(plt)

    plt.hist(y_pred - y_test_np)
    plt.xlabel("Residuals_rf")
    plt.ylabel("Frequency")
    plt.title("Histogram of the residuals in Random Forest")
    st.pyplot(plt)

    threshold = 0.5
    y_test_binary = np.array([1 if val > threshold else 0 for val in y_test.values.flatten()])
    y_pred_binary = np.array([1 if val > threshold else 0 for val in y_pred.flatten()])
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    st.pyplot(plt)

def decision_tree_operations(df):
    st.subheader("Decision Trees Operations:")
    st.subheader("Random Forest Operations:")
    columns_to_drop = st.multiselect("Select Columns for Target Variables:", df.columns)

    X = df.drop(columns=columns_to_drop)
    Y = df[columns_to_drop]

    st.subheader("Train-Test Split:")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)
    y_test_np = y_test.to_numpy()
    st.write(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    st.write(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    tree_model = DecisionTreeRegressor()
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

    for i in range(y_test.shape[1]):
        plt.scatter(y_test.iloc[:, i], y_pred[:, i], color='black', alpha=0.5)
        plt.plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'k--', lw=2)
        plt.xlabel(f'Actual {y_test.columns[i]}')
        plt.ylabel(f'Predicted {y_test.columns[i]}')
        plt.title(f'Scatter plot of Actual vs Predicted for {y_test.columns[i]}')
        plt.show()
        st.pyplot(plt)
    
    plt.hist(y_pred - y_test_np)
    plt.xlabel("Residuals_dt")
    plt.ylabel("Frequency")
    plt.title("Histogram of the residuals in Decision Tree")
    plt.show()
    st.pyplot(plt)

    threshold = 0.5
    y_test_binary = np.array([1 if val > threshold else 0 for val in y_test.values.flatten()])
    y_pred_binary = np.array([1 if val > threshold else 0 for val in y_pred.flatten()])

    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_binary)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()
    st.pyplot(plt)

def detect_and_remove_outliers(df, columns_for_outliers, threshold=1.5):
    outliers = {}
    df_cleaned = df.copy()

    for col in columns_for_outliers:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        outliers[col] = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]  # Removal of outliers

    return df_cleaned, outliers

def convert_columns_to_specific_types(df, columns_for_float, columns_for_int):

    for col in columns_for_float:
        df[col] = df[col].astype(float)

    for col in columns_for_int:
        df[col] = df[col].astype(int)

    return df

def label_encode_column(df, column):
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    return df

def drop_specific_columns(df, column):
    df.drop(columns=column, inplace=True)
    return df

def column_scaling(df, column):
    scaler = MinMaxScaler()
    for i in column:
        df[i] = scaler.fit_transform(df[i].values.reshape(-1,1))
    return df
    

def main():
    st.title("DataFrame Operations")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
      
        df = pd.read_csv(uploaded_file)

        st.subheader("DataFrame Preview:")
        st.write(df.head())
        st.write(df.describe())
        st.write(df.shape)
        st.write(df.drop_duplicates())
        st.write(df.dropna())

        # columns_for_outliers = ['temperature_celsius', 'temperature_fahrenheit', 'air_quality_us-epa-index',
        #                         'air_quality_gb-defra-index', 'air_quality_Carbon_Monoxide', 'air_quality_Ozone',
        #                         'air_quality_Nitrogen_dioxide', 'air_quality_Sulphur_dioxide', 'air_quality_PM2.5',
        #                         'air_quality_PM10', 'wind_mph', 'wind_kph', 'pressure_mb', 'pressure_in']

        columns_for_outliers = st.multiselect("Select Columns for outliers:", df.columns)

        df_cleaned, outliers = detect_and_remove_outliers(df, columns_for_outliers)

        columns_for_float = st.multiselect("Select Columns for converting into Float:", df.columns)
        columns_for_int = st.multiselect("Select Columns for converting into Integer:", df.columns)
        df_cleaned = convert_columns_to_specific_types(df_cleaned, columns_for_float, columns_for_int)

        # columns_for_label_encoding = st.multiselect("Select Columns for Label Encoding:", df.columns)
        # df_cleaned = label_encode_column(df_cleaned, columns_for_label_encoding)

        if st.button("Perform Optional Column Operations"):
            # Strip leading and trailing spaces from column names
            df_cleaned.columns = [col.strip() for col in df_cleaned.columns]
            
            # Lowercase column names
            df_cleaned.columns = df_cleaned.columns.str.lower()

        specific_columns_to_drop = st.multiselect("Select Columns for dropping specifically:", df.columns)
        df_cleaned = drop_specific_columns(df_cleaned, specific_columns_to_drop)

        columns_to_scale = st.multiselect("Select Columns to scale:", df.columns)
        df_cleaned = column_scaling(df_cleaned, columns_to_scale)

        st.subheader("Cleaned DataFrame Preview:")
        st.write(df_cleaned.head())
        st.write(df_cleaned.describe())
        st.write(df_cleaned.shape)
        # # Operations on DataFrame
        # columns_for_model = ['air_quality_us-epa-index', 'dewpoint', 'windchill']

        # # Drop columns for model
        # X = df.drop(columns=columns_for_model)

        # # Select columns for model
        # Y = df[columns_for_model]

        # # Split data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=7)

        # Dropdown for machine learning operations
        operation_options = [
            "Select Operation",
            "Linear Regression",
            "K Nearest Neighbours",
            "K Means Clustering",
            "Naive Bayes",
            "Support vector machine",
            "Random forest",
            "Decision Trees"
        ]
        selected_operation = st.selectbox("Select Machine Learning Operation", operation_options)

        # Perform selected operation
        if selected_operation == "Linear Regression":
            linear_regression_operations(df)
        elif selected_operation == "K Nearest Neighbours":
            knn_operations(df)
        elif selected_operation == "K Means Clustering":
            kmeans_operations(df)
        elif selected_operation == "Naive Bayes":
            naive_bayes_operations(df)
        elif selected_operation == "Support vector machine":
            svm_operations(df)
        elif selected_operation == "Random forest":
            random_forest_operations(df)
        elif selected_operation == "Decision Trees":
            decision_tree_operations(df)

if __name__ == "__main__":
    main()
