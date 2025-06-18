import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(
    df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42
) -> tuple[Pipeline,]:
    """Preprocess a pandas dataframe into training data

    Args:
        df (pd.DataFrame): DataFrame to be processed.
        target_column (str): Name of the target column.
        test_size (float, optional): Size of test partition, stratified by @target_column. Defaults to 0.2.
        random_state (int, optional): Random state for sampling. Defaults to 42.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    df[numerical_cols] = df[numerical_cols].astype(float)

    # Preprocessing for numeric data
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return preprocessor, X_train, X_test, y_train, y_test
