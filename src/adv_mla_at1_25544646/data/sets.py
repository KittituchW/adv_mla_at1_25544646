from sklearn.model_selection import train_test_split


def split_dataset(df, target_col, test_size=0.2, val_size=0.25, random_state=42):

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Step 1: Split into train+eval and test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Step 2: Split train+eval into train and eval
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp
    )

    # Print summary
    print(f"Train set shape: {X_train.shape}")
    print(f"Eval set shape:  {X_val.shape}")
    print(f"Test set shape:  {X_test.shape}")

    print(f"Draft rate in train: {y_train.mean():.3f}")
    print(f"Draft rate in eval:  {y_val.mean():.3f}")
    print(f"Draft rate in test:  {y_test.mean():.3f}")

    return X_train, X_val, X_test, y_train, y_val, y_test
