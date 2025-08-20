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

def convert_date_height_inplace(df, colname):
    """
    Converts a date-like height column (e.g., '11-May', 'Jun-00') directly into numeric inches,
    replacing the original column in place.

    Parameters:
    - df: pd.DataFrame
    - colname: str, column to convert

    Returns:
    - df: updated DataFrame (column replaced with total inches)
    """
    MONTH_TO_FEET = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }

    def to_inches(value):
        if pd.isna(value):
            return np.nan
        s = str(value).strip().lower()
        if s in {"ht", "height", ""}:
            return np.nan

        m1 = re.match(r"^(\d{1,2})[-/ ]([a-z]{3,})$", s)
        m2 = re.match(r"^([a-z]{3,})[-/ ](\d{1,2})$", s)

        feet, inches = None, None
        if m1:
            inches = int(m1.group(1))
            month = m1.group(2)
        elif m2:
            month = m2.group(1)
            inches = int(m2.group(2))
        else:
            return np.nan

        feet = MONTH_TO_FEET.get(month[:3])
        if feet is None or inches is None:
            return np.nan

        return feet * 12 + inches

    df[colname] = df[colname].apply(to_inches)
    return df


