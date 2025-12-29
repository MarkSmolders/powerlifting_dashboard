import pandas as pd

def prepare_improvement_dataset_raw(dataset_path, redundant_features):
    """
    This function prepares the dataset from Openpowerlifting for the competition total improvement prediction model by dropping
    redundant features, filtering for athletes that have competed at least three times and creates new features.
    """
    df = pd.read_csv(dataset_path)
    df = df[(df['Equipment'] == 'Raw') & (df['Event'] == 'SBD') & (df['Tested'] == 'Yes')]
    df.drop(redundant_features, axis=1)
    
    df['days_since_last_comp'] = df.groupby('Name')['Date'].diff() * -1
    df['days_since_last_comp'] = df['days_since_last_comp'].fillna(pd.Timedelta(days=0))
    return df 



# Function for Equipped lifting can be added below