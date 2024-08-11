import pandas as pd

def prepare_dataset(csv_path):
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Combine personality and question into a single prompt
    df['prompt'] = df.apply(lambda row: f"Answer as if you are {row['Target Personality']} about {row['Edit Topic']}: {row['Question']}", axis=1)

    # Select the necessary columns
    df = df[['prompt', 'Answer']]
    
    return df

if __name__ == "__main__":
    csv_path = 'updated_personality_data_train.csv'
    df = prepare_dataset(csv_path)
    df.to_csv('prepared_dataset.csv', index=False)
