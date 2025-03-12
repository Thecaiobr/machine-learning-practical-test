import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import plotly.express as px

def load_dataset(pickleFile):
    with open(pickleFile, 'rb') as file:
        data = pickle.load(file)
    return data

def flatten_data(data):
    flattened_data = []
    for syndrome_id, subject in data.items():
        for subject_id, image in subject.items():
            for image_id, embedding in image.items():
                entry_data = ({
                    'syndrome_id': syndrome_id,
                    'subject_id': subject_id,
                    'image_id': image_id,
                    'embedding': embedding
                })
                flattened_data.append(entry_data)
    return pd.DataFrame(flattened_data)

def preprocess_data(df):
    if df.isnull().values.any():
        df = df.dropna()
        df = df.drop_duplicates()
        
    
    embeddings = np.vstack(df['embedding'].values)
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)
    df['embedding'] = list(embeddings_normalized)

    syndrome_counts = df['syndrome_id'].value_counts()
    num_syndromes = syndrome_counts.count()
    images_per_syndrome_count = syndrome_counts.values

    print(f"Syndromes distribution:\n{syndrome_counts}")
    print(f"Number of syndromes: {num_syndromes}")
    print(f"Images per syndrome: {images_per_syndrome_count}")
    
    subject_per_syndrome_count = df.groupby('syndrome_id')['subject_id'].nunique()
    print(f"Subjects per syndrome: {subject_per_syndrome_count}")
    
    return df



def data_visualization(embeddings, syndrome_ids):
    tsne = TSNE(n_components=2, n_iter=1500, perplexity=40)
    embeddings_2d = tsne.fit_transform(embeddings)

    df = pd.DataFrame(data=embeddings_2d, columns=['x', 'y'])
    df['syndrome_id'] = syndrome_ids

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='x', y='y',
        hue='syndrome_id',
        palette=sns.color_palette("hsv", len(df['syndrome_id'].unique())),
        data=df,
        legend="full",
        alpha=0.6
    )
    plt.title('t-SNE Embeddings Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Syndrome ID')
    plt.show()
    
def main():
    data = load_dataset('../data/mini_gm_public_v0.1.p')
    flattened_df = flatten_data(data)
    processed_df = preprocess_data(flattened_df)
    
    
    print(processed_df.head())
    print(processed_df.tail())

    embeddings = np.vstack(processed_df['embedding'].values)
    syndrome_ids = processed_df['syndrome_id'].values
    data_visualization(embeddings, syndrome_ids)
    
    return processed_df

if __name__ == "__main__":
    main()