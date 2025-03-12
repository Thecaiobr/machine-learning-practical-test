from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import top_k_accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import roc_curve,auc
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(file_path):
    with open(file_path, 'rb') as file:
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
    print(f"Distribuição das síndromes:\n{syndrome_counts}")
    
    return df

def prepare_data(df):
    X = np.vstack(df['embedding'].values)
    y = df['syndrome_id'].values
    return X, y


def knn_classification(X, y, k_neighbors, distance_metric='euclidean'):
    results = {}
    roc_curves = {}
    kf = KFold(n_splits=10, shuffle=True, random_state=40)
    for k in k_neighbors:
        roc_auc_scores = []
        f1_scores = []
        topK_accuracies = []
        tprs = []
        mean_fpr = np.linspace(0, 1, 100)
        knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        for train, test in kf.split(X):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            y_proba = knn.predict_proba(X_test)
            
            roc_auc_scores.append(roc_auc_score(y_test, y_proba, multi_class='ovr'))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            topK_accuracies.append(top_k_accuracy_score(y_test, y_proba, k=3))
            
            y_test_binarized = label_binarize(y_test, classes=np.unique(y))
            fpr, tpr, _ = roc_curve(y_test_binarized.ravel(), y_proba.ravel())
            tprs.append(np.interp(mean_fpr, fpr, tpr))
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_auc = auc(mean_fpr, mean_tpr)
        
        results[k] = {
            'roc_auc': np.mean(roc_auc_scores),
            'f1_score': np.mean(f1_scores),
            'topK_accuracy': np.mean(topK_accuracies)
        }
        roc_curves[k] = (mean_fpr, mean_tpr, mean_auc)
    
    return results, roc_curves

    
def plot_best_roc_curve(roc_curves_euclidean, roc_curves_cosine, best_k_euclidean, best_k_cosine):
    plt.figure(figsize=(10, 8))
    
    mean_fpr_euclidean, mean_tpr_euclidean, mean_auc_euclidean = roc_curves_euclidean[best_k_euclidean]
    plt.plot(mean_fpr_euclidean, mean_tpr_euclidean, label=f'Euclidean k={best_k_euclidean} (AUC = {mean_auc_euclidean:.2f})')
    
    mean_fpr_cosine, mean_tpr_cosine, mean_auc_cosine = roc_curves_cosine[best_k_cosine]
    plt.plot(mean_fpr_cosine, mean_tpr_cosine, label=f'Cosine k={best_k_cosine} (AUC = {mean_auc_cosine:.2f})')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Best ROC Curves for KNN with Euclidean and Cosine Distance Metrics')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def createMetricsTables(result_euclidian_distance, result_cosine_distance):
    table = pd.DataFrame(columns=["Distance Metric", "k-neighbors", "ROC-AUC", "F1-SCORE", "TOP-K-ACCURACY"])
    for k in result_cosine_distance:
        table = pd.concat([table, pd.DataFrame(
            [
                {
                    'Distance Metric' : 'Cosine Distance',
                    'k-neighbors': k,
                    'ROC-AUC': result_cosine_distance[k]['roc_auc'],
                    'F1-SCORE': result_cosine_distance[k]['f1_score'],
                    'TOP-K-ACCURACY': result_cosine_distance[k]['topK_accuracy'],
                }
            ]
        )])
        
    for k in result_euclidian_distance:
        table = pd.concat([table, pd.DataFrame(
            [
                {
                    'Distance Metric' : 'Euclidian Distance',
                    'k-neighbors': k,
                    'ROC-AUC': result_euclidian_distance[k]['roc_auc'],
                    'F1-SCORE': result_euclidian_distance[k]['f1_score'],
                    'TOP-K-ACCURACY': result_euclidian_distance[k]['topK_accuracy'],
                }
            ]
        )])
    
    plt.figure(figsize=(12, 8))
    plt.axis('off')

    tbl = plt.table(
        cellText=table.values,
        colLabels=table.columns,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)
    plt.title("KNN Evaluation Metrics", fontsize=14, pad=20)
    plt.show()
    
    return table


def main():
    data = load_dataset('../data/mini_gm_public_v0.1.p')
    flattened_df = flatten_data(data)
    processed_df = preprocess_data(flattened_df)
    X, y = prepare_data(processed_df)
    
    k_neighbors = range(1, 16)
    results_euclidean_distance, roc_auc_curve_euclidian = knn_classification(X, y, k_neighbors, distance_metric='euclidean')
    results_cosine_distance, roc_auc_curve_cosine = knn_classification(X, y, k_neighbors, distance_metric='cosine')
    
    best_k_euclidean = max(results_euclidean_distance, key=lambda k: results_euclidean_distance[k]['roc_auc'])
    best_k_cosine = max(results_cosine_distance, key=lambda k: results_cosine_distance[k]['roc_auc'])
    
    print(f'Best K for euclidean distance: {best_k_euclidean}')
    print(f'Best K for cosine distance: {best_k_cosine}')
    plot_best_roc_curve(roc_auc_curve_euclidian, roc_auc_curve_cosine, best_k_euclidean, best_k_cosine)
    
    results_table = createMetricsTables(results_euclidean_distance, results_cosine_distance)
    print(results_table)

if __name__ == "__main__":
    main()