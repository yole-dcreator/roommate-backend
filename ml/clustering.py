"""
clustering.py
K-Means clustering with Elbow Method and Silhouette Score.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

CLUSTER_LABELS = {
    0: 'Cluster A',
    1: 'Cluster B',
    2: 'Cluster C',
    3: 'Cluster D',
    4: 'Cluster E',
}

CLUSTER_PROFILES = {
    0: {'name': 'Cluster A', 'color': '#003087', 'description': 'Mixed levels, energetic, afternoon/morning study'},
    1: {'name': 'Cluster B', 'color': '#C9A84C', 'description': 'Predominantly female, lower levels, late-night study, very calm'},
    2: {'name': 'Cluster C', 'color': '#40916C', 'description': 'Predominantly male, upper levels, energetic atmosphere'},
    3: {'name': 'Cluster D', 'color': '#C0392B', 'description': 'Predominantly male, very calm, late-night/evening study'},
    4: {'name': 'Cluster E', 'color': '#8E44AD', 'description': 'Predominantly female, upper levels, morning study'},
}


def find_optimal_k(X_scaled, k_range=range(2, 11)):
    """Compute inertia and silhouette scores for each K."""
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        sil_scores.append(float(silhouette_score(X_scaled, km.labels_)))
    best_k = list(k_range)[sil_scores.index(max(sil_scores))]
    return {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': [round(s, 4) for s in sil_scores],
        'suggested_k': best_k,
        'best_silhouette': round(max(sil_scores), 4),
    }


def run_kmeans(X_scaled, k=5):
    """Fit K-Means with chosen K, return labels and model."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = float(silhouette_score(X_scaled, labels))
    return km, labels, round(sil, 4)


def get_pca_coords(X_scaled, labels, km):
    """Reduce to 2D with PCA for scatter plot."""
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(km.cluster_centers_)
    var = pca.explained_variance_ratio_

    points = []
    for i in range(len(X_pca)):
        points.append({
            'x': round(float(X_pca[i, 0]), 4),
            'y': round(float(X_pca[i, 1]), 4),
            'cluster': int(labels[i]),
        })
    centroids = [
        {'x': round(float(c[0]), 4), 'y': round(float(c[1]), 4), 'cluster': i}
        for i, c in enumerate(centroids_pca)
    ]
    return {
        'points': points,
        'centroids': centroids,
        'variance_explained': [round(float(v), 4) for v in var],
    }


def get_cluster_summary(df_raw, labels):
    """Return per-cluster statistics."""
    df = df_raw.copy()
    df['Cluster'] = labels
    summary = []
    for k in sorted(df['Cluster'].unique()):
        sub = df[df['Cluster'] == k]
        profile = CLUSTER_PROFILES.get(k, {})
        summary.append({
            'cluster_id': int(k),
            'name': profile.get('name', f'Cluster {k}'),
            'color': profile.get('color', '#999'),
            'description': profile.get('description', ''),
            'size': int(len(sub)),
            'male_count': int((sub['Gender'] == 'Male').sum()),
            'female_count': int((sub['Gender'] == 'Female').sum()),
            'levels': sub['Current academic level'].value_counts().sort_index().to_dict(),
            'top_study_time': sub['Preferred study time'].mode()[0],
            'top_atmosphere': sub['Atmosphere created'].mode()[0],
            'top_living_env': sub['Preferred living environment'].mode()[0],
        })
    return summary
