"""
eda.py
Generates all EDA statistics for the frontend charts.
Returns pure JSON-serialisable Python dicts — no matplotlib needed.
"""

from collections import Counter


def get_distribution(df_raw):
    """Value counts for every categorical column."""
    result = {}
    for col in df_raw.columns:
        if col == 'Matric Number':
            continue
        counts = df_raw[col].astype(str).value_counts()
        result[col] = {
            'labels': counts.index.tolist(),
            'values': counts.values.tolist(),
        }
    return result


def get_gender_breakdown(df_raw):
    male = int((df_raw['Gender'] == 'Male').sum())
    female = int((df_raw['Gender'] == 'Female').sum())
    return {'Male': male, 'Female': female, 'total': male + female}


def get_level_breakdown(df_raw):
    vc = df_raw['Current academic level'].value_counts().sort_index()
    return {str(k): int(v) for k, v in vc.items()}


def get_cross_tab(df_raw, col_a, col_b):
    """Cross-tabulation of two columns for grouped bar charts."""
    pivot = df_raw.groupby([col_a, col_b]).size().unstack(fill_value=0)
    return {
        'groups': pivot.index.astype(str).tolist(),
        'series': {
            str(col): pivot[col].tolist()
            for col in pivot.columns
        }
    }


def get_descriptive_stats(df_enc):
    """Mean, std, min, max for encoded features."""
    feature_cols = [c for c in df_enc.columns if c != 'Matric Number']
    desc = df_enc[feature_cols].describe().round(3)
    return {
        col: {
            'mean': float(desc[col]['mean']),
            'std':  float(desc[col]['std']),
            'min':  float(desc[col]['min']),
            'max':  float(desc[col]['max']),
        }
        for col in desc.columns
    }


def get_correlation_matrix(df_enc):
    """Pearson correlation matrix for heatmap."""
    feature_cols = [c for c in df_enc.columns if c != 'Matric Number']
    short = {
        'Gender': 'Gender',
        'Current academic level': 'Acad. Level',
        'Preferred living environment': 'Living Env.',
        'Organize personal space frequency': 'Org. Freq.',
        'Productive environment': 'Prod. Env.',
        'Atmosphere created': 'Atmosphere',
        'Invite friends frequency': 'Invite Freq.',
        'Comfortable with roommate bringing guests': 'Guest OK',
        'Sleep lights preference': 'Lights',
        'Fan speed preference': 'Fan Speed',
        'Preferred study time': 'Study Time',
        'Okay with roommate studying late night': 'Late Study OK',
    }
    corr = df_enc[feature_cols].corr().round(3)
    cols = [short.get(c, c) for c in corr.columns]
    return {
        'columns': cols,
        'matrix': corr.values.tolist(),
    }


def run_eda(df_raw, df_enc):

    # Gender Distribution
    gender_counts = df_raw['Gender'].value_counts()
    genderDistribution = [
        {"name": str(k), "value": int(v)}
        for k, v in gender_counts.items()
    ]

    # Study Habits
    study_counts = df_raw['Preferred study time'].value_counts()
    studyHabits = [
        {"habit": str(k), "count": int(v)}
        for k, v in study_counts.items()
    ]

    # Sleep Patterns
    sleep_counts = df_raw['Sleep lights preference'].value_counts()
    sleepPatterns = [
        {"pattern": str(k), "count": int(v)}
        for k, v in sleep_counts.items()
    ]

    # Cleanliness Levels
    clean_counts = df_raw['Organize personal space frequency'].value_counts()
    cleanlinessLevels = [
        {"level": str(k), "count": int(v)}
        for k, v in clean_counts.items()
    ]

    # Academic Levels (used as Age Distribution)
    level_counts = df_raw['Current academic level'].value_counts().sort_index()
    ageDistribution = [
        {"range": str(k), "count": int(v)}
        for k, v in level_counts.items()
    ]

    return {
        # charts for frontend
        "ageDistribution": ageDistribution,
        "genderDistribution": genderDistribution,
        "sleepPatterns": sleepPatterns,
        "studyHabits": studyHabits,
        "cleanlinessLevels": cleanlinessLevels,

        # existing analytics (keep them)
        'distribution': get_distribution(df_raw),
        'gender': get_gender_breakdown(df_raw),
        'levels': get_level_breakdown(df_raw),
        'gender_vs_study': get_cross_tab(df_raw, 'Gender', 'Preferred study time'),
        'gender_vs_atmosphere': get_cross_tab(df_raw, 'Gender', 'Atmosphere created'),
        'level_vs_study': get_cross_tab(df_raw, 'Current academic level', 'Preferred study time'),
        'level_vs_lights': get_cross_tab(df_raw, 'Current academic level', 'Sleep lights preference'),
        'descriptive_stats': get_descriptive_stats(df_enc),
        'correlation': get_correlation_matrix(df_enc),
    }
