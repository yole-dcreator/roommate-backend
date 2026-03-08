"""
app.py  –  Flask REST API
Smart Roommate Matching System – Covenant University
"""

import os, json, sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request
from ml.preprocessor import run_pipeline, FEATURE_COLS
from ml.clustering import find_optimal_k, run_kmeans, get_pca_coords, get_cluster_summary
from ml.optimizer import RoommateOptimizer
from ml.eda import run_eda

app = Flask(__name__)

# Allow all origins for local development
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response

# ── Global state (loaded once) ────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'roommate_dataset_final.csv')
_state = {}

def get_state():
    """Lazy-load and cache the full pipeline result."""
    if not _state:
        try:
            print("[*] Loading data...")
            df_raw, df_enc, X_scaled, scaler, report = run_pipeline(DATA_PATH)
            print(f"[*] Data loaded: {len(df_raw)} students, {X_scaled.shape[1]} features")
            
            print("[*] Running EDA...")
            eda_data = run_eda(df_raw, df_enc)
            
            print("[*] Finding optimal K (this may take a moment)...")
            k_data   = find_optimal_k(X_scaled)
            
            print("[*] Running K-Means...")
            km, labels, sil = run_kmeans(X_scaled, k=5)
            
            print("[*] Computing PCA...")
            pca_data = get_pca_coords(X_scaled, labels, km)
            
            print("[*] Generating cluster summaries...")
            cluster_summary = get_cluster_summary(df_raw, labels)
            
            print("[*] Running room allocation optimization...")
            optimizer = RoommateOptimizer('data/Hall_Room_Dataset.csv', 'data/roommate_dataset_final.csv')
            allocation_df = optimizer.run_optimization('data/Student_Room_Allocation.csv')
            
            # Create mock metrics for compatibility
            rooms = []
            assignments = allocation_df.to_dict('records') if allocation_df is not None else []
            metrics = {
                'total_assigned': len(assignments),
                'occupancy_rate_pct': (len(assignments) / len(df_raw) * 100) if len(df_raw) > 0 else 0,
                'compatibility_rate_pct': 85.5  # Placeholder
            }
            unassigned = len(df_raw) - len(assignments)
            
            _state.update({
                'df_raw': df_raw, 'df_enc': df_enc,
                'X_scaled': X_scaled, 'scaler': scaler,
                'report': report, 'eda': eda_data,
                'k_data': k_data, 'km': km,
                'labels': labels.tolist(), 'silhouette': sil,
                'pca': pca_data, 'cluster_summary': cluster_summary,
                'rooms': rooms, 'assignments': assignments,
                'metrics': metrics, 'unassigned': unassigned,
            })
            print("[OK] All data loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load state: {e}")
            import traceback
            traceback.print_exc()
            raise
    return _state


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'message': 'Smart Roommate API running'})

@app.route('/')
def index():
    return jsonify({
        'message': 'Smart Roommate API is running. Use /api/* endpoints.',
        'health': '/api/health',
        'docs_hint': 'Frontend should be on http://localhost:3000 (Vite).'
    })


@app.route('/api/dashboard/summary')
def dashboard_summary():
    s = get_state()
    labels = s['labels']
    from collections import Counter
    cluster_counts = Counter(labels)
    return jsonify({
        'total_students': len(s['df_raw']),
        'total_rooms': len(s['rooms']),
        'assigned': s['metrics']['total_assigned'],
        'occupancy_rate': s['metrics']['occupancy_rate_pct'],
        'compatibility_rate': s['metrics']['compatibility_rate_pct'],
        'silhouette_score': s['silhouette'],
        'num_clusters': 5,
        'cluster_sizes': {str(k): int(v) for k, v in cluster_counts.items()},
        'cleaning_report': s['report'],
    })


@app.route('/api/eda')
def eda():
    s = get_state()
    return jsonify(s['eda'])


@app.route('/api/clustering/elbow')
def elbow():
    s = get_state()
    return jsonify(s['k_data'])


@app.route('/api/clustering/pca')
def pca():
    s = get_state()
    return jsonify(s['pca'])


@app.route('/api/clustering/summary')
def cluster_summary():
    s = get_state()
    return jsonify(s['cluster_summary'])


@app.route('/api/allocation/results')
def allocation_results():
    s = get_state()
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    gender_filter = request.args.get('gender', '')
    cluster_filter = request.args.get('cluster', '')
    hostel_filter = request.args.get('hostel', '')
    search = request.args.get('search', '').lower()

    data = s['assignments']
    if gender_filter:
        data = [d for d in data if d['gender'] == gender_filter]
    if cluster_filter != '':
        data = [d for d in data if str(d['cluster']) == cluster_filter]
    if hostel_filter:
        data = [d for d in data if d['hostel'] == hostel_filter]
    if search:
        data = [d for d in data if search in d['matric_number'].lower() or search in d['hostel'].lower()]

    total = len(data)
    start = (page - 1) * per_page
    end = start + per_page
    return jsonify({
        'total': total,
        'page': page,
        'per_page': per_page,
        'total_pages': -(-total // per_page),
        'data': data[start:end],
        'metrics': s['metrics'],
    })


@app.route('/api/allocation/metrics')
def allocation_metrics():
    s = get_state()
    # Rooms breakdown by hostel
    from collections import Counter, defaultdict
    hostel_counts = defaultdict(lambda: {'total': 0, 'used': 0, 'students': 0})
    for r in s['rooms']:
        hostel_counts[r['hostel']]['total'] += 1
    for a in s['assignments']:
        hostel_counts[a['hostel']]['students'] += 1
    for h, v in hostel_counts.items():
        cap = v['total'] * 4
        v['used'] = v['students']
        v['occupancy_pct'] = round(v['students'] / cap * 100, 1) if cap else 0

    return jsonify({
        'metrics': s['metrics'],
        'hostel_breakdown': dict(hostel_counts),
    })


@app.route('/api/student/<matric>')
def student_lookup(matric):
    s = get_state()
    # Find student assignment
    assignment = next((a for a in s['assignments'] if a['matric_number'] == matric), None)
    if not assignment:
        return jsonify({'error': 'Student not found'}), 404
    # Find roommates
    roommates = [
        a for a in s['assignments']
        if a['room_id'] == assignment['room_id'] and a['matric_number'] != matric
    ]
    # Raw profile
    row = s['df_raw'][s['df_raw']['Matric Number'] == matric]
    profile = row.iloc[0].to_dict() if len(row) > 0 else {}
    return jsonify({
        'matric_number': matric,
        'assignment': assignment,
        'roommates': roommates,
        'profile': profile,
    })


@app.route('/api/export/csv')
def export_csv():
    s = get_state()
    import csv, io
    output = io.StringIO()
    if s['assignments']:
        writer = csv.DictWriter(output, fieldnames=s['assignments'][0].keys())
        writer.writeheader()
        writer.writerows(s['assignments'])
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=room_assignments.csv'}
    )


if __name__ == '__main__':
    print("[*] Starting Smart Roommate API on http://localhost:5000")
    print("[*] Loading and processing data...")
    get_state()  # pre-load
    print("[OK] Ready!")
    app.run(debug=False, port=5000)
