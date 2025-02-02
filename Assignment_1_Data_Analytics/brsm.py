import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# Ensure required packages are installed
try:
    import micropip
    await micropip.install(["numpy", "pandas", "scipy", "matplotlib"])
except ImportError:
    pass  # micropip is not available in all environments

# Step 1: Simulate Patient Data
np.random.seed(42)
n_patients = 400

data = pd.DataFrame({
    'patient_id': np.arange(n_patients),
    'baseline_pain': np.random.randint(0, 10, n_patients),
    'baseline_urgency': np.random.randint(0, 10, n_patients),
    'baseline_frequency': np.random.randint(0, 10, n_patients),
    'treatment_time': np.random.choice([3, 6, 9, 12, None], size=n_patients, p=[0.2, 0.2, 0.2, 0.2, 0.2])
})
data['treated'] = data['treatment_time'].notna().astype(int)

# Step 2: Matching using Hungarian Algorithm (Approximating Integer Programming)
def mahalanobis_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

treated = data[data['treated'] == 1].reset_index(drop=True)
controls = data[data['treated'] == 0].reset_index(drop=True)

if len(treated) > 0 and len(controls) > 0:
    distance_matrix = np.zeros((len(treated), len(controls)))
    for i, (_, t_row) in enumerate(treated.iterrows()):
        for j, (_, c_row) in enumerate(controls.iterrows()):
            distance_matrix[i, j] = mahalanobis_distance(
                t_row[['baseline_pain', 'baseline_urgency', 'baseline_frequency']].values,
                c_row[['baseline_pain', 'baseline_urgency', 'baseline_frequency']].values)

    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    matched_pairs = pd.DataFrame({
        'treated_id': treated.iloc[row_ind]['patient_id'].values,
        'control_id': controls.iloc[col_ind]['patient_id'].values,
        'distance': distance_matrix[row_ind, col_ind]
    })
else:
    matched_pairs = pd.DataFrame(columns=['treated_id', 'control_id', 'distance'])

# Step 3: Simulating Post-Treatment Data
data['post_treatment_pain'] = data['baseline_pain'] - np.random.randint(0, 3, n_patients)
data['post_treatment_urgency'] = data['baseline_urgency'] - np.random.randint(0, 3, n_patients)
data['post_treatment_frequency'] = data['baseline_frequency'] - np.random.randint(0, 3, n_patients)

# Step 4: Statistical Analysis
if not matched_pairs.empty:
    treated_outcomes = data.set_index('patient_id').loc[matched_pairs['treated_id'], ['post_treatment_pain', 'post_treatment_urgency', 'post_treatment_frequency']]
    control_outcomes = data.set_index('patient_id').loc[matched_pairs['control_id'], ['post_treatment_pain', 'post_treatment_urgency', 'post_treatment_frequency']]

    for col in treated_outcomes.columns:
        stat, p = wilcoxon(treated_outcomes[col], control_outcomes[col])
        print(f'Wilcoxon test for {col}: statistic={stat}, p-value={p}')
else:
    print("No matched pairs available for statistical analysis.")

# Step 5: Visualization
plt.figure(figsize=(12, 5))
if not matched_pairs.empty:
    for i, col in enumerate(['post_treatment_pain', 'post_treatment_urgency', 'post_treatment_frequency']):
        plt.subplot(1, 3, i+1)
        plt.boxplot([treated_outcomes[col], control_outcomes[col]], labels=['Treated', 'Control'])
        plt.title(col.replace('_', ' ').title())
    plt.show()
else:
    print("No matched pairs available for visualization.")