from data.load_dataset import load_breast_cancer_data

print("=== Testing Data Load ===")
try:
    data = load_breast_cancer_data()
    print("SUCCESS! Data loaded:")
    print("- Shape:", data.shape)
    print("- Columns:", data.columns.tolist())
    print("- Target counts:\n", data['diagnosis'].value_counts())
except Exception as e:
    print("FAILED:", str(e))