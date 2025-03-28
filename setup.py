import os

structure = {
    'data': ['sample.csv'],
    'config': ['params.yaml'],
    'automl': [
        '__init__.py',
        'preprocess.py',
        'feature_eng.py',
        'model_selection.py',
        'hyperparam_tuning.py',
        'evaluate.py'
    ],
    '': ['main.py', 'requirements.txt']
}

for folder, files in structure.items():
    if folder:
        os.makedirs(folder, exist_ok=True)
    for file in files:
        path = os.path.join(folder, file) if folder else file
        with open(path, 'w') as f:
            pass  # Creates empty file

print("Project structure created!")