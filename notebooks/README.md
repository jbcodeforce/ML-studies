# Test all notebooks

```sh
python verify_notebooks.py --all
```

### Test specific notebooks
```sh
python verify_notebooks.py notebook1.ipynb notebook2.ipynb
```

### Test notebooks matching a pattern
```sh
python verify_notebooks.py --pattern "Deep*.ipynb"
```

### Test with custom timeout (e.g., 600 seconds)
```sh
python verify_notebooks.py --all --timeout 600
```

### Verbose output (show full error traces)
```sh
python verify_notebooks.py --all --verbose
```

### Test notebooks in a specific directory
```sh
python verify_notebooks.py --all --directory /path/to/notebooks
```