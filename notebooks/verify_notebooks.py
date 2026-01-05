#!/usr/bin/env python3
"""
Notebook Verification Tool

A reusable script to verify that Jupyter notebooks execute without errors.
Supports testing all notebooks, specific notebooks, or notebooks matching patterns.

Usage:
    # Test all notebooks
    python verify_notebooks.py --all

    # Test specific notebooks
    python verify_notebooks.py notebook1.ipynb notebook2.ipynb

    # Test notebooks matching a pattern
    python verify_notebooks.py --pattern "Deep*.ipynb"

    # Test with custom timeout
    python verify_notebooks.py --all --timeout 600

    # Verbose output (show full errors)
    python verify_notebooks.py --all --verbose
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_notebook(nb_path: Path, timeout: int = 400) -> tuple[bool, str | None]:
    """
    Execute a notebook and return success status and error message if any.
    
    Args:
        nb_path: Path to the notebook file
        timeout: Maximum execution time in seconds
        
    Returns:
        Tuple of (success: bool, error_message: str | None)
    """
    try:
        result = subprocess.run(
            [
                sys.executable, '-m', 'jupyter', 'nbconvert',
                '--to', 'notebook',
                '--execute',
                '--inplace',
                f'--ExecutePreprocessor.timeout={timeout}',
                str(nb_path)
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 60  # Allow extra time for nbconvert overhead
        )
        if result.returncode == 0:
            return True, None
        else:
            err = result.stderr.strip()
            return False, err
    except subprocess.TimeoutExpired:
        return False, f'Timeout: Notebook took longer than {timeout}s'
    except Exception as e:
        return False, str(e)


def find_notebooks(base_path: Path, pattern: str | None = None) -> list[Path]:
    """
    Find all notebook files in the given path.
    
    Args:
        base_path: Directory to search
        pattern: Optional glob pattern to filter notebooks
        
    Returns:
        Sorted list of notebook paths
    """
    if pattern:
        notebooks = list(base_path.rglob(pattern))
    else:
        notebooks = list(base_path.rglob('*.ipynb'))
    
    # Filter out checkpoint files
    notebooks = [nb for nb in notebooks if '.ipynb_checkpoints' not in str(nb)]
    
    return sorted(notebooks)


def main():
    parser = argparse.ArgumentParser(
        description='Verify Jupyter notebooks execute without errors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'notebooks',
        nargs='*',
        help='Specific notebook files to test'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Test all notebooks in the current directory and subdirectories'
    )
    parser.add_argument(
        '--pattern', '-p',
        type=str,
        help='Glob pattern to match notebooks (e.g., "Deep*.ipynb")'
    )
    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=400,
        help='Timeout in seconds for each notebook (default: 400)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show full error messages'
    )
    parser.add_argument(
        '--directory', '-d',
        type=str,
        default='.',
        help='Base directory to search for notebooks (default: current directory)'
    )
    
    args = parser.parse_args()
    
    base_path = Path(args.directory).resolve()
    
    # Determine which notebooks to test
    if args.notebooks:
        notebooks = [Path(nb) for nb in args.notebooks]
    elif args.all or args.pattern:
        notebooks = find_notebooks(base_path, args.pattern)
    else:
        parser.print_help()
        print("\nError: Specify notebooks to test, use --all, or use --pattern")
        sys.exit(1)
    
    if not notebooks:
        print("No notebooks found to test")
        sys.exit(1)
    
    print(f'NOTEBOOK VERIFICATION')
    print(f'=====================')
    print(f'Testing {len(notebooks)} notebook(s)')
    print(f'Timeout: {args.timeout}s per notebook')
    print('=' * 70)
    
    results = {}
    start_time = time.time()
    
    for i, nb in enumerate(notebooks, 1):
        # Make path relative for display if possible
        try:
            display_path = nb.relative_to(base_path)
        except ValueError:
            display_path = nb
        
        print(f'[{i:2d}/{len(notebooks)}] {str(display_path):50s}', end=' ', flush=True)
        
        nb_start = time.time()
        success, error = run_notebook(nb, args.timeout)
        elapsed = time.time() - nb_start
        
        results[str(nb)] = {
            'success': success,
            'error': error,
            'time': elapsed
        }
        
        status = '\033[92mPASS\033[0m' if success else '\033[91mFAIL\033[0m'
        print(f'{status} ({elapsed:.1f}s)')
        
        if not success and args.verbose and error:
            print(f'    Error: {error}')
    
    total_time = time.time() - start_time
    
    # Summary
    print('=' * 70)
    passed = sum(1 for r in results.values() if r['success'])
    failed_list = [k for k, v in results.items() if not v['success']]
    
    print()
    print('SUMMARY')
    print('-------')
    print(f'  Total notebooks: {len(results)}')
    print(f'  Passed: \033[92m{passed}\033[0m')
    print(f'  Failed: \033[91m{len(failed_list)}\033[0m')
    print(f'  Success rate: {100 * passed / len(results):.0f}%')
    print(f'  Total time: {total_time:.1f}s')
    
    if failed_list:
        print()
        print('FAILED NOTEBOOKS:')
        for f in failed_list:
            print(f'  - {f}')
            if not args.verbose:
                err = results[f].get('error', '')
                if err:
                    # Show last meaningful line of error
                    lines = [l for l in err.split('\n') if l.strip()]
                    if lines:
                        last_line = lines[-1][:100]
                        print(f'    Error: {last_line}')
    else:
        print()
        print('\033[92mALL NOTEBOOKS PASSED!\033[0m')
    
    # Exit with appropriate code
    sys.exit(0 if not failed_list else 1)


if __name__ == '__main__':
    main()

