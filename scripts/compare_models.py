#!/usr/bin/env python3
"""
Compare two model results using statistical tests.

Usage examples:
# MTL vs Non-MTL (real3dad)
python ./compare_models.py \
    --ref /mnt/data/cossad/predictions/shapenet_real3dad/2025_12_11_v3_local/stats/all.csv \
    --test /mnt/data/cossad/predictions/shapenet_real3dad/2025_12_08_no_mined/stats/all.csv

# MTL vs Non-MTL (shapenet)
python ./compare_models.py \
    --ref /mnt/data/cossad/predictions/shapenet_real3dad/2025_12_11_v3_local_shapenet/stats/all.csv \
    --test /mnt/data/cossad/predictions/shapenet_real3dad/2025_12_11_v3_local_shapenet_anomalyonly/stats/all.csv

# With custom metric
python ./compare_models.py \
    --ref /mnt/data/cossad/predictions/shapenet_real3dad/2025_12_11_v3_local/stats/all.csv \
    --test /mnt/data/cossad/predictions/shapenet_real3dad/2025_12_11_v3_lambda/stats/all.csv \
    --metric P-ROCAUC
"""

import argparse
import sys
import pandas as pd
import numpy as np
from scipy import stats


def load_and_group(filepath, metric):
    """Load CSV and group metric values by class_name."""
    df = pd.read_csv(filepath)
    return df.groupby('class_name')[metric].apply(list).to_dict()


def compare_models(ref_file, test_file, metric):
    ref_data = load_and_group(ref_file, metric)
    test_data = load_and_group(test_file, metric)

    # Get all class names present in both files
    all_classes = sorted(set(ref_data.keys()) & set(test_data.keys()))

    results = []
    for cls in all_classes:
        ref_vals = np.array(ref_data[cls])
        test_vals = np.array(test_data[cls])
        print(f'Class: {cls}, {ref_vals.shape[0]}. {test_vals.shape[0]}')

        # Basic stats
        ref_mean, ref_std = np.mean(ref_vals), np.std(ref_vals, ddof=1)
        test_mean, test_std = np.mean(test_vals), np.std(test_vals, ddof=1)

        # Mann-Whitney U tests (one-sided)
        # H0: distributions are equal
        # alternative='greater' tests if ref > test
        _, p_ref_higher = stats.mannwhitneyu(ref_vals, test_vals, alternative='greater')
        _, p_ref_lower = stats.mannwhitneyu(ref_vals, test_vals, alternative='less')

        results.append({
            'class_name': cls,
            'ref_mean': ref_mean,
            'ref_std': ref_std,
            'test_mean': test_mean,
            'test_std': test_std,
            'p_ref_higher': p_ref_higher,
            'p_ref_lower': p_ref_lower,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description='Compare two model results using Mann-Whitney U statistical tests.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--ref',
        required=True,
        help='Path to reference model CSV file'
    )

    parser.add_argument(
        '--test',
        required=True,
        help='Path to test model CSV file'
    )

    parser.add_argument(
        '--metric',
        default='O-ROCAUC',
        choices=['O-ROCAUC', 'P-ROCAUC'],
        help='Metric to compare (default: O-ROCAUC)'
    )

    parser.add_argument(
        '--output',
        default='comparison_results.csv',
        help='Path to save comparison results CSV (default: comparison_results.csv)'
    )

    parser.add_argument(
        '--strict',
        action='store_true',
        default=False,
        help='Exit with code 1 if test overall mean is below reference overall mean (regression detected)'
    )

    args = parser.parse_args()

    results = compare_models(args.ref, args.test, args.metric)

    # Compute overall mean/std (no statistical test - samples are not independent)
    overall_row = {
        'class_name': 'OVERALL',
        'ref_mean': results['ref_mean'].mean(),
        'ref_std': results['ref_std'].mean(),
        'test_mean': results['test_mean'].mean(),
        'test_std': results['test_std'].mean(),
        'p_ref_higher': np.nan,
        'p_ref_lower': np.nan,
    }
    results = pd.concat([results, pd.DataFrame([overall_row])], ignore_index=True)

    # Format output
    print(f"\nModel Comparison Results (Mann-Whitney U test, metric: {args.metric})")
    print("=" * 100)
    print(
        f"{'class_name':<12} {'ref_mean':>10} {'ref_std':>10} {'test_mean':>10} {'test_std':>10} {'p(ref>test)':>12} {'p(ref<test)':>12}")
    print("-" * 100)

    for _, row in results.iterrows():
        if row['class_name'] == 'OVERALL':
            print("-" * 100)
        p_higher = f"{row['p_ref_higher']:>12.4f}" if not np.isnan(row['p_ref_higher']) else f"{'—':>12}"
        p_lower = f"{row['p_ref_lower']:>12.4f}" if not np.isnan(row['p_ref_lower']) else f"{'—':>12}"
        print(f"{row['class_name']:<12} {row['ref_mean']:>10.4f} {row['ref_std']:>10.4f} "
              f"{row['test_mean']:>10.4f} {row['test_std']:>10.4f} "
              f"{p_higher} {p_lower}")

    # Also save to CSV
    results.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Note about statistical power
    print("\n⚠️  Note: With n≈5 per group, minimum achievable p-value is ~0.004.")
    print("   Consider paired tests (Wilcoxon signed-rank) if samples are matched.")

    if args.strict:
        overall = results[results['class_name'] == 'OVERALL']
        if not overall.empty and overall.iloc[0]['test_mean'] < overall.iloc[0]['ref_mean']:
            print(f"\nREGRESSION DETECTED: overall test mean "
                  f"({overall.iloc[0]['test_mean']:.4f}) < ref mean "
                  f"({overall.iloc[0]['ref_mean']:.4f})")
            sys.exit(1)


if __name__ == "__main__":
    main()