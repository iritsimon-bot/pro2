#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ניתוח נתונים ודוח מסכם של המשתנים והתיוגים
"""

import pandas as pd
import os
from collections import Counter

def analyze_dataset(file_path, dataset_name):
    """ניתוח קובץ נתונים יחיד"""
    print(f"\n{'='*80}")
    print(f"ניתוח: {dataset_name}")
    print(f"{'='*80}")
    
    df = pd.read_csv(file_path)
    
    results = {
        'dataset': dataset_name,
        'total_records': len(df),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'gene_types': df['GeneType'].value_counts().to_dict(),
        'gene_group_methods': df['GeneGroupMethod'].value_counts().to_dict(),
        'unique_symbols': df['Symbol'].nunique(),
        'unique_descriptions': df['Description'].nunique(),
        'unique_gene_ids': df['NCBIGeneID'].nunique(),
    }
    
    print(f"\nמספר רשומות כולל: {results['total_records']:,}")
    print(f"\nעמודות במערך:")
    for i, col in enumerate(results['columns'], 1):
        print(f"  {i}. {col}")
    
    print(f"\nערכים חסרים:")
    for col, count in results['missing_values'].items():
        if count > 0:
            print(f"  {col}: {count}")
    
    print(f"\nתפלגות סוגי גנים (GeneType):")
    for gene_type, count in sorted(results['gene_types'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / results['total_records']) * 100
        print(f"  {gene_type}: {count:,} ({percentage:.2f}%)")
    
    print(f"\nתפלגות שיטות קיבוץ גנים (GeneGroupMethod):")
    for method, count in sorted(results['gene_group_methods'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / results['total_records']) * 100
        print(f"  {method}: {count:,} ({percentage:.2f}%)")
    
    print(f"\nמספר ייחודי של:")
    print(f"  סמלים (Symbols): {results['unique_symbols']:,}")
    print(f"  תיאורים (Descriptions): {results['unique_descriptions']:,}")
    print(f"  מזהה גן NCBI: {results['unique_gene_ids']:,}")
    
    # ניתוח אורך רצפי נוקליאוטידים
    df['SequenceLength'] = df['NucleotideSequence'].str.len()
    print(f"\nסטטיסטיקות אורך רצפי נוקליאוטידים:")
    print(f"  ממוצע: {df['SequenceLength'].mean():.2f}")
    print(f"  חציון: {df['SequenceLength'].median():.2f}")
    print(f"  מינימום: {df['SequenceLength'].min()}")
    print(f"  מקסימום: {df['SequenceLength'].max()}")
    
    return results

def generate_summary_report():
    """יצירת דוח מסכם כולל"""
    
    datasets = {
        'train.csv': 'אימון (Train)',
        'test.csv': 'בדיקה (Test)',
        'validation.csv': 'ולידציה (Validation)'
    }
    
    all_results = {}
    
    print("\n" + "="*80)
    print("דוח מסכם - ניתוח משתנים ותיוגים בנתונים")
    print("="*80)
    
    # ניתוח כל קובץ
    for file_path, dataset_name in datasets.items():
        if os.path.exists(file_path):
            all_results[dataset_name] = analyze_dataset(file_path, dataset_name)
        else:
            print(f"\nקובץ לא נמצא: {file_path}")
    
    # דוח מסכם כולל
    print("\n" + "="*80)
    print("סיכום כולל")
    print("="*80)
    
    total_records = sum(r['total_records'] for r in all_results.values())
    print(f"\nסה\"כ רשומות בכל הנתונים: {total_records:,}")
    
    # איחוד כל סוגי הגנים
    all_gene_types = Counter()
    for results in all_results.values():
        all_gene_types.update(results['gene_types'])
    
    print(f"\nתפלגות כוללת של סוגי גנים בכל הנתונים:")
    for gene_type, count in sorted(all_gene_types.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_records) * 100
        print(f"  {gene_type}: {count:,} ({percentage:.2f}%)")
    
    # איחוד כל שיטות הקיבוץ
    all_methods = Counter()
    for results in all_results.values():
        all_methods.update(results['gene_group_methods'])
    
    print(f"\nתפלגות כוללת של שיטות קיבוץ גנים בכל הנתונים:")
    for method, count in sorted(all_methods.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_records) * 100
        print(f"  {method}: {count:,} ({percentage:.2f}%)")
    
    # השוואה בין מערכי הנתונים
    print(f"\n{'='*80}")
    print("השוואה בין מערכי הנתונים")
    print(f"{'='*80}")
    print(f"{'מערך נתונים':<20} {'מספר רשומות':<20} {'סוגי גנים ייחודיים':<25}")
    print("-" * 80)
    for dataset_name, results in all_results.items():
        print(f"{dataset_name:<20} {results['total_records']:<20,} {len(results['gene_types']):<25}")
    
    return all_results

if __name__ == "__main__":
    try:
        results = generate_summary_report()
        print("\n" + "="*80)
        print("הניתוח הושלם בהצלחה!")
        print("="*80)
    except Exception as e:
        print(f"\nשגיאה בניתוח: {e}")
        import traceback
        traceback.print_exc()

