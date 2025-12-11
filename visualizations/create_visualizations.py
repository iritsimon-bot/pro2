#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
יצירת ויזואליזציות מקיפות לניתוח נתוני גנים
===============================================
תאריך: דצמבר 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# הגדרת סגנון גרפים
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# צבעים מותאמים אישית
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'accent': '#F18F01',
    'success': '#C73E1D',
    'info': '#3B1F2B',
    'palette': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', 
                '#5FAD56', '#F2C14E', '#78C4D4', '#8B5CF6', '#EC4899']
}

def load_data():
    """טעינת כל מערכי הנתונים"""
    base_path = Path(__file__).parent.parent
    
    datasets = {}
    for name in ['train', 'test', 'validation']:
        file_path = base_path / f'{name}.csv'
        if file_path.exists():
            df = pd.read_csv(file_path)
            # הוספת עמודת אורך רצף
            df['SequenceLength'] = df['NucleotideSequence'].str.replace('<', '').str.replace('>', '').str.len()
            # חישוב תוכן GC
            df['GC_Content'] = df['NucleotideSequence'].apply(calculate_gc_content)
            datasets[name] = df
            print(f"✓ נטען {name}.csv: {len(df):,} רשומות")
    
    return datasets

def calculate_gc_content(seq):
    """חישוב אחוז GC ברצף"""
    seq = str(seq).replace('<', '').replace('>', '').upper()
    if len(seq) == 0:
        return 0
    gc_count = seq.count('G') + seq.count('C')
    return round(gc_count / len(seq) * 100, 2)

def plot_gene_type_distribution(datasets, output_dir):
    """גרף 1: התפלגות סוגי הגנים"""
    print("\n📊 יוצר גרף התפלגות סוגי גנים...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # גרף עוגה - Train
    train_counts = datasets['train']['GeneType'].value_counts()
    colors = COLORS['palette'][:len(train_counts)]
    
    ax1 = axes[0, 0]
    wedges, texts, autotexts = ax1.pie(train_counts.values, labels=train_counts.index, 
                                        autopct='%1.1f%%', colors=colors,
                                        explode=[0.05 if i < 3 else 0 for i in range(len(train_counts))],
                                        shadow=True)
    ax1.set_title('התפלגות סוגי גנים - מערך אימון\n(Train Dataset)', fontsize=14, fontweight='bold')
    
    # גרף עמודות - כל המערכים
    ax2 = axes[0, 1]
    all_types = list(datasets['train']['GeneType'].unique())
    x = np.arange(len(all_types))
    width = 0.25
    
    for i, (name, df) in enumerate(datasets.items()):
        counts = df['GeneType'].value_counts().reindex(all_types, fill_value=0)
        ax2.bar(x + i*width, counts.values, width, label=name.capitalize(), color=COLORS['palette'][i])
    
    ax2.set_xlabel('סוג גן')
    ax2.set_ylabel('מספר רשומות')
    ax2.set_title('השוואת התפלגות סוגי גנים בין המערכים', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(all_types, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # גרף עמודות אופקי - Train
    ax3 = axes[1, 0]
    train_counts_sorted = train_counts.sort_values()
    bars = ax3.barh(train_counts_sorted.index, train_counts_sorted.values, color=COLORS['palette'][:len(train_counts)])
    ax3.set_xlabel('מספר רשומות')
    ax3.set_title('התפלגות סוגי גנים (ממויין)', fontsize=14, fontweight='bold')
    
    # הוספת מספרים על העמודות
    for bar, val in zip(bars, train_counts_sorted.values):
        ax3.text(val + 100, bar.get_y() + bar.get_height()/2, f'{val:,}', 
                va='center', fontsize=10)
    
    # גרף לוגריתמי
    ax4 = axes[1, 1]
    ax4.bar(train_counts.index, train_counts.values, color=COLORS['palette'][:len(train_counts)])
    ax4.set_yscale('log')
    ax4.set_xlabel('סוג גן')
    ax4.set_ylabel('מספר רשומות (סקאלה לוגריתמית)')
    ax4.set_title('התפלגות סוגי גנים - סקאלה לוגריתמית\n(מדגישה את חוסר האיזון)', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_gene_type_distribution.png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ נשמר: 01_gene_type_distribution.png")

def plot_sequence_length_analysis(datasets, output_dir):
    """גרף 2: ניתוח אורך רצפים"""
    print("\n📊 יוצר גרף ניתוח אורך רצפים...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    train_df = datasets['train']
    
    # היסטוגרמה של אורכי רצפים
    ax1 = axes[0, 0]
    ax1.hist(train_df['SequenceLength'], bins=50, color=COLORS['primary'], 
             edgecolor='white', alpha=0.7)
    ax1.axvline(train_df['SequenceLength'].mean(), color=COLORS['secondary'], 
                linestyle='--', linewidth=2, label=f'ממוצע: {train_df["SequenceLength"].mean():.0f}')
    ax1.axvline(train_df['SequenceLength'].median(), color=COLORS['accent'], 
                linestyle='--', linewidth=2, label=f'חציון: {train_df["SequenceLength"].median():.0f}')
    ax1.set_xlabel('אורך רצף (בסיסים)')
    ax1.set_ylabel('תדירות')
    ax1.set_title('התפלגות אורכי רצפים', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot לפי סוג גן
    ax2 = axes[0, 1]
    gene_order = train_df.groupby('GeneType')['SequenceLength'].median().sort_values(ascending=False).index
    sns.boxplot(data=train_df, x='GeneType', y='SequenceLength', order=gene_order,
                palette=COLORS['palette'], ax=ax2)
    ax2.set_xlabel('סוג גן')
    ax2.set_ylabel('אורך רצף')
    ax2.set_title('אורך רצף לפי סוג גן', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Violin plot
    ax3 = axes[1, 0]
    sns.violinplot(data=train_df, x='GeneType', y='SequenceLength', order=gene_order,
                   palette=COLORS['palette'], ax=ax3)
    ax3.set_xlabel('סוג גן')
    ax3.set_ylabel('אורך רצף')
    ax3.set_title('התפלגות אורך רצף לפי סוג גן (Violin Plot)', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # השוואה בין מערכים
    ax4 = axes[1, 1]
    data_for_box = []
    labels_for_box = []
    for name, df in datasets.items():
        data_for_box.append(df['SequenceLength'].values)
        labels_for_box.append(name.capitalize())
    
    bp = ax4.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], COLORS['palette'][:3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel('אורך רצף')
    ax4.set_title('השוואת אורכי רצפים בין המערכים', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_sequence_length_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ נשמר: 02_sequence_length_analysis.png")

def plot_gc_content_analysis(datasets, output_dir):
    """גרף 3: ניתוח תוכן GC"""
    print("\n📊 יוצר גרף ניתוח תוכן GC...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    train_df = datasets['train']
    
    # היסטוגרמה של תוכן GC
    ax1 = axes[0, 0]
    ax1.hist(train_df['GC_Content'], bins=50, color=COLORS['primary'], 
             edgecolor='white', alpha=0.7)
    ax1.axvline(train_df['GC_Content'].mean(), color=COLORS['secondary'], 
                linestyle='--', linewidth=2, label=f'ממוצע: {train_df["GC_Content"].mean():.1f}%')
    ax1.set_xlabel('תוכן GC (%)')
    ax1.set_ylabel('תדירות')
    ax1.set_title('התפלגות תוכן GC', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot לפי סוג גן
    ax2 = axes[0, 1]
    gene_order = train_df.groupby('GeneType')['GC_Content'].median().sort_values(ascending=False).index
    sns.boxplot(data=train_df, x='GeneType', y='GC_Content', order=gene_order,
                palette=COLORS['palette'], ax=ax2)
    ax2.set_xlabel('סוג גן')
    ax2.set_ylabel('תוכן GC (%)')
    ax2.set_title('תוכן GC לפי סוג גן', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # Scatter plot: GC vs Length
    ax3 = axes[1, 0]
    sample_df = train_df.sample(min(5000, len(train_df)), random_state=42)
    scatter = ax3.scatter(sample_df['SequenceLength'], sample_df['GC_Content'], 
                          c=sample_df['GeneType'].astype('category').cat.codes,
                          cmap='tab10', alpha=0.5, s=20)
    ax3.set_xlabel('אורך רצף')
    ax3.set_ylabel('תוכן GC (%)')
    ax3.set_title('קשר בין אורך רצף לתוכן GC', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # KDE plot לפי סוג גן
    ax4 = axes[1, 1]
    for i, gene_type in enumerate(train_df['GeneType'].unique()[:5]):
        subset = train_df[train_df['GeneType'] == gene_type]['GC_Content']
        if len(subset) > 10:
            sns.kdeplot(subset, ax=ax4, label=gene_type, color=COLORS['palette'][i])
    ax4.set_xlabel('תוכן GC (%)')
    ax4.set_ylabel('צפיפות')
    ax4.set_title('התפלגות תוכן GC לפי סוגי גנים עיקריים', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_gc_content_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ נשמר: 03_gc_content_analysis.png")

def plot_class_imbalance(datasets, output_dir):
    """גרף 4: ויזואליזציה של חוסר איזון קלאסים"""
    print("\n📊 יוצר גרף חוסר איזון קלאסים...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    train_df = datasets['train']
    counts = train_df['GeneType'].value_counts()
    
    # גרף יחס לקלאס הקטן
    ax1 = axes[0]
    min_count = counts.min()
    ratios = counts / min_count
    bars = ax1.bar(ratios.index, ratios.values, color=COLORS['palette'][:len(ratios)])
    ax1.set_xlabel('סוג גן')
    ax1.set_ylabel('יחס לקלאס הקטן ביותר')
    ax1.set_title('יחס כמות לקלאס הקטן ביותר (scRNA=1)\nמדגיש את חוסר האיזון הקיצוני', 
                  fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_yscale('log')
    
    # הוספת ערכים
    for bar, val in zip(bars, ratios.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.0f}x', 
                ha='center', va='bottom', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # Treemap style - גרף שטח
    ax2 = axes[1]
    
    # יצירת גרף שטח פשוט
    total = counts.sum()
    percentages = (counts / total * 100).values
    labels = [f'{name}\n{pct:.1f}%\n({count:,})' 
              for name, pct, count in zip(counts.index, percentages, counts.values)]
    
    # גרף waffle-style
    sizes = counts.values
    colors = COLORS['palette'][:len(sizes)]
    
    squarify_like = ax2.pie(sizes, labels=labels, colors=colors, 
                            autopct='', startangle=90,
                            wedgeprops=dict(width=0.7))
    ax2.set_title('חלוקת הנתונים לפי סוג גן\n(גודל יחסי)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_class_imbalance.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ נשמר: 04_class_imbalance.png")

def plot_dataset_comparison(datasets, output_dir):
    """גרף 5: השוואה בין מערכי הנתונים"""
    print("\n📊 יוצר גרף השוואת מערכים...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # גרף 1: גודל מערכים
    ax1 = axes[0, 0]
    sizes = [len(df) for df in datasets.values()]
    names = [name.capitalize() for name in datasets.keys()]
    bars = ax1.bar(names, sizes, color=COLORS['palette'][:3])
    ax1.set_ylabel('מספר רשומות')
    ax1.set_title('גודל מערכי הנתונים', fontsize=14, fontweight='bold')
    
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200, 
                f'{size:,}', ha='center', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # גרף 2: אחוז מהנתונים
    ax2 = axes[0, 1]
    total = sum(sizes)
    percentages = [s/total*100 for s in sizes]
    wedges, texts, autotexts = ax2.pie(percentages, labels=names, autopct='%1.1f%%',
                                        colors=COLORS['palette'][:3], explode=[0.05, 0, 0],
                                        shadow=True)
    ax2.set_title('חלוקת הנתונים בין המערכים', fontsize=14, fontweight='bold')
    
    # גרף 3: השוואת ממוצע אורך רצף
    ax3 = axes[1, 0]
    means = [df['SequenceLength'].mean() for df in datasets.values()]
    stds = [df['SequenceLength'].std() for df in datasets.values()]
    x = np.arange(len(names))
    bars = ax3.bar(x, means, yerr=stds, color=COLORS['palette'][:3], 
                   capsize=5, error_kw={'linewidth': 2})
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_ylabel('אורך רצף ממוצע')
    ax3.set_title('ממוצע אורך רצף לפי מערך (עם סטיית תקן)', fontsize=14, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # גרף 4: השוואת תוכן GC
    ax4 = axes[1, 1]
    gc_means = [df['GC_Content'].mean() for df in datasets.values()]
    gc_stds = [df['GC_Content'].std() for df in datasets.values()]
    bars = ax4.bar(x, gc_means, yerr=gc_stds, color=COLORS['palette'][:3],
                   capsize=5, error_kw={'linewidth': 2})
    ax4.set_xticks(x)
    ax4.set_xticklabels(names)
    ax4.set_ylabel('תוכן GC ממוצע (%)')
    ax4.set_title('ממוצע תוכן GC לפי מערך (עם סטיית תקן)', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_dataset_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ נשמר: 05_dataset_comparison.png")

def plot_heatmap_correlation(datasets, output_dir):
    """גרף 6: מפת חום של קורלציות"""
    print("\n📊 יוצר מפת חום קורלציות...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    train_df = datasets['train']
    
    # יצירת משתנים מספריים
    train_df['GeneType_Code'] = train_df['GeneType'].astype('category').cat.codes
    train_df['DescLength'] = train_df['Description'].str.len()
    train_df['SymbolLength'] = train_df['Symbol'].str.len()
    
    # מטריצת קורלציה
    numeric_cols = ['SequenceLength', 'GC_Content', 'GeneType_Code', 'DescLength', 'SymbolLength']
    corr_matrix = train_df[numeric_cols].corr()
    
    ax1 = axes[0]
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, ax=ax1, fmt='.2f',
                xticklabels=['אורך רצף', 'GC %', 'סוג גן', 'אורך תיאור', 'אורך סמל'],
                yticklabels=['אורך רצף', 'GC %', 'סוג גן', 'אורך תיאור', 'אורך סמל'])
    ax1.set_title('מטריצת קורלציה בין משתנים מספריים', fontsize=14, fontweight='bold')
    
    # גרף scatter עם קו רגרסיה
    ax2 = axes[1]
    sample = train_df.sample(min(3000, len(train_df)), random_state=42)
    sns.regplot(data=sample, x='SequenceLength', y='GC_Content', 
                scatter_kws={'alpha': 0.3, 'color': COLORS['primary']},
                line_kws={'color': COLORS['secondary'], 'linewidth': 2},
                ax=ax2)
    ax2.set_xlabel('אורך רצף')
    ax2.set_ylabel('תוכן GC (%)')
    ax2.set_title('קשר בין אורך רצף לתוכן GC (עם קו רגרסיה)', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '06_correlation_heatmap.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ נשמר: 06_correlation_heatmap.png")

def plot_rare_classes_analysis(datasets, output_dir):
    """גרף 7: ניתוח קלאסים נדירים"""
    print("\n📊 יוצר גרף ניתוח קלאסים נדירים...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # איסוף נתונים על קלאסים נדירים
    rare_data = []
    for name, df in datasets.items():
        counts = df['GeneType'].value_counts()
        for gene_type, count in counts.items():
            rare_data.append({'Dataset': name.capitalize(), 'GeneType': gene_type, 'Count': count})
    
    rare_df = pd.DataFrame(rare_data)
    
    # סינון קלאסים נדירים
    rare_types = ['scRNA', 'snRNA', 'rRNA', 'OTHER', 'tRNA']
    rare_subset = rare_df[rare_df['GeneType'].isin(rare_types)]
    
    ax1 = axes[0]
    pivot_df = rare_subset.pivot(index='GeneType', columns='Dataset', values='Count').fillna(0)
    pivot_df.plot(kind='bar', ax=ax1, color=COLORS['palette'][:3])
    ax1.set_xlabel('סוג גן')
    ax1.set_ylabel('מספר רשומות')
    ax1.set_title('התפלגות קלאסים נדירים בין המערכים', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='מערך')
    ax1.grid(axis='y', alpha=0.3)
    
    # גרף cumulative
    ax2 = axes[1]
    train_counts = datasets['train']['GeneType'].value_counts().sort_values(ascending=False)
    cumsum = train_counts.cumsum() / train_counts.sum() * 100
    
    ax2.plot(range(len(cumsum)), cumsum.values, marker='o', color=COLORS['primary'], 
             linewidth=2, markersize=8)
    ax2.fill_between(range(len(cumsum)), cumsum.values, alpha=0.3, color=COLORS['primary'])
    ax2.set_xticks(range(len(cumsum)))
    ax2.set_xticklabels(cumsum.index, rotation=45, ha='right')
    ax2.set_xlabel('סוג גן')
    ax2.set_ylabel('אחוז מצטבר מהנתונים')
    ax2.set_title('התפלגות מצטברת - כמה קלאסים מכסים את הנתונים?', fontsize=14, fontweight='bold')
    ax2.axhline(y=80, color=COLORS['secondary'], linestyle='--', label='80%')
    ax2.axhline(y=95, color=COLORS['accent'], linestyle='--', label='95%')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / '07_rare_classes_analysis.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ נשמר: 07_rare_classes_analysis.png")

def plot_sequence_characteristics(datasets, output_dir):
    """גרף 8: מאפייני רצפים מתקדמים"""
    print("\n📊 יוצר גרף מאפייני רצפים...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    train_df = datasets['train'].copy()
    
    # חישוב מאפיינים נוספים
    train_df['A_Content'] = train_df['NucleotideSequence'].apply(
        lambda x: str(x).replace('<','').replace('>','').upper().count('A') / max(len(str(x))-2, 1) * 100)
    train_df['T_Content'] = train_df['NucleotideSequence'].apply(
        lambda x: str(x).replace('<','').replace('>','').upper().count('T') / max(len(str(x))-2, 1) * 100)
    train_df['G_Content'] = train_df['NucleotideSequence'].apply(
        lambda x: str(x).replace('<','').replace('>','').upper().count('G') / max(len(str(x))-2, 1) * 100)
    train_df['C_Content'] = train_df['NucleotideSequence'].apply(
        lambda x: str(x).replace('<','').replace('>','').upper().count('C') / max(len(str(x))-2, 1) * 100)
    
    # גרף 1: תוכן נוקליאוטידים ממוצע
    ax1 = axes[0, 0]
    nucleotides = ['A', 'T', 'G', 'C']
    means = [train_df['A_Content'].mean(), train_df['T_Content'].mean(),
             train_df['G_Content'].mean(), train_df['C_Content'].mean()]
    bars = ax1.bar(nucleotides, means, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_ylabel('אחוז ממוצע')
    ax1.set_title('תוכן נוקליאוטידים ממוצע ברצפים', fontsize=14, fontweight='bold')
    ax1.axhline(y=25, color='gray', linestyle='--', alpha=0.5, label='צפי אחיד (25%)')
    ax1.legend()
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # גרף 2: היסטוגרמה של אורך רצפים לפי טווחים
    ax2 = axes[0, 1]
    bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
    train_df['LengthBin'] = pd.cut(train_df['SequenceLength'], bins=bins, labels=labels)
    bin_counts = train_df['LengthBin'].value_counts().reindex(labels)
    ax2.bar(labels, bin_counts.values, color=COLORS['palette'][:6])
    ax2.set_xlabel('טווח אורך (בסיסים)')
    ax2.set_ylabel('מספר רצפים')
    ax2.set_title('התפלגות אורכי רצפים לפי טווחים', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # גרף 3: Swarm plot לקלאסים עיקריים
    ax3 = axes[1, 0]
    main_types = train_df['GeneType'].value_counts().head(5).index
    subset = train_df[train_df['GeneType'].isin(main_types)].sample(min(2000, len(train_df)), random_state=42)
    sns.stripplot(data=subset, x='GeneType', y='GC_Content', hue='GeneType',
                  palette=COLORS['palette'][:5], alpha=0.5, ax=ax3, legend=False)
    ax3.set_xlabel('סוג גן')
    ax3.set_ylabel('תוכן GC (%)')
    ax3.set_title('פיזור תוכן GC ב-5 סוגי הגנים העיקריים', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # גרף 4: 2D histogram
    ax4 = axes[1, 1]
    h = ax4.hist2d(train_df['SequenceLength'], train_df['GC_Content'], 
                   bins=[50, 50], cmap='YlOrRd')
    plt.colorbar(h[3], ax=ax4, label='מספר רצפים')
    ax4.set_xlabel('אורך רצף')
    ax4.set_ylabel('תוכן GC (%)')
    ax4.set_title('מפת צפיפות: אורך רצף מול תוכן GC', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / '08_sequence_characteristics.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("  ✓ נשמר: 08_sequence_characteristics.png")

def generate_summary_stats(datasets, output_dir):
    """יצירת קובץ סטטיסטיקות מסכם"""
    print("\n📝 יוצר דוח סטטיסטיקות מסכם...")
    
    summary = []
    summary.append("=" * 80)
    summary.append("דוח סטטיסטיקות מסכם - ניתוח נתוני גנים")
    summary.append("=" * 80)
    summary.append("")
    
    # סטטיסטיקות כלליות
    total_records = sum(len(df) for df in datasets.values())
    summary.append(f"סה\"כ רשומות בכל המערכים: {total_records:,}")
    summary.append("")
    
    for name, df in datasets.items():
        summary.append(f"\n--- {name.upper()} ---")
        summary.append(f"  מספר רשומות: {len(df):,}")
        summary.append(f"  סוגי גנים ייחודיים: {df['GeneType'].nunique()}")
        summary.append(f"  ממוצע אורך רצף: {df['SequenceLength'].mean():.1f}")
        summary.append(f"  חציון אורך רצף: {df['SequenceLength'].median():.1f}")
        summary.append(f"  ממוצע תוכן GC: {df['GC_Content'].mean():.1f}%")
        
        summary.append("\n  התפלגות סוגי גנים:")
        for gene_type, count in df['GeneType'].value_counts().items():
            pct = count / len(df) * 100
            summary.append(f"    {gene_type}: {count:,} ({pct:.1f}%)")
    
    # שמירה לקובץ
    with open(output_dir / 'summary_statistics.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print("  ✓ נשמר: summary_statistics.txt")

def generate_html_report(datasets, output_dir):
    """יצירת דוח HTML מסכם עם מסקנות"""
    print("\n📝 יוצר דוח HTML מסכם...")
    
    train_df = datasets['train']
    total_records = sum(len(df) for df in datasets.values())
    
    html_content = f"""<!DOCTYPE html>
<html dir="rtl" lang="he">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>דוח סיכום ומסקנות - ניתוח נתוני גנים</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700&display=swap');
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Heebo', Arial, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            min-height: 100vh;
            color: #e0e0e0;
            line-height: 1.8;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}
        
        h1 {{
            color: #00d4ff;
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
        }}
        
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 40px;
            font-size: 1.1em;
        }}
        
        h2 {{
            color: #ff6b6b;
            border-bottom: 2px solid #ff6b6b;
            padding-bottom: 10px;
            margin: 30px 0 20px 0;
        }}
        
        h3 {{
            color: #4ecdc4;
            margin: 25px 0 15px 0;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, rgba(78, 205, 196, 0.2), rgba(0, 212, 255, 0.1));
            border: 1px solid rgba(78, 205, 196, 0.3);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-number {{
            font-size: 2.5em;
            font-weight: 700;
            color: #00d4ff;
            display: block;
        }}
        
        .stat-label {{
            color: #aaa;
            font-size: 0.95em;
        }}
        
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
            margin: 30px 0;
        }}
        
        .image-card {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 15px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        .image-card img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        
        .image-title {{
            padding: 15px;
            text-align: center;
            color: #4ecdc4;
            font-weight: 500;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        
        th, td {{
            padding: 15px;
            text-align: right;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        th {{
            background: rgba(78, 205, 196, 0.2);
            color: #4ecdc4;
            font-weight: 500;
        }}
        
        tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        
        .alert {{
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        
        .alert-danger {{
            background: rgba(255, 107, 107, 0.2);
            border: 1px solid #ff6b6b;
        }}
        
        .alert-warning {{
            background: rgba(255, 193, 7, 0.2);
            border: 1px solid #ffc107;
        }}
        
        .alert-success {{
            background: rgba(78, 205, 196, 0.2);
            border: 1px solid #4ecdc4;
        }}
        
        .conclusion-box {{
            background: linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(78, 205, 196, 0.1));
            border: 2px solid #00d4ff;
            border-radius: 15px;
            padding: 30px;
            margin: 30px 0;
        }}
        
        .conclusion-box h3 {{
            color: #00d4ff;
            margin-bottom: 15px;
        }}
        
        ul {{
            padding-right: 25px;
            margin: 15px 0;
        }}
        
        li {{
            margin: 10px 0;
        }}
        
        .emoji {{
            font-size: 1.2em;
        }}
        
        .highlight {{
            color: #ffc107;
            font-weight: 500;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 דוח סיכום ומסקנות</h1>
        <p class="subtitle">ניתוח מקיף של נתוני גנים | דצמבר 2024</p>
        
        <h2>📊 סטטיסטיקות מרכזיות</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <span class="stat-number">{total_records:,}</span>
                <span class="stat-label">סה"כ רשומות</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{len(datasets['train']):,}</span>
                <span class="stat-label">מערך אימון</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{len(datasets['test']):,}</span>
                <span class="stat-label">מערך בדיקה</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{len(datasets['validation']):,}</span>
                <span class="stat-label">מערך ולידציה</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{train_df['GeneType'].nunique()}</span>
                <span class="stat-label">סוגי גנים</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{train_df['SequenceLength'].mean():.0f}</span>
                <span class="stat-label">אורך רצף ממוצע</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{train_df['GC_Content'].mean():.1f}%</span>
                <span class="stat-label">תוכן GC ממוצע</span>
            </div>
            <div class="stat-card">
                <span class="stat-number">{train_df['NCBIGeneID'].nunique():,}</span>
                <span class="stat-label">גנים ייחודיים</span>
            </div>
        </div>
        
        <h2>📈 ויזואליזציות</h2>
        <div class="image-grid">
            <div class="image-card">
                <img src="01_gene_type_distribution.png" alt="התפלגות סוגי גנים">
                <div class="image-title">התפלגות סוגי גנים</div>
            </div>
            <div class="image-card">
                <img src="02_sequence_length_analysis.png" alt="ניתוח אורך רצפים">
                <div class="image-title">ניתוח אורך רצפים</div>
            </div>
            <div class="image-card">
                <img src="03_gc_content_analysis.png" alt="ניתוח תוכן GC">
                <div class="image-title">ניתוח תוכן GC</div>
            </div>
            <div class="image-card">
                <img src="04_class_imbalance.png" alt="חוסר איזון קלאסים">
                <div class="image-title">חוסר איזון קלאסים</div>
            </div>
            <div class="image-card">
                <img src="05_dataset_comparison.png" alt="השוואת מערכים">
                <div class="image-title">השוואת מערכים</div>
            </div>
            <div class="image-card">
                <img src="06_correlation_heatmap.png" alt="מפת קורלציות">
                <div class="image-title">מפת קורלציות</div>
            </div>
            <div class="image-card">
                <img src="07_rare_classes_analysis.png" alt="קלאסים נדירים">
                <div class="image-title">ניתוח קלאסים נדירים</div>
            </div>
            <div class="image-card">
                <img src="08_sequence_characteristics.png" alt="מאפייני רצפים">
                <div class="image-title">מאפייני רצפים</div>
            </div>
        </div>
        
        <h2>🔍 ממצאים עיקריים</h2>
        
        <h3>התפלגות סוגי גנים</h3>
        <table>
            <tr>
                <th>סוג גן</th>
                <th>כמות</th>
                <th>אחוז</th>
            </tr>
            {"".join(f'<tr><td>{gt}</td><td>{count:,}</td><td>{count/len(train_df)*100:.1f}%</td></tr>' 
                     for gt, count in train_df['GeneType'].value_counts().items())}
        </table>
        
        <h2>⚠️ בעיות שזוהו</h2>
        
        <div class="alert alert-danger">
            <h3><span class="emoji">🔴</span> בעיה קריטית: חוסר איזון קלאסים</h3>
            <p>קיים חוסר איזון קיצוני בין הקלאסים:</p>
            <ul>
                <li><span class="highlight">PSEUDO</span> ו-<span class="highlight">BIOLOGICAL_REGION</span> מהווים יחד <strong>76%</strong> מהנתונים</li>
                <li><span class="highlight">scRNA</span> מכיל רק <strong>3 דוגמאות</strong> במערך האימון</li>
                <li>יחס הקלאס הגדול לקטן: <strong>3,407:1</strong></li>
            </ul>
        </div>
        
        <div class="alert alert-warning">
            <h3><span class="emoji">🟠</span> אזהרה: Data Leakage</h3>
            <p>עמודת <span class="highlight">Description</span> מכילה מילות מפתח שחושפות את סוג הגן:</p>
            <ul>
                <li>"pseudogene" → PSEUDO (99.9%)</li>
                <li>"microRNA" → ncRNA (100%)</li>
                <li>"regulatory region" → BIOLOGICAL_REGION (100%)</li>
            </ul>
        </div>
        
        <div class="alert alert-success">
            <h3><span class="emoji">🟢</span> נקודות חזקות</h3>
            <ul>
                <li>אין ערכים חסרים בנתונים</li>
                <li>מבנה נתונים אחיד בכל המערכים</li>
                <li>NCBIGeneID ייחודי לכל רשומה</li>
                <li>רצפי נוקליאוטידים תקינים (A, T, G, C)</li>
            </ul>
        </div>
        
        <h2>💡 מסקנות והמלצות</h2>
        
        <div class="conclusion-box">
            <h3>1. הכנת הנתונים</h3>
            <ul>
                <li><strong>להסיר:</strong> GeneGroupMethod (ערך קבוע), עמודת Index</li>
                <li><strong>לא להשתמש:</strong> Description לחיזוי GeneType (Data Leakage)</li>
                <li><strong>לסנן:</strong> רצפים קצרים מ-20 בסיסים</li>
            </ul>
        </div>
        
        <div class="conclusion-box">
            <h3>2. טיפול בחוסר איזון</h3>
            <ul>
                <li>להשתמש ב-<strong>SMOTE</strong> או <strong>ADASYN</strong> להגדלת קלאסים קטנים</li>
                <li>להשתמש ב-<strong>Class Weights</strong> בפונקציית ההפסד</li>
                <li>לשקול <strong>מיזוג קלאסים נדירים</strong> (scRNA, snRNA, rRNA → smallRNA)</li>
            </ul>
        </div>
        
        <div class="conclusion-box">
            <h3>3. Feature Engineering</h3>
            <ul>
                <li><strong>אורך רצף</strong> - נמצאה שונות גבוהה בין סוגי גנים</li>
                <li><strong>תוכן GC</strong> - מאפיין ביולוגי חשוב</li>
                <li><strong>קידומת Symbol</strong> - מנבאת סוג גן ב-70-99%</li>
                <li><strong>תוכן נוקליאוטידים</strong> (A%, T%, G%, C%)</li>
            </ul>
        </div>
        
        <div class="conclusion-box">
            <h3>4. מדדי הערכה מומלצים</h3>
            <ul>
                <li><strong>לא להשתמש ב-Accuracy</strong> - מטעה בגלל חוסר איזון</li>
                <li><strong>F1-Score (macro)</strong> - ממוצע שווה על כל הקלאסים</li>
                <li><strong>Confusion Matrix</strong> - לזיהוי טעויות ספציפיות</li>
                <li><strong>ROC-AUC (One-vs-Rest)</strong> - להשוואת ביצועים בין קלאסים</li>
            </ul>
        </div>
        
        <h2>🎯 סיכום סופי</h2>
        <p>הנתונים מכילים מידע עשיר על גנים, אך דורשים טיפול מוקדם לפני בניית מודלים:</p>
        <ul>
            <li>✅ מבנה נתונים איכותי ומאורגן</li>
            <li>✅ מגוון רחב של סוגי גנים</li>
            <li>⚠️ חוסר איזון קיצוני בקלאסים</li>
            <li>⚠️ Data Leakage בעמודת Description</li>
            <li>⚠️ קלאסים נדירים עם דוגמאות בודדות</li>
        </ul>
        <p><strong>המלצה:</strong> לבצע ניקוי נתונים לפי ההנחיות לעיל לפני בניית מודלי Machine Learning.</p>
        
        <div class="footer">
            <p>דוח זה נוצר אוטומטית | דצמבר 2024</p>
            <p>📁 כל הגרפים נשמרו בתיקייה: visualizations/</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_dir / 'summary_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("  ✓ נשמר: summary_report.html")

def main():
    """פונקציה ראשית"""
    print("\n" + "="*60)
    print("🧬 יצירת ויזואליזציות לניתוח נתוני גנים")
    print("="*60)
    
    # יצירת תיקיית פלט
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # טעינת נתונים
    print("\n📂 טוען נתונים...")
    datasets = load_data()
    
    if not datasets:
        print("❌ לא נמצאו קבצי נתונים!")
        return
    
    # יצירת כל הויזואליזציות
    plot_gene_type_distribution(datasets, output_dir)
    plot_sequence_length_analysis(datasets, output_dir)
    plot_gc_content_analysis(datasets, output_dir)
    plot_class_imbalance(datasets, output_dir)
    plot_dataset_comparison(datasets, output_dir)
    plot_heatmap_correlation(datasets, output_dir)
    plot_rare_classes_analysis(datasets, output_dir)
    plot_sequence_characteristics(datasets, output_dir)
    
    # יצירת דוחות
    generate_summary_stats(datasets, output_dir)
    generate_html_report(datasets, output_dir)
    
    print("\n" + "="*60)
    print("✅ הויזואליזציות נוצרו בהצלחה!")
    print(f"📁 כל הקבצים נשמרו בתיקייה: {output_dir}")
    print("="*60)
    print("\nקבצים שנוצרו:")
    print("  • 01_gene_type_distribution.png")
    print("  • 02_sequence_length_analysis.png")
    print("  • 03_gc_content_analysis.png")
    print("  • 04_class_imbalance.png")
    print("  • 05_dataset_comparison.png")
    print("  • 06_correlation_heatmap.png")
    print("  • 07_rare_classes_analysis.png")
    print("  • 08_sequence_characteristics.png")
    print("  • summary_statistics.txt")
    print("  • summary_report.html")

if __name__ == "__main__":
    main()

