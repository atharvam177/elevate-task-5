"""
Task 5: Decision Trees and Random Forests - ENHANCED VERSION
============================================================
Objective: Learn tree-based models with STUNNING visualizations
Tools: Scikit-learn, Graphviz, Advanced Matplotlib/Seaborn
Dataset: Heart Disease Dataset (Auto-downloaded from Kaggle)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Advanced styling configuration
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Custom color palettes
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#06A77D',
    'danger': '#D90368',
    'warning': '#F77F00',
    'info': '#4CC9F0',
    'dark': '#2B2D42',
    'light': '#EDF2F4',
    'gradient1': ['#667eea', '#764ba2'],
    'gradient2': ['#f093fb', '#f5576c'],
    'gradient3': ['#4facfe', '#00f2fe'],
    'gradient4': ['#43e97b', '#38f9d7'],
    'gradient5': ['#fa709a', '#fee140'],
    'categorical': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2']
}

# =============================================================================
# ENHANCED PLOTTING UTILITIES
# =============================================================================

def create_gradient_background(ax, colors, direction='vertical'):
    """Create a gradient background for plots"""
    gradient = np.linspace(0, 1, 256).reshape(-1, 1) if direction == 'vertical' else np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=plt.cm.colors.LinearSegmentedColormap.from_list('', colors),
              extent=[ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]], 
              zorder=0, alpha=0.3)

def add_value_labels(ax, spacing=5, format_str='{:.2f}'):
    """Add value labels on top of bars"""
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        label = format_str.format(y_value)
        ax.annotate(label, (x_value, y_value), xytext=(0, spacing),
                   textcoords="offset points", ha='center', va='bottom',
                   fontweight='bold', fontsize=11, color='#2B2D42')

def style_axis(ax, title, xlabel='', ylabel='', title_color=COLORS['dark']):
    """Apply modern styling to axis"""
    ax.set_title(title, fontsize=16, fontweight='bold', color=title_color, pad=20)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=13, fontweight='600', color=COLORS['dark'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, fontweight='600', color=COLORS['dark'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors=COLORS['dark'])
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.8)

def add_watermark(fig):
    """Add a professional watermark"""
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | @Atharva0177', 
             ha='right', va='bottom', fontsize=8, color='gray', alpha=0.6, style='italic')

# =============================================================================
# KAGGLE DATASET DOWNLOAD
# =============================================================================

def download_kaggle_dataset():
    """Download Heart Disease dataset directly from Kaggle"""
    print("=" * 80)
    print("📥 DOWNLOADING DATASET FROM KAGGLE")
    print("=" * 80)
    
    try:
        import kaggle
        print("\n✓ Kaggle API found")
    except ImportError:
        print("\n⚠ Kaggle package not found. Installing...")
        os.system('pip install kaggle')
        import kaggle
        print("✓ Kaggle package installed successfully")
    
    kaggle_json_path = os.path.expanduser('~/.kaggle/kaggle.json')
    if not os.path.exists(kaggle_json_path):
        print("\n" + "!" * 80)
        print("⚠ KAGGLE API SETUP REQUIRED")
        print("!" * 80)
        print("\nPlease follow these steps:")
        print("1. Go to https://www.kaggle.com/settings/account")
        print("2. Scroll down to 'API' section")
        print("3. Click 'Create New Token' - this will download kaggle.json")
        print("4. Move kaggle.json to: ~/.kaggle/kaggle.json (Linux/Mac)")
        print("   Or: C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json (Windows)")
        print("5. Run this script again")
        print("!" * 80)
        
        kaggle_dir = os.path.expanduser('~/.kaggle')
        if not os.path.exists(kaggle_dir):
            os.makedirs(kaggle_dir)
            print(f"\n✓ Created directory: {kaggle_dir}")
            print(f"Now place your kaggle.json file in: {kaggle_dir}")
        
        return None
    
    print(f"✓ Kaggle credentials found at: {kaggle_json_path}")
    
    if os.name != 'nt':
        os.chmod(kaggle_json_path, 0o600)
    
    dataset_name = "johnsmith88/heart-disease-dataset"
    download_path = "./heart_disease_data"
    
    print(f"\n📦 Downloading dataset: {dataset_name}")
    print(f"📂 Destination: {download_path}")
    
    try:
        if not os.path.exists(download_path):
            os.makedirs(download_path)
        
        kaggle.api.dataset_download_files(
            dataset_name, 
            path=download_path, 
            unzip=True,
            quiet=False
        )
        
        print("✓ Dataset downloaded successfully!")
        
        csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
        
        if csv_files:
            csv_path = os.path.join(download_path, csv_files[0])
            print(f"✓ Found CSV file: {csv_files[0]}")
            return csv_path
        else:
            print("✗ No CSV file found in downloaded data")
            return None
            
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Ensure you have accepted the dataset terms on Kaggle website")
        print("2. Visit: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
        print("3. Click 'Download' button to accept terms")
        print("4. Run this script again")
        return None

# =============================================================================
# STEP 1: ENHANCED DATA EXPLORATION
# =============================================================================

def load_and_explore_data(filepath=None):
    """Load and explore dataset with enhanced visualizations"""
    print("\n" + "=" * 80)
    print("📊 LOADING AND EXPLORING DATASET")
    print("=" * 80)
    
    if filepath is None:
        filepath = download_kaggle_dataset()
        if filepath is None:
            print("\n✗ Failed to download dataset. Exiting...")
            return None
    
    try:
        df = pd.read_csv(filepath)
        print(f"\n✓ Dataset loaded successfully from: {filepath}")
    except Exception as e:
        print(f"\n✗ Error loading dataset: {str(e)}")
        return None
    
    print(f"\n📈 Dataset Shape: {df.shape}")
    print(f"📋 Samples: {df.shape[0]} | Features: {df.shape[1]}")
    
    print("\n📝 Column Names:")
    print(df.columns.tolist())
    
    print("\n🔍 First 5 rows:")
    print(df.head())
    
    print("\n📊 Target Distribution:")
    print(df['target'].value_counts())
    
    # ENHANCED VISUALIZATION 1: Target Distribution with Style
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # Bar plot
    ax1 = fig.add_subplot(gs[0, 0])
    target_counts = df['target'].value_counts()
    bars = ax1.bar(['No Disease', 'Disease'], target_counts.values, 
                   color=[COLORS['success'], COLORS['danger']], 
                   edgecolor='white', linewidth=3, width=0.6)
    
    # Add gradient effect
    for i, bar in enumerate(bars):
        bar.set_alpha(0.8)
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=13, fontweight='bold', color=COLORS['dark'])
    
    style_axis(ax1, '❤️ Heart Disease Distribution', xlabel='Diagnosis', ylabel='Number of Patients')
    ax1.set_ylim(0, max(target_counts.values) * 1.15)
    
    # Pie chart with explosion
    ax2 = fig.add_subplot(gs[0, 1])
    colors_pie = [COLORS['success'], COLORS['danger']]
    explode = (0.05, 0.05)
    wedges, texts, autotexts = ax2.pie(target_counts.values, 
                                         labels=['No Disease', 'Disease'],
                                         autopct='%1.1f%%',
                                         startangle=90,
                                         colors=colors_pie,
                                         explode=explode,
                                         shadow=True,
                                         textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(13)
        autotext.set_fontweight('bold')
    
    ax2.set_title('Class Distribution', fontsize=16, fontweight='bold', color=COLORS['dark'], pad=20)
    
    plt.suptitle('📊 Dataset Overview - Heart Disease Classification', 
                 fontsize=18, fontweight='bold', color=COLORS['dark'], y=1.02)
    add_watermark(fig)
    plt.savefig('01_target_distribution_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # ENHANCED VISUALIZATION 2: Feature Correlation Heatmap
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Calculate correlation
    corr_matrix = df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Custom colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                annot=True, fmt='.2f', annot_kws={'size': 9},
                ax=ax)
    
    style_axis(ax, '🔥 Feature Correlation Matrix', xlabel='', ylabel='')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    add_watermark(fig)
    plt.tight_layout()
    plt.savefig('02_correlation_heatmap_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return df

# =============================================================================
# STEP 2: Data Preprocessing
# =============================================================================

def preprocess_data(df):
    """Preprocess the dataset"""
    print("\n" + "=" * 80)
    print("⚙️ DATA PREPROCESSING")
    print("=" * 80)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n✓ Training set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, feature_names

# =============================================================================
# STEP 3: ENHANCED DECISION TREE VISUALIZATION
# =============================================================================

def train_and_visualize_decision_tree(X_train, X_test, y_train, y_test, feature_names):
    """Train Decision Tree with stunning visualizations"""
    print("\n" + "=" * 80)
    print("🌳 STEP 1: DECISION TREE CLASSIFIER - ENHANCED")
    print("=" * 80)
    
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    
    y_pred_train = dt_classifier.predict(X_train)
    y_pred_test = dt_classifier.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\n📊 Decision Tree Performance:")
    print(f"   ├─ Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   ├─ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   ├─ Tree Depth: {dt_classifier.get_depth()}")
    print(f"   └─ Number of Leaves: {dt_classifier.get_n_leaves()}")
    
    # ENHANCED VISUALIZATION: Decision Tree
    fig, ax = plt.subplots(figsize=(28, 16), facecolor='#F8F9FA')
    
    plot_tree(
        dt_classifier, 
        feature_names=feature_names,
        class_names=['No Disease', 'Disease'],
        filled=True,
        rounded=True,
        fontsize=11,
        ax=ax,
        proportion=True,
        precision=2
    )
    
    ax.set_title('🌳 Decision Tree - Complete Visualization\n'
                 f'Depth: {dt_classifier.get_depth()} | Leaves: {dt_classifier.get_n_leaves()} | '
                 f'Test Accuracy: {test_accuracy:.2%}',
                 fontsize=22, fontweight='bold', color=COLORS['dark'], pad=30)
    
    add_watermark(fig)
    plt.tight_layout()
    plt.savefig('03_decision_tree_full_enhanced.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.show()
    
    # Limited depth version
    fig, ax = plt.subplots(figsize=(24, 14), facecolor='#F8F9FA')
    
    plot_tree(
        dt_classifier, 
        feature_names=feature_names,
        class_names=['No Disease', 'Disease'],
        filled=True,
        rounded=True,
        fontsize=12,
        ax=ax,
        max_depth=3,
        proportion=True,
        precision=2
    )
    
    ax.set_title('🌳 Decision Tree - Simplified View (Depth Limited to 3)\n'
                 'Easier to interpret the top decision rules',
                 fontsize=22, fontweight='bold', color=COLORS['dark'], pad=30)
    
    add_watermark(fig)
    plt.tight_layout()
    plt.savefig('04_decision_tree_limited_enhanced.png', dpi=300, bbox_inches='tight', facecolor='#F8F9FA')
    plt.show()
    
    # ENHANCED CONFUSION MATRIX
    cm = confusion_matrix(y_test, y_pred_test)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap
    cmap = sns.light_palette(COLORS['primary'], as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                cbar_kws={'label': 'Count'},
                linewidths=2, linecolor='white',
                annot_kws={'size': 18, 'weight': 'bold'},
                ax=ax)
    
    # Add percentage annotations
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / cm.sum() * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                   ha='center', va='center', fontsize=12, color='gray', style='italic')
    
    style_axis(ax, '🎯 Decision Tree - Confusion Matrix', 
               xlabel='Predicted Label', ylabel='True Label')
    
    # Add metrics box
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
    recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}'
    ax.text(1.3, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.8, edgecolor=COLORS['primary'], linewidth=2))
    
    add_watermark(fig)
    plt.tight_layout()
    plt.savefig('05_dt_confusion_matrix_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['No Disease', 'Disease']))
    
    return dt_classifier

# =============================================================================
# STEP 4: ENHANCED OVERFITTING ANALYSIS
# =============================================================================

def analyze_overfitting(X_train, X_test, y_train, y_test):
    """Analyze overfitting with beautiful visualizations"""
    print("\n" + "=" * 80)
    print("📈 STEP 2: OVERFITTING ANALYSIS - ENHANCED")
    print("=" * 80)
    
    max_depths = range(1, 21)
    train_accuracies = []
    test_accuracies = []
    
    for depth in max_depths:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, dt.predict(X_train))
        test_acc = accuracy_score(y_test, dt.predict(X_test))
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
    
    # ENHANCED VISUALIZATION
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot with gradient fill
    ax.plot(max_depths, train_accuracies, label='Training Accuracy', 
            marker='o', linewidth=3, markersize=8, color=COLORS['primary'], zorder=3)
    ax.plot(max_depths, test_accuracies, label='Test Accuracy', 
            marker='s', linewidth=3, markersize=8, color=COLORS['danger'], zorder=3)
    
    # Fill between
    ax.fill_between(max_depths, train_accuracies, alpha=0.3, color=COLORS['primary'])
    ax.fill_between(max_depths, test_accuracies, alpha=0.3, color=COLORS['danger'])
    
    # Mark best point
    best_depth_idx = np.argmax(test_accuracies)
    best_depth = list(max_depths)[best_depth_idx]
    best_acc = test_accuracies[best_depth_idx]
    
    ax.scatter([best_depth], [best_acc], s=300, color=COLORS['warning'], 
              edgecolors='white', linewidths=3, zorder=5, marker='*')
    ax.annotate(f'Best Depth: {best_depth}\nAccuracy: {best_acc:.4f}',
               xy=(best_depth, best_acc), xytext=(best_depth+3, best_acc-0.05),
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['warning'], alpha=0.8, edgecolor='white', linewidth=2),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='white', lw=2))
    
    style_axis(ax, '📊 Overfitting Analysis: Training vs Test Accuracy', 
              xlabel='Maximum Tree Depth', ylabel='Accuracy Score')
    
    ax.legend(loc='lower right', frameon=True, shadow=True, fancybox=True, fontsize=12)
    ax.set_ylim([0.65, 1.05])
    ax.set_xlim([0, 21])
    
    # Add overfitting zone
    ax.axvspan(15, 21, alpha=0.1, color='red', label='_Overfitting Zone')
    ax.text(18, 0.68, 'Overfitting\nZone', ha='center', fontsize=10, 
           style='italic', color='red', weight='bold')
    
    add_watermark(fig)
    plt.tight_layout()
    plt.savefig('06_overfitting_analysis_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"\n🎯 Optimal Configuration:")
    print(f"   ├─ Best Max Depth: {best_depth}")
    print(f"   ├─ Best Test Accuracy: {test_accuracies[best_depth_idx]:.4f}")
    print(f"   └─ Training Accuracy at Best Depth: {train_accuracies[best_depth_idx]:.4f}")
    
    # Grid Search
    print("\n🔍 Performing Grid Search for Optimal Hyperparameters...")
    
    param_grid = {
        'max_depth': [3, 5, 7, 9, 11],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']
    }
    
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✓ Best Parameters: {grid_search.best_params_}")
    print(f"✓ Best CV Score: {grid_search.best_score_:.4f}")
    
    optimized_dt = grid_search.best_estimator_
    test_acc = accuracy_score(y_test, optimized_dt.predict(X_test))
    print(f"✓ Test Accuracy (Optimized): {test_acc:.4f}")
    
    return optimized_dt

# =============================================================================
# STEP 5: ENHANCED RANDOM FOREST COMPARISON
# =============================================================================

def train_random_forest(X_train, X_test, y_train, y_test, dt_classifier):
    """Train Random Forest with stunning comparisons"""
    print("\n" + "=" * 80)
    print("🌲 STEP 3: RANDOM FOREST - ENHANCED")
    print("=" * 80)
    
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    
    y_pred_train_rf = rf_classifier.predict(X_train)
    y_pred_test_rf = rf_classifier.predict(X_test)
    y_pred_test_dt = dt_classifier.predict(X_test)
    
    train_acc_rf = accuracy_score(y_train, y_pred_train_rf)
    test_acc_rf = accuracy_score(y_test, y_pred_test_rf)
    test_acc_dt = accuracy_score(y_test, y_pred_test_dt)
    
    print(f"\n📊 Random Forest Performance:")
    print(f"   ├─ Training Accuracy: {train_acc_rf:.4f} ({train_acc_rf*100:.2f}%)")
    print(f"   └─ Test Accuracy: {test_acc_rf:.4f} ({test_acc_rf*100:.2f}%)")
    
    print(f"\n📊 Decision Tree Performance:")
    print(f"   └─ Test Accuracy: {test_acc_dt:.4f} ({test_acc_dt*100:.2f}%)")
    
    improvement = test_acc_rf - test_acc_dt
    print(f"\n✨ Improvement: {improvement:.4f} ({(improvement / test_acc_dt * 100):.2f}%)")
    
    # ENHANCED COMPARISON VISUALIZATION
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Bar Comparison
    ax1 = fig.add_subplot(gs[0, :])
    models = ['Decision Tree', 'Random Forest']
    accuracies = [test_acc_dt, test_acc_rf]
    colors_bars = [COLORS['primary'], COLORS['success']]
    
    bars = ax1.bar(models, accuracies, color=colors_bars, width=0.5, 
                   edgecolor='white', linewidth=3, alpha=0.9)
    
    # Add gradient effect with patterns
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Add value on top
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}\n({height*100:.2f}%)',
                ha='center', va='bottom', fontsize=14, fontweight='bold', color=COLORS['dark'])
        
        # Add emoji icons
        emoji = '🌳' if i == 0 else '🌲'
        ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                emoji, ha='center', va='center', fontsize=40, alpha=0.3)
    
    # Add improvement arrow
    ax1.annotate('', xy=(1, test_acc_rf), xytext=(0, test_acc_dt),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLORS['success']))
    ax1.text(0.5, (test_acc_dt + test_acc_rf)/2, 
            f'+{improvement:.4f}\n({(improvement / test_acc_dt * 100):.2f}%)',
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=COLORS['success'], alpha=0.3, edgecolor=COLORS['success'], linewidth=2))
    
    style_axis(ax1, '🏆 Model Comparison: Accuracy Showdown', 
              xlabel='Model Type', ylabel='Test Accuracy')
    ax1.set_ylim([0.7, 1.0])
    
    # 2. Confusion Matrix - Decision Tree
    ax2 = fig.add_subplot(gs[1, 0])
    cm_dt = confusion_matrix(y_test, y_pred_test_dt)
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                cbar=False, linewidths=2, linecolor='white',
                annot_kws={'size': 16, 'weight': 'bold'}, ax=ax2)
    style_axis(ax2, '🌳 Decision Tree\nConfusion Matrix', xlabel='Predicted', ylabel='Actual')
    
    # 3. Confusion Matrix - Random Forest
    ax3 = fig.add_subplot(gs[1, 1])
    cm_rf = confusion_matrix(y_test, y_pred_test_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                cbar=False, linewidths=2, linecolor='white',
                annot_kws={'size': 16, 'weight': 'bold'}, ax=ax3)
    style_axis(ax3, '🌲 Random Forest\nConfusion Matrix', xlabel='Predicted', ylabel='Actual')
    
    plt.suptitle('📊 Comprehensive Model Comparison', fontsize=20, fontweight='bold', 
                color=COLORS['dark'], y=0.98)
    add_watermark(fig)
    plt.savefig('07_model_comparison_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # ENHANCED ROC CURVE
    y_pred_proba_dt = dt_classifier.predict_proba(X_test)[:, 1]
    y_pred_proba_rf = rf_classifier.predict_proba(X_test)[:, 1]
    
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
    
    auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
    auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot curves with fill
    ax.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.3f})', 
            linewidth=4, color=COLORS['primary'], marker='o', markersize=4, 
            markevery=10, alpha=0.9)
    ax.fill_between(fpr_dt, tpr_dt, alpha=0.2, color=COLORS['primary'])
    
    ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.3f})', 
            linewidth=4, color=COLORS['success'], marker='s', markersize=4, 
            markevery=10, alpha=0.9)
    ax.fill_between(fpr_rf, tpr_rf, alpha=0.2, color=COLORS['success'])
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)', alpha=0.6)
    
    # Add perfect classifier line
    ax.plot([0, 0, 1], [0, 1, 1], ':', linewidth=2, color='gold', 
           label='Perfect Classifier (AUC = 1.000)', alpha=0.7)
    
    style_axis(ax, '📈 ROC Curve Comparison - Model Performance Analysis', 
              xlabel='False Positive Rate (1 - Specificity)', 
              ylabel='True Positive Rate (Sensitivity)')
    
    ax.legend(loc='lower right', frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    # Add interpretation box
    interp_text = ('Higher AUC = Better Model\n'
                  'Closer to top-left = Better\n'
                  f'Winner: {"Random Forest" if auc_rf > auc_dt else "Decision Tree"} 🏆')
    ax.text(0.98, 0.02, interp_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, 
                     edgecolor=COLORS['warning'], linewidth=2))
    
    add_watermark(fig)
    plt.tight_layout()
    plt.savefig('08_roc_curve_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("\n📋 Random Forest Classification Report:")
    print(classification_report(y_test, y_pred_test_rf, target_names=['No Disease', 'Disease']))
    
    return rf_classifier

# =============================================================================
# STEP 6: ENHANCED FEATURE IMPORTANCE
# =============================================================================

def interpret_feature_importance(rf_classifier, dt_classifier, feature_names):
    """Interpret feature importances with stunning visuals"""
    print("\n" + "=" * 80)
    print("🎯 STEP 4: FEATURE IMPORTANCE ANALYSIS - ENHANCED")
    print("=" * 80)
    
    rf_importances = rf_classifier.feature_importances_
    dt_importances = dt_classifier.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Random Forest': rf_importances,
        'Decision Tree': dt_importances
    }).sort_values('Random Forest', ascending=False)
    
    print("\n📊 Feature Importances Ranking:")
    print(importance_df.to_string(index=False))
    
    # ENHANCED VISUALIZATION 1: Horizontal Bar Chart with Gradient
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    
    # Random Forest
    importance_df_rf = importance_df.sort_values('Random Forest', ascending=True)
    colors_rf = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    
    bars_rf = axes[0].barh(importance_df_rf['Feature'], importance_df_rf['Random Forest'], 
                           color=colors_rf, edgecolor='white', linewidth=2, height=0.7)
    
    # Add values
    for i, (bar, val) in enumerate(zip(bars_rf, importance_df_rf['Random Forest'])):
        axes[0].text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center', fontsize=10, fontweight='bold', color=COLORS['dark'])
        # Add ranking
        axes[0].text(-0.002, bar.get_y() + bar.get_height()/2, 
                    f'#{len(feature_names)-i}', va='center', ha='right',
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='circle', facecolor=colors_rf[i], edgecolor='white'))
    
    style_axis(axes[0], '🌲 Random Forest Feature Importance', xlabel='Importance Score', ylabel='')
    axes[0].set_xlim([0, max(importance_df_rf['Random Forest']) * 1.15])
    
    # Decision Tree
    importance_df_dt = importance_df.sort_values('Decision Tree', ascending=True)
    colors_dt = plt.cm.plasma(np.linspace(0.3, 0.9, len(feature_names)))
    
    bars_dt = axes[1].barh(importance_df_dt['Feature'], importance_df_dt['Decision Tree'], 
                           color=colors_dt, edgecolor='white', linewidth=2, height=0.7)
    
    # Add values
    for i, (bar, val) in enumerate(zip(bars_dt, importance_df_dt['Decision Tree'])):
        axes[1].text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center', fontsize=10, fontweight='bold', color=COLORS['dark'])
        # Add ranking
        axes[1].text(-0.002, bar.get_y() + bar.get_height()/2, 
                    f'#{len(feature_names)-i}', va='center', ha='right',
                    fontsize=9, fontweight='bold', color='white',
                    bbox=dict(boxstyle='circle', facecolor=colors_dt[i], edgecolor='white'))
    
    style_axis(axes[1], '🌳 Decision Tree Feature Importance', xlabel='Importance Score', ylabel='')
    axes[1].set_xlim([0, max(importance_df_dt['Decision Tree']) * 1.15])
    
    plt.suptitle('🎯 Feature Importance Analysis - What Matters Most?', 
                fontsize=20, fontweight='bold', color=COLORS['dark'], y=0.98)
    add_watermark(fig)
    plt.tight_layout()
    plt.savefig('09_feature_importance_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # ENHANCED VISUALIZATION 2: Lollipop Chart
    fig, ax = plt.subplots(figsize=(14, 10))
    
    importance_df_sorted = importance_df.sort_values('Random Forest', ascending=True)
    y_pos = np.arange(len(importance_df_sorted))
    
    # Plot lollipops for Random Forest
    ax.hlines(y=y_pos, xmin=0, xmax=importance_df_sorted['Random Forest'], 
             color=COLORS['success'], alpha=0.8, linewidth=3)
    ax.scatter(importance_df_sorted['Random Forest'], y_pos, 
              s=300, color=COLORS['success'], alpha=0.9, edgecolors='white', linewidth=3, zorder=3)
    
    # Plot lollipops for Decision Tree
    ax.hlines(y=y_pos, xmin=0, xmax=importance_df_sorted['Decision Tree'], 
             color=COLORS['danger'], alpha=0.5, linewidth=2, linestyle='--')
    ax.scatter(importance_df_sorted['Decision Tree'], y_pos, 
              s=200, color=COLORS['danger'], alpha=0.7, edgecolors='white', linewidth=2, 
              zorder=2, marker='D')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df_sorted['Feature'])
    
    style_axis(ax, '🍭 Feature Importance Lollipop Chart', 
              xlabel='Importance Score', ylabel='')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['success'], 
              markersize=12, label='Random Forest'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['danger'], 
              markersize=10, label='Decision Tree')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, 
             shadow=True, fancybox=True, fontsize=11)
    
    add_watermark(fig)
    plt.tight_layout()
    plt.savefig('10_feature_importance_lollipop_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Top 5 Features
    print("\n🏆 Top 5 Most Important Features (Random Forest):")
    for idx, row in importance_df.head(5).iterrows():
        print(f"   {idx+1}. {row['Feature']}: {row['Random Forest']:.4f}")

# =============================================================================
# STEP 7: ENHANCED CROSS-VALIDATION
# =============================================================================

def cross_validation_evaluation(X_train, y_train, dt_classifier, rf_classifier):
    """Cross-validation with beautiful visualizations"""
    print("\n" + "=" * 80)
    print("✅ STEP 5: CROSS-VALIDATION ANALYSIS - ENHANCED")
    print("=" * 80)
    
    X = X_train
    y = y_train
    
    # Cross-validation
    print("\n🔄 Performing 5-Fold Cross-Validation...")
    dt_cv_scores = cross_val_score(dt_classifier, X, y, cv=5, scoring='accuracy')
    rf_cv_scores = cross_val_score(rf_classifier, X, y, cv=5, scoring='accuracy')
    
    print(f"\n🌳 Decision Tree CV Scores: {dt_cv_scores}")
    print(f"   Mean: {dt_cv_scores.mean():.4f} (+/- {dt_cv_scores.std() * 2:.4f})")
    
    print(f"\n🌲 Random Forest CV Scores: {rf_cv_scores}")
    print(f"   Mean: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std() * 2:.4f})")
    
    # ENHANCED VISUALIZATION 1: Box Plot with Violin
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Box Plot
    ax1 = axes[0]
    bp = ax1.boxplot([dt_cv_scores, rf_cv_scores],
                     labels=['Decision Tree', 'Random Forest'],
                     patch_artist=True,
                     widths=0.5,
                     boxprops=dict(linewidth=2, edgecolor='white'),
                     medianprops=dict(linewidth=3, color='red'),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2))
    
    colors = [COLORS['primary'], COLORS['success']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add mean markers
    means = [dt_cv_scores.mean(), rf_cv_scores.mean()]
    ax1.plot([1, 2], means, 'D', color='gold', markersize=15, label='Mean', 
            zorder=3, markeredgecolor='white', markeredgewidth=2)
    
    # Add data points
    for i, scores in enumerate([dt_cv_scores, rf_cv_scores], 1):
        x = np.random.normal(i, 0.04, size=len(scores))
        ax1.scatter(x, scores, alpha=0.6, s=100, color='white', edgecolors='black', linewidth=1.5, zorder=2)
    
    style_axis(ax1, '📦 Cross-Validation Score Distribution', xlabel='Model', ylabel='Accuracy Score')
    ax1.legend(fontsize=11)
    ax1.set_ylim([0.7, 1.0])
    
    # Violin Plot
    ax2 = axes[1]
    parts = ax2.violinplot([dt_cv_scores, rf_cv_scores], positions=[1, 2],
                           widths=0.6, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor('white')
        pc.set_linewidth(2)
    
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor('white')
            vp.set_linewidth(2)
    
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(['Decision Tree', 'Random Forest'])
    
    style_axis(ax2, '🎻 Cross-Validation Score Distribution (Violin)', 
              xlabel='Model', ylabel='Accuracy Score')
    ax2.set_ylim([0.7, 1.0])
    
    plt.suptitle('📊 5-Fold Cross-Validation Results - Model Stability Analysis', 
                fontsize=20, fontweight='bold', color=COLORS['dark'], y=0.98)
    add_watermark(fig)
    plt.tight_layout()
    plt.savefig('11_cross_validation_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Multiple Metrics
    print("\n📊 Evaluating Multiple Metrics...")
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    dt_scores = {}
    rf_scores = {}
    
    for metric in metrics:
        dt_scores[metric] = cross_val_score(dt_classifier, X, y, cv=5, scoring=metric).mean()
        rf_scores[metric] = cross_val_score(rf_classifier, X, y, cv=5, scoring=metric).mean()
    
    print("\n🌳 Decision Tree Scores:")
    for metric, score in dt_scores.items():
        print(f"   {metric.capitalize()}: {score:.4f}")
    
    print("\n🌲 Random Forest Scores:")
    for metric, score in rf_scores.items():
        print(f"   {metric.capitalize()}: {score:.4f}")
    
    # ENHANCED VISUALIZATION 2: Radar Chart
    fig = plt.figure(figsize=(16, 8))
    
    # Subplot 1: Bar Chart
    ax1 = plt.subplot(1, 2, 1)
    metrics_df = pd.DataFrame({
        'Decision Tree': list(dt_scores.values()),
        'Random Forest': list(rf_scores.values())
    }, index=[m.capitalize() for m in metrics])
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, metrics_df['Decision Tree'], width, 
                   label='Decision Tree', color=COLORS['primary'], alpha=0.8, edgecolor='white', linewidth=2)
    bars2 = ax1.bar(x + width/2, metrics_df['Random Forest'], width, 
                   label='Random Forest', color=COLORS['success'], alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.capitalize() for m in metrics], rotation=15, ha='right')
    style_axis(ax1, '📊 Multi-Metric Performance Comparison', xlabel='Metric', ylabel='Score')
    ax1.legend(loc='lower right', frameon=True, shadow=True, fancybox=True)
    ax1.set_ylim([0.7, 1.05])
    
    # Subplot 2: Radar Chart
    ax2 = plt.subplot(1, 2, 2, projection='polar')
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    dt_values = list(dt_scores.values())
    rf_values = list(rf_scores.values())
    
    # Close the plot
    angles += angles[:1]
    dt_values += dt_values[:1]
    rf_values += rf_values[:1]
    
    ax2.plot(angles, dt_values, 'o-', linewidth=3, label='Decision Tree', 
            color=COLORS['primary'], markersize=10)
    ax2.fill(angles, dt_values, alpha=0.25, color=COLORS['primary'])
    
    ax2.plot(angles, rf_values, 's-', linewidth=3, label='Random Forest', 
            color=COLORS['success'], markersize=10)
    ax2.fill(angles, rf_values, alpha=0.25, color=COLORS['success'])
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([m.capitalize() for m in metrics], fontsize=11, fontweight='bold')
    ax2.set_ylim(0.7, 1.0)
    ax2.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax2.set_yticklabels(['0.70', '0.80', '0.90', '1.00'], fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, shadow=True)
    ax2.set_title('🎯 Radar Chart', fontsize=14, fontweight='bold', color=COLORS['dark'], pad=20)
    
    plt.suptitle('📈 Comprehensive Multi-Metric Evaluation', 
                fontsize=20, fontweight='bold', color=COLORS['dark'], y=0.98)
    add_watermark(fig)
    plt.tight_layout()
    plt.savefig('12_multi_metric_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# =============================================================================
# FINAL SUMMARY DASHBOARD
# =============================================================================

def create_summary_dashboard(dt_classifier, rf_classifier, X_test, y_test, feature_names):
    """Create a comprehensive summary dashboard"""
    print("\n" + "=" * 80)
    print("📊 CREATING FINAL SUMMARY DASHBOARD")
    print("=" * 80)
    
    # Calculate all metrics
    dt_pred = dt_classifier.predict(X_test)
    rf_pred = rf_classifier.predict(X_test)
    
    dt_acc = accuracy_score(y_test, dt_pred)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    dt_cm = confusion_matrix(y_test, dt_pred)
    rf_cm = confusion_matrix(y_test, rf_pred)
    
    # Create dashboard
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)
    
    # Title
    fig.suptitle('🎯 DECISION TREES vs RANDOM FORESTS - FINAL DASHBOARD\n'
                'Heart Disease Classification - Complete Analysis',
                fontsize=24, fontweight='bold', color=COLORS['dark'], y=0.98)
    
    # 1. Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Decision\nTree', 'Random\nForest']
    accs = [dt_acc, rf_acc]
    bars = ax1.bar(models, accs, color=[COLORS['primary'], COLORS['success']], 
                   width=0.6, edgecolor='white', linewidth=3, alpha=0.9)
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    style_axis(ax1, '🎯 Accuracy', xlabel='', ylabel='Score')
    ax1.set_ylim([0.7, 1.0])
    
    # 2. Feature Importance Top 5 (RF)
    ax2 = fig.add_subplot(gs[0, 1:])
    rf_importances = rf_classifier.feature_importances_
    top_indices = np.argsort(rf_importances)[-5:]
    top_features = [feature_names[i] for i in top_indices]
    top_values = rf_importances[top_indices]
    
    colors_grad = plt.cm.viridis(np.linspace(0.3, 0.9, 5))
    bars = ax2.barh(top_features, top_values, color=colors_grad, 
                    edgecolor='white', linewidth=2, height=0.6)
    for bar, val in zip(bars, top_values):
        ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    style_axis(ax2, '⭐ Top 5 Important Features (Random Forest)', xlabel='Importance', ylabel='')
    
    # 3. Confusion Matrix - DT
    ax3 = fig.add_subplot(gs[1, 0])
    sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                linewidths=2, linecolor='white', annot_kws={'size': 14, 'weight': 'bold'}, ax=ax3)
    style_axis(ax3, '🌳 DT Confusion Matrix', xlabel='Predicted', ylabel='Actual')
    
    # 4. Confusion Matrix - RF
    ax4 = fig.add_subplot(gs[1, 1])
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'],
                linewidths=2, linecolor='white', annot_kws={'size': 14, 'weight': 'bold'}, ax=ax4)
    style_axis(ax4, '🌲 RF Confusion Matrix', xlabel='Predicted', ylabel='Actual')
    
    # 5. Metrics Summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    
    dt_precision = dt_cm[1,1] / (dt_cm[1,1] + dt_cm[0,1]) if (dt_cm[1,1] + dt_cm[0,1]) > 0 else 0
    dt_recall = dt_cm[1,1] / (dt_cm[1,1] + dt_cm[1,0]) if (dt_cm[1,1] + dt_cm[1,0]) > 0 else 0
    dt_f1 = 2 * (dt_precision * dt_recall) / (dt_precision + dt_recall) if (dt_precision + dt_recall) > 0 else 0
    
    rf_precision = rf_cm[1,1] / (rf_cm[1,1] + rf_cm[0,1]) if (rf_cm[1,1] + rf_cm[0,1]) > 0 else 0
    rf_recall = rf_cm[1,1] / (rf_cm[1,1] + rf_cm[1,0]) if (rf_cm[1,1] + rf_cm[1,0]) > 0 else 0
    rf_f1 = 2 * (rf_precision * rf_recall) / (rf_precision + rf_recall) if (rf_precision + rf_recall) > 0 else 0
    
    summary_text = f"""
    📊 PERFORMANCE METRICS
    {'='*30}
    
    🌳 DECISION TREE:
    ├─ Accuracy:  {dt_acc:.3f}
    ├─ Precision: {dt_precision:.3f}
    ├─ Recall:    {dt_recall:.3f}
    └─ F1-Score:  {dt_f1:.3f}
    
    🌲 RANDOM FOREST:
    ├─ Accuracy:  {rf_acc:.3f}
    ├─ Precision: {rf_precision:.3f}
    ├─ Recall:    {rf_recall:.3f}
    └─ F1-Score:  {rf_f1:.3f}
    
    🏆 WINNER: {'RF' if rf_acc > dt_acc else 'DT'}
    Improvement: {abs(rf_acc - dt_acc):.3f}
    ({abs(rf_acc - dt_acc)/dt_acc*100:.1f}%)
    """
    
    ax5.text(0.1, 0.5, summary_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=COLORS['light'], alpha=0.9, 
                     edgecolor=COLORS['primary'], linewidth=3))
    
    # 6. ROC Curves
    ax6 = fig.add_subplot(gs[2, :])
    
    dt_proba = dt_classifier.predict_proba(X_test)[:, 1]
    rf_proba = rf_classifier.predict_proba(X_test)[:, 1]
    
    fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_proba)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
    
    auc_dt = roc_auc_score(y_test, dt_proba)
    auc_rf = roc_auc_score(y_test, rf_proba)
    
    ax6.plot(fpr_dt, tpr_dt, linewidth=4, color=COLORS['primary'], 
            label=f'🌳 Decision Tree (AUC = {auc_dt:.3f})', alpha=0.9)
    ax6.fill_between(fpr_dt, tpr_dt, alpha=0.2, color=COLORS['primary'])
    
    ax6.plot(fpr_rf, tpr_rf, linewidth=4, color=COLORS['success'], 
            label=f'🌲 Random Forest (AUC = {auc_rf:.3f})', alpha=0.9)
    ax6.fill_between(fpr_rf, tpr_rf, alpha=0.2, color=COLORS['success'])
    
    ax6.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random Classifier')
    
    style_axis(ax6, '📈 ROC Curve Comparison', 
              xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax6.legend(loc='lower right', frameon=True, shadow=True, fancybox=True, fontsize=11)
    ax6.set_xlim([-0.02, 1.02])
    ax6.set_ylim([-0.02, 1.02])
    
    add_watermark(fig)
    plt.savefig('13_final_dashboard_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("\n✓ Final dashboard created successfully!")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function with enhanced visualizations"""
    print("\n" + "=" * 80)
    print("🌟 DECISION TREES AND RANDOM FORESTS - ENHANCED VERSION")
    print("Heart Disease Classification with Stunning Visualizations")
    print("=" * 80)
    print(f"User: @Atharva0177")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Step 1: Load and explore data
    df = load_and_explore_data()
    
    if df is None:
        print("\n✗ Failed to load dataset. Please check the error messages above.")
        return
    
    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Step 3: Train and visualize decision tree
    dt_classifier = train_and_visualize_decision_tree(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # Step 4: Analyze overfitting and control tree depth
    optimized_dt = analyze_overfitting(X_train, X_test, y_train, y_test)
    
    # Step 5: Train random forest and compare
    rf_classifier = train_random_forest(
        X_train, X_test, y_train, y_test, optimized_dt
    )
    
    # Step 6: Interpret feature importances
    interpret_feature_importance(rf_classifier, optimized_dt, feature_names)
    
    # Step 7: Cross-validation evaluation
    cross_validation_evaluation(X_train, y_train, optimized_dt, rf_classifier)
    
    # Step 8: Create final summary dashboard
    create_summary_dashboard(optimized_dt, rf_classifier, X_test, y_test, feature_names)
    
    print("\n" + "=" * 80)
    print("✨ ANALYSIS COMPLETE! ✨")
    print("=" * 80)
    print("\n🎨 Generated Enhanced Visualizations:")
    print("   01. target_distribution_enhanced.png - Dataset overview")
    print("   02. correlation_heatmap_enhanced.png - Feature correlations")
    print("   03. decision_tree_full_enhanced.png - Complete tree structure")
    print("   04. decision_tree_limited_enhanced.png - Simplified tree (depth 3)")
    print("   05. dt_confusion_matrix_enhanced.png - DT confusion matrix")
    print("   06. overfitting_analysis_enhanced.png - Depth vs accuracy")
    print("   07. model_comparison_enhanced.png - DT vs RF comparison")
    print("   08. roc_curve_enhanced.png - ROC curves with AUC")
    print("   09. feature_importance_enhanced.png - Importance analysis")
    print("   10. feature_importance_lollipop_enhanced.png - Lollipop chart")
    print("   11. cross_validation_enhanced.png - CV results")
    print("   12. multi_metric_enhanced.png - Multi-metric evaluation")
    print("   13. final_dashboard_enhanced.png - Complete summary dashboard")
    
    print("\n🎉 All visualizations saved successfully!")
    print("📊 Total: 13 high-quality, professional-grade visualizations")
    print("\n💡 Key Improvements:")
    print("   ✓ Modern color schemes with gradients")
    print("   ✓ Enhanced typography and styling")
    print("   ✓ Professional annotations and labels")
    print("   ✓ Comprehensive dashboard view")
    print("   ✓ Publication-ready quality")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()