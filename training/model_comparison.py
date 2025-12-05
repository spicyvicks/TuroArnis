import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

def load_results():
    """Load all saved model results"""
    results = {}
    
    # Load MediaPipe results (from your existing trained model)
    mediapipe_path = Path('models/arnis_coordinates_classifier.keras')
    if mediapipe_path.exists():
        print("✓ Found MediaPipe model")
        # You'll need to evaluate MediaPipe on the same test set
        # For now, load from a results file if it exists
        mp_results_path = Path('models/mediapipe_results.json')
        if mp_results_path.exists():
            with open(mp_results_path, 'r') as f:
                results['MediaPipe'] = json.load(f)
        else:
            print("  ⚠ No mediapipe_results.json - need to evaluate MediaPipe")
    
    # Load MoveNet results
    movenet_path = Path('models/movenet_results.json')
    if movenet_path.exists():
        with open(movenet_path, 'r') as f:
            results['MoveNet Thunder'] = json.load(f)
        print("✓ Loaded MoveNet Thunder results")
    
    # Load PoseNet results
    posenet_path = Path('models/posenet_results.json')
    if posenet_path.exists():
        with open(posenet_path, 'r') as f:
            results['PoseNet Lightning'] = json.load(f)
        print("✓ Loaded PoseNet Lightning results")
    
    return results


def plot_metrics_comparison(results):
    """Create bar chart comparing accuracy, precision, recall, F1"""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    model_names = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = {'MediaPipe': '#3498db', 'MoveNet Thunder': '#e74c3c', 'PoseNet Lightning': '#2ecc71'}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        values = [results[model].get(metric, 0) for model in model_names]
        bars = ax.bar(model_names, values, color=[colors.get(m, '#95a5a6') for m in model_names])
        
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylim([0, 1.0])
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=13, fontweight='bold')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plots/metrics_comparison.png")


def plot_speed_comparison(results):
    """Compare inference speed and FPS"""
    model_names = list(results.keys())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Inference Speed Comparison', fontsize=16, fontweight='bold')
    
    colors = {'MediaPipe': '#3498db', 'MoveNet Thunder': '#e74c3c', 'PoseNet Lightning': '#2ecc71'}
    
    # Inference time
    inference_times = [results[model].get('avg_inference_time_ms', 0) for model in model_names]
    bars1 = ax1.bar(model_names, inference_times, color=[colors.get(m, '#95a5a6') for m in model_names])
    ax1.set_ylabel('Milliseconds', fontsize=12)
    ax1.set_title('Average Inference Time (ms)', fontsize=13, fontweight='bold')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=10)
    
    # FPS
    fps_values = [results[model].get('fps', 0) for model in model_names]
    bars2 = ax2.bar(model_names, fps_values, color=[colors.get(m, '#95a5a6') for m in model_names])
    ax2.set_ylabel('Frames Per Second', fontsize=12)
    ax2.set_title('FPS (Frames Per Second)', fontsize=13, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('plots/speed_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plots/speed_comparison.png")


def plot_inference_time_line(results):
    """Line graph showing inference time per frame"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {'MediaPipe': '#3498db', 'MoveNet Thunder': '#e74c3c', 'PoseNet Lightning': '#2ecc71'}
    
    for model_name, data in results.items():
        if 'inference_times' in data:
            times = data['inference_times'][:100]  # First 100 frames
            ax.plot(range(len(times)), times, 
                   label=model_name, 
                   color=colors.get(model_name, '#95a5a6'),
                   linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Frame Number', fontsize=12)
    ax.set_ylabel('Inference Time (ms)', fontsize=12)
    ax.set_title('Inference Time Per Frame (First 100 Frames)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/inference_time_line.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plots/inference_time_line.png")


def plot_radar_chart(results):
    """Radar chart comparing multiple metrics"""
    from math import pi
    
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    model_names = list(results.keys())
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = {'MediaPipe': '#3498db', 'MoveNet Thunder': '#e74c3c', 'PoseNet Lightning': '#2ecc71'}
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    
    for model_name, data in results.items():
        values = [
            data.get('accuracy', 0),
            data.get('precision', 0),
            data.get('recall', 0),
            data.get('f1_score', 0)
        ]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=model_name, color=colors.get(model_name, '#95a5a6'))
        ax.fill(angles, values, alpha=0.15, color=colors.get(model_name, '#95a5a6'))
    
    ax.set_ylim(0, 1)
    ax.set_title('Performance Radar Chart', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/radar_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plots/radar_comparison.png")


def plot_summary_table(results):
    """Create a summary table with all metrics"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 
               'Inference (ms)', 'FPS', 'Training Time']
    
    table_data = []
    for model_name, data in results.items():
        row = [
            model_name,
            f"{data.get('accuracy', 0):.4f}",
            f"{data.get('precision', 0):.4f}",
            f"{data.get('recall', 0):.4f}",
            f"{data.get('f1_score', 0):.4f}",
            f"{data.get('avg_inference_time_ms', 0):.1f}",
            f"{data.get('fps', 0):.2f}",
            f"{data.get('training_time_s', 0)/60:.1f} min"
        ]
        table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.12, 0.08, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    colors = {'MediaPipe': '#ebf5fb', 'MoveNet Thunder': '#fadbd8', 'PoseNet Lightning': '#d4efdf'}
    for i, model_name in enumerate(results.keys(), 1):
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(colors.get(model_name, '#f2f2f2'))
    
    plt.title('Model Comparison Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('plots/summary_table.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plots/summary_table.png")


def plot_combined_training_history():
    """Plot combined training history from all models on same axes"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Training History Comparison - All Models', fontsize=16, fontweight='bold')
    
    colors = {'MediaPipe': '#3498db', 'MoveNet Thunder': '#e74c3c', 'PoseNet Lightning': '#2ecc71'}
    linestyles = {'MediaPipe': '-', 'MoveNet Thunder': '--', 'PoseNet Lightning': '-.'}
    
    history_files = {
        'MoveNet Thunder': Path('models/movenet_history.json'),
        'PoseNet Lightning': Path('models/posenet_history.json'),
        'MediaPipe': Path('models/mediapipe_history.json')
    }
    
    found_any = False
    
    # Plot accuracy (training and validation on same graph)
    for model_name, filepath in history_files.items():
        if filepath.exists():
            found_any = True
            with open(filepath, 'r') as f:
                history = json.load(f)
            
            color = colors.get(model_name, '#95a5a6')
            linestyle = linestyles.get(model_name, '-')
            
            if 'accuracy' in history:
                ax1.plot(history['accuracy'], label=f'{model_name} (Train)', 
                        color=color, linestyle=linestyle, linewidth=2.5, alpha=0.8)
            
            if 'val_accuracy' in history:
                ax1.plot(history['val_accuracy'], label=f'{model_name} (Val)',
                        color=color, linestyle=linestyle, linewidth=2, alpha=0.5)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Accuracy Over Epochs', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)  # Start from 0
    
    # Plot loss (training and validation on same graph)
    for model_name, filepath in history_files.items():
        if filepath.exists():
            with open(filepath, 'r') as f:
                history = json.load(f)
            
            color = colors.get(model_name, '#95a5a6')
            linestyle = linestyles.get(model_name, '-')
            
            if 'loss' in history:
                ax2.plot(history['loss'], label=f'{model_name} (Train)',
                        color=color, linestyle=linestyle, linewidth=2.5, alpha=0.8)
            
            if 'val_loss' in history:
                ax2.plot(history['val_loss'], label=f'{model_name} (Val)',
                        color=color, linestyle=linestyle, linewidth=2, alpha=0.5)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Loss Over Epochs', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    if found_any:
        plt.tight_layout()
        plt.savefig('plots/combined_training_history.png', dpi=300, bbox_inches='tight')
        print("✓ Saved plots/combined_training_history.png")
    else:
        print("⚠ No training history files found - skipping combined plot")
        plt.close(fig)


def main():
    print("="*60)
    print("MODEL COMPARISON - LOADING RESULTS")
    print("="*60)
    
    # Create plots directory
    Path('plots').mkdir(exist_ok=True)
    
    # Load results
    results = load_results()
    
    if len(results) < 2:
        print("\n⚠ Error: Need at least 2 models to compare")
        print("Make sure you've run train_movenet_posenet.py first")
        return
    
    print(f"\n✓ Loaded {len(results)} models for comparison")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_metrics_comparison(results)
    plot_speed_comparison(results)
    
    # Only plot line graph if we have inference times
    has_inference_times = any('inference_times' in data for data in results.values())
    if has_inference_times:
        plot_inference_time_line(results)
    else:
        print("⚠ Skipping inference time line (no per-frame data)")
    
    plot_radar_chart(results)
    plot_summary_table(results)
    plot_combined_training_history()  # NEW: Combined training history
    
    print("\n" + "="*60)
    print("✓ COMPARISON COMPLETE")
    print("="*60)
    print("\nGenerated plots in plots/ directory:")
    print("  - metrics_comparison.png")
    print("  - speed_comparison.png")
    if has_inference_times:
        print("  - inference_time_line.png")
    print("  - radar_comparison.png")
    print("  - summary_table.png")
    print("  - combined_training_history.png")  # NEW
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for model_name, data in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {data.get('accuracy', 0):.4f}")
        print(f"  F1-Score: {data.get('f1_score', 0):.4f}")
        print(f"  Inference: {data.get('avg_inference_time_ms', 0):.1f} ms")
        print(f"  FPS: {data.get('fps', 0):.2f}")


if __name__ == '__main__':
    main()
