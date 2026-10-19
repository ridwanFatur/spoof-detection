import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast
import timm
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    confusion_matrix,
    classification_report,
    roc_curve
)
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive metrics"""
    logger.info("\n" + "="*60)
    logger.info("CALCULATING METRICS")
    logger.info("="*60)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    # AUC
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc = 0.0
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Same as precision
    
    # False acceptance rate and false rejection rate
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Equal error rate (approximate)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'specificity': specificity,
        'npv': npv,
        'ppv': ppv,
        'far': far,
        'frr': frr,
        'eer': eer,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total_samples': len(y_true),
        'live_samples': int(np.sum(y_true)),
        'spoof_samples': int(len(y_true) - np.sum(y_true))
    }
    
    return metrics, cm

def print_metrics(metrics):
    """Pretty print metrics to stdout"""
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS")
    logger.info("="*60)
    
    logger.info(f"\nOverall Performance:")
    logger.info(f"  Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    logger.info(f"  AUC-ROC:         {metrics['auc']:.4f}")
    logger.info(f"  F1-Score:        {metrics['f1_score']:.4f}")
    
    logger.info(f"\nDetection Metrics (Live Class):")
    logger.info(f"  Precision (PPV): {metrics['precision']:.4f}")
    logger.info(f"  Recall (TPR):    {metrics['recall']:.4f}")
    logger.info(f"  Specificity:     {metrics['specificity']:.4f}")
    logger.info(f"  NPV:             {metrics['npv']:.4f}")
    
    logger.info(f"\nError Rates:")
    logger.info(f"  FAR (FPR):       {metrics['far']:.4f} ({metrics['far']*100:.2f}%)")
    logger.info(f"  FRR (FNR):       {metrics['frr']:.4f} ({metrics['frr']*100:.2f}%)")
    logger.info(f"  EER:             {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)")
    
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  True Negatives:  {metrics['tn']}")
    logger.info(f"  False Positives: {metrics['fp']}")
    logger.info(f"  False Negatives: {metrics['fn']}")
    logger.info(f"  True Positives:  {metrics['tp']}")
    
    logger.info(f"\nDataset Info:")
    logger.info(f"  Total Samples:   {metrics['total_samples']}")
    logger.info(f"  Live Samples:    {metrics['live_samples']}")
    logger.info(f"  Spoof Samples:   {metrics['spoof_samples']}")
    
    logger.info("\n" + "="*60)
    
def save_results(metrics, cm, y_true, y_pred, y_prob, dataset, indices, output_dir):
    """Save results to CSV and generate plots"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save overall metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / f"test_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"\nMetrics saved to: {metrics_path}")
    
    # Save per-image results
    results_data = []
    for idx in indices:
        results_data.append({
            'image_path': dataset.samples[idx],
            'image_name': dataset.image_names[idx],
            'subject_id': dataset.subjects[idx],
            'true_label': 'live' if dataset.labels[idx] == 1 else 'spoof',
            'predicted_label': 'live' if y_pred[np.where(indices == idx)[0][0]] == 1 else 'spoof',
            'probability_live': y_prob[np.where(indices == idx)[0][0]],
            'probability_spoof': 1 - y_prob[np.where(indices == idx)[0][0]],
            'correct': y_true[np.where(indices == idx)[0][0]] == y_pred[np.where(indices == idx)[0][0]]
        })
    
    results_df = pd.DataFrame(results_data)
    results_path = output_dir / f"per_image_results_{timestamp}.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Per-image results saved to: {results_path}")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        columns=['Predicted Spoof', 'Predicted Live'],
        index=['Actual Spoof', 'Actual Live']
    )
    cm_path = output_dir / f"confusion_matrix_{timestamp}.csv"
    cm_df.to_csv(cm_path)
    logger.info(f"Confusion matrix saved to: {cm_path}")
    
    # Generate and save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Spoof', 'Live'],
                yticklabels=['Spoof', 'Live'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plot_path = output_dir / f"confusion_matrix_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix plot saved to: {plot_path}")
    
    # Generate ROC curve
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        roc_path = output_dir / f"roc_curve_{timestamp}.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"ROC curve saved to: {roc_path}")
    except Exception as e:
        logger.warning(f"Could not generate ROC curve: {e}")
    
    # Per-subject analysis
    subject_results = results_df.groupby('subject_id').agg({
        'correct': ['sum', 'count', 'mean']
    }).round(4)
    subject_results.columns = ['correct_predictions', 'total_images', 'accuracy']
    subject_results = subject_results.sort_values('accuracy')
    subject_path = output_dir / f"per_subject_results_{timestamp}.csv"
    subject_results.to_csv(subject_path)
    logger.info(f"Per-subject results saved to: {subject_path}")
    
    # Summary report
    summary_path = output_dir / f"test_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FACE ANTI-SPOOFING TEST REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Checkpoint: checkpoints/best_model_run_8f3ihjcw.pth\n\n")
        
        f.write("OVERALL PERFORMANCE\n")
        f.write("-"*60 + "\n")
        f.write(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"AUC-ROC:         {metrics['auc']:.4f}\n")
        f.write(f"F1-Score:        {metrics['f1_score']:.4f}\n\n")
        
        f.write("DETECTION METRICS\n")
        f.write("-"*60 + "\n")
        f.write(f"Precision (PPV): {metrics['precision']:.4f}\n")
        f.write(f"Recall (TPR):    {metrics['recall']:.4f}\n")
        f.write(f"Specificity:     {metrics['specificity']:.4f}\n")
        f.write(f"NPV:             {metrics['npv']:.4f}\n\n")
        
        f.write("ERROR RATES\n")
        f.write("-"*60 + "\n")
        f.write(f"FAR (FPR):       {metrics['far']:.4f} ({metrics['far']*100:.2f}%)\n")
        f.write(f"FRR (FNR):       {metrics['frr']:.4f} ({metrics['frr']*100:.2f}%)\n")
        f.write(f"EER:             {metrics['eer']:.4f} ({metrics['eer']*100:.2f}%)\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-"*60 + "\n")
        f.write(f"True Negatives:  {metrics['tn']}\n")
        f.write(f"False Positives: {metrics['fp']}\n")
        f.write(f"False Negatives: {metrics['fn']}\n")
        f.write(f"True Positives:  {metrics['tp']}\n\n")
        
        f.write("DATASET INFO\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Samples:   {metrics['total_samples']}\n")
        f.write(f"Live Samples:    {metrics['live_samples']}\n")
        f.write(f"Spoof Samples:   {metrics['spoof_samples']}\n")
    
    logger.info(f"Summary report saved to: {summary_path}")
    
    return metrics_path, results_path
