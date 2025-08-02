### SUGGESTION 1: ADDED NEW FILE FOR UNCERTAINTY METRICS ###
# This file contains functions to calculate ECE, Brier Score, OOD AUC, 
# and to plot reliability diagrams, as requested for a more thorough evaluation.

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def calculate_brier_score(probs, labels):
    """
    Calculates the multi-class Brier score.
    
    Args:
        probs (np.array): Predicted probabilities, shape (n_samples, n_classes).
        labels (np.array): True labels, shape (n_samples,).
        
    Returns:
        float: The Brier score.
    """
    if probs.ndim != 2 or labels.ndim != 1:
        print(f"Warning: Invalid shapes for Brier score. Probs: {probs.shape}, Labels: {labels.shape}")
        return float('nan')
        
    labels_one_hot = np.eye(probs.shape[1])[labels]
    return np.mean(np.sum((probs - labels_one_hot)**2, axis=1))


def calculate_ece(probs, labels, n_bins=15):
    """
    计算期望校准误差 (ECE)
    
    Args:
        probs (np.array): 预测概率，形状 (n_samples, n_classes)
        labels (np.array): 真实标签，形状 (n_samples,)
        n_bins (int): 用于校准计算的分箱数
        
    Returns:
        float: ECE分数
        dict: 用于绘制可靠性图的数据
    """
    try:
        # 确保输入是numpy数组
        probs = np.asarray(probs)
        labels = np.asarray(labels)
        
        # 检查输入维度
        if len(probs) == 0 or len(labels) == 0:
            print(f"ECE计算错误: 空输入 probs={probs.shape}, labels={labels.shape}")
            return float('nan'), {'bin_accuracies': [], 'bin_confidences': [], 'bin_counts': []}
            
        # 确保labels是一维的
        labels = labels.flatten()
        
        # 如果probs是一维的，将其转换为二维
        if len(probs.shape) == 1:
            # 假设这是二分类问题的正类概率
            probs = np.vstack([1-probs, probs]).T
        
        # 确保样本数量匹配
        if probs.shape[0] != labels.shape[0]:
            print(f"ECE计算错误: 维度不匹配 probs={probs.shape}, labels={labels.shape}")
            return float('nan'), {'bin_accuracies': [], 'bin_confidences': [], 'bin_counts': []}
        
        # 获取预测类别和置信度
        predictions = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        # 计算ECE
        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        diagram_data = {'bin_accuracies': [], 'bin_confidences': [], 'bin_counts': []}
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_count = np.sum(in_bin)
            
            if bin_count > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += (bin_count / len(confidences)) * np.abs(avg_confidence_in_bin - accuracy_in_bin)
                
                diagram_data['bin_accuracies'].append(accuracy_in_bin)
                diagram_data['bin_confidences'].append(avg_confidence_in_bin)
                diagram_data['bin_counts'].append(bin_count)
            else:
                diagram_data['bin_accuracies'].append(0)
                diagram_data['bin_confidences'].append(0)
                diagram_data['bin_counts'].append(0)
                
        return ece, diagram_data
    except Exception as e:
        print(f"ECE计算错误: {str(e)}")
        return float('nan'), {'bin_accuracies': [], 'bin_confidences': [], 'bin_counts': []}

def plot_reliability_diagram(diagram_data, filename='reliability_diagram.png'):
    """
    Saves a reliability diagram plot.
    """
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    # Ensure data is valid before plotting
    if diagram_data['bin_confidences'] and diagram_data['bin_accuracies']:
        plt.bar(
            diagram_data['bin_confidences'], 
            diagram_data['bin_accuracies'], 
            width=0.05, 
            alpha=0.7, 
            edgecolor='black',
            label='Model Output'
        )

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def calculate_ood_auc(id_uncertainties, ood_uncertainties):
    """
    Calculates the Area Under the ROC Curve (AUC) for OOD detection.
    
    Args:
        id_uncertainties (np.array): Uncertainty scores for in-distribution data.
        ood_uncertainties (np.array): Uncertainty scores for out-of-distribution data.
        
    Returns:
        float: The AUC-ROC score.
    """
    labels = np.concatenate([np.zeros(len(id_uncertainties)), np.ones(len(ood_uncertainties))])
    scores = np.concatenate([id_uncertainties, ood_uncertainties])
    
    return roc_auc_score(labels, scores)

def plot_uncertainty_distribution(id_epistemic, id_aleatoric, ood_epistemic=None, ood_aleatoric=None, save_path=None):
    """绘制不确定性分布对比图"""
    plt.figure(figsize=(12, 5))
    
    # 绘制认知不确定性分布
    plt.subplot(1, 2, 1)
    plt.hist(id_epistemic, bins=30, alpha=0.7, label='ID', density=True)
    if ood_epistemic is not None:
        plt.hist(ood_epistemic, bins=30, alpha=0.7, label='OOD', density=True)
    plt.title('Epistemic Uncertainty Distribution')
    plt.legend()
    
    # 绘制偶然不确定性分布
    plt.subplot(1, 2, 2)
    plt.hist(id_aleatoric, bins=30, alpha=0.7, label='ID', density=True)
    if ood_aleatoric is not None:
        plt.hist(ood_aleatoric, bins=30, alpha=0.7, label='OOD', density=True)
    plt.title('Aleatoric Uncertainty Distribution')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
    return save_path

def plot_uncertainty_vs_accuracy(probs, labels, epistemic_u, aleatoric_u, num_bins=10, save_path=None):
    """绘制不确定性与准确率关系图"""
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels)
    
    # 按认知不确定性排序并分箱
    sorted_indices = np.argsort(epistemic_u)
    bin_size = len(sorted_indices) // num_bins
    
    bin_accuracies = []
    bin_epistemic = []
    bin_aleatoric = []
    
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = (i+1) * bin_size if i < num_bins-1 else len(sorted_indices)
        bin_indices = sorted_indices[start_idx:end_idx]
        
        bin_accuracies.append(np.mean(correct[bin_indices]))
        bin_epistemic.append(np.mean(epistemic_u[bin_indices]))
        bin_aleatoric.append(np.mean(aleatoric_u[bin_indices]))
    
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(bin_epistemic, bin_accuracies, 'o-', label='Accuracy vs Epistemic')
    plt.xlabel('Epistemic Uncertainty')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(bin_aleatoric, bin_accuracies, 'o-', label='Accuracy vs Aleatoric')
    plt.xlabel('Aleatoric Uncertainty')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    
    return save_path

def calculate_aleatoric_consistency(aleatoric_uncertainties, correct_predictions):
    """
    计算偶然不确定性一致性 (使用Mann-Whitney U检验替代t-test)
    检查正确和错误预测的偶然不确定性是否有显著差异
    """
    try:
        # 确保输入是numpy数组
        aleatoric_uncertainties = np.asarray(aleatoric_uncertainties)
        correct_predictions = np.asarray(correct_predictions).astype(bool)
        
        # 确保维度匹配
        if len(aleatoric_uncertainties) != len(correct_predictions):
            print(f"维度不匹配: aleatoric shape {aleatoric_uncertainties.shape}, correct shape {correct_predictions.shape}")
            return float('nan')
        
        # 分离正确和错误预测的不确定性
        correct_uncertainties = aleatoric_uncertainties[correct_predictions]
        incorrect_uncertainties = aleatoric_uncertainties[~correct_predictions]
        
        # 确保每组至少有1个样本
        if len(correct_uncertainties) < 1 or len(incorrect_uncertainties) < 1:
            print(f"样本数不足: 正确样本={len(correct_uncertainties)}, 错误样本={len(incorrect_uncertainties)}")
            return float('nan')
            
        # 使用Mann-Whitney U检验替代t检验
        from scipy.stats import mannwhitneyu
        try:
            # 计算Mann-Whitney U统计量和p值
            u_stat, p_value = mannwhitneyu(correct_uncertainties, incorrect_uncertainties, alternative='two-sided')
            # 一致性指标：p值越小，差异越显著
            consistency = 1 - min(p_value, 1.0)
            return consistency
        except ValueError as ve:
            # 如果Mann-Whitney失败，尝试使用中位数差异
            correct_median = np.median(correct_uncertainties)
            incorrect_median = np.median(incorrect_uncertainties)
            # 计算标准化的中位数差异
            median_diff = abs(correct_median - incorrect_median) / (np.std(aleatoric_uncertainties) + 1e-8)
            # 将差异映射到0-1范围
            consistency = min(1.0, median_diff)
            return consistency
            
    except Exception as e:
        print(f"计算偶然不确定性一致性时出错: {str(e)}")
        return float('nan')
