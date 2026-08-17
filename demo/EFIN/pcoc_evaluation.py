import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class PCOCEvaluator:
    """
    PCOC指标评估器
    用于计算Uplift模型的PCOC和cal-N指标
    """
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def calculate_pcoc(self, 
                      pdf: pd.DataFrame, 
                      pred_col: str,
                      label_col: str,
                      group_col: str,
                      treatment_group: str, 
                      control_group: str) -> Dict:
        """
        计算整体PCOC指标
        
        在Uplift场景下:
        - Predicted Uplift = 预测的uplift列的平均值
        - Actual Uplift = Treatment组实际CVR - Control组实际CVR
        - PCOC = Predicted Uplift / Actual Uplift
        
        PCOC越接近1，说明预估越准确
        
        :param pdf: pandas DataFrame
        :param pred_col: 预测uplift列名
        :param label_col: 真实标签列名
        :param group_col: 分组列名（区分treatment和control）
        :param treatment_group: treatment组标识
        :param control_group: control组标识
        :return: 包含PCOC和详细统计的字典
        """
        # 筛选treatment和control组
        treatment_data = pdf[pdf[group_col] == treatment_group]
        control_data = pdf[pdf[group_col] == control_group]
        
        if len(treatment_data) == 0 or len(control_data) == 0:
            raise ValueError(f"数据为空: Treatment组({treatment_group})={len(treatment_data)}样本, "
                           f"Control组({control_group})={len(control_data)}样本")
        
        # 计算预测的平均uplift (Predicted)
        predicted_uplift = pdf[pred_col].mean()
        
        # 计算实际的CVR
        actual_treatment_cvr = treatment_data[label_col].mean()
        actual_control_cvr = control_data[label_col].mean()
        
        # 计算实际的uplift (Actual)
        actual_uplift = actual_treatment_cvr - actual_control_cvr
        
        # 计算PCOC
        if actual_uplift == 0:
            pcoc = float('inf') if predicted_uplift != 0 else 1.0
        else:
            pcoc = predicted_uplift / actual_uplift
        
        stats = {
            'pcoc': pcoc,
            'predicted_uplift': predicted_uplift,
            'actual_uplift': actual_uplift,
            'actual_treatment_cvr': actual_treatment_cvr,
            'actual_control_cvr': actual_control_cvr,
            'n_treatment': len(treatment_data),
            'n_control': len(control_data)
        }
        
        print(f"\n{'='*60}")
        print(f"PCOC指标计算结果")
        print(f"{'='*60}")
        print(f"  PCOC: {pcoc:.4f}")
        print(f"  PCOC偏离度: {abs(1 - pcoc):.6f}")
        print(f"{'='*60}\n")
        
        return stats
    
    def calculate_cal_n(self,
                       pdf: pd.DataFrame,
                       pred_col: str,
                       label_col: str,
                       group_col: str,
                       treatment_group: str,
                       control_group: str,
                       n_bins: int = 10) -> Tuple[float, Optional[pd.DataFrame]]:
        """
        计算Cal-N指标（分桶后的对称误差RMS）
        
        步骤:
        1. 按预测uplift分成n_bins个桶
        2. 对每个桶计算predicted_uplift和actual_uplift
        3. 对每个桶计算PCOC和symmetric_error
        4. Cal-N = RMS of symmetric_errors
        
        :param pdf: pandas DataFrame
        :param pred_col: 预测uplift列名
        :param label_col: 真实标签列名
        :param group_col: 分组列名
        :param treatment_group: treatment组标识
        :param control_group: control组标识
        :param n_bins: 分桶数量
        :return: (cal_n值, 分桶详细结果DataFrame)
        """
        # 按预测uplift分桶
        try:
            pdf['bin'] = pd.qcut(pdf[pred_col], q=n_bins, labels=False, duplicates='drop')
        except ValueError as e:
            print(f"[WARNING] 分桶失败: {e}, 返回cal_n=inf")
            return float('inf'), None
        
        binned_results = []
        symmetric_errors = []
        
        for bin_id in sorted(pdf['bin'].unique()):
            bin_data = pdf[pdf['bin'] == bin_id]
            
            treatment_bin = bin_data[bin_data[group_col] == treatment_group]
            control_bin = bin_data[bin_data[group_col] == control_group]
            
            if len(treatment_bin) == 0 or len(control_bin) == 0:
                continue
            
            # 计算该桶的predicted和actual uplift
            predicted_uplift = bin_data[pred_col].mean()
            
            actual_treatment_cvr = treatment_bin[label_col].mean()
            actual_control_cvr = control_bin[label_col].mean()
            actual_uplift = actual_treatment_cvr - actual_control_cvr
            
            # 计算PCOC
            if actual_uplift == 0:
                pcoc = float('inf') if predicted_uplift != 0 else 1.0
                symmetric_error = float('inf')
            else:
                pcoc = predicted_uplift / actual_uplift
                symmetric_error = self._calculate_symmetric_error(pcoc)
            
            binned_results.append({
                'bin': bin_id,
                'predicted_uplift': predicted_uplift,
                'actual_uplift': actual_uplift,
                'pcoc': pcoc,
                'symmetric_error': symmetric_error,
                'n_samples': len(bin_data),
                'n_treatment': len(treatment_bin),
                'n_control': len(control_bin)
            })
            
            if not np.isinf(symmetric_error):
                symmetric_errors.append(symmetric_error)
        
        # 计算Cal-N（对称误差的RMS）
        if len(symmetric_errors) == 0:
            cal_n = float('inf')
        else:
            cal_n = np.sqrt(np.mean(np.array(symmetric_errors) ** 2))
        
        binned_df = pd.DataFrame(binned_results) if binned_results else None
        
        print(f"\n{'='*60}")
        print(f"Cal-N指标计算结果")
        print(f"{'='*60}")
        print(f"  分桶数: {n_bins}")
        print(f"  有效桶数: {len(symmetric_errors)}")
        print(f"  Cal-N (对称误差RMS): {cal_n:.6f}")
        print(f"{'='*60}\n")
        
        return cal_n, binned_df
    
    def _calculate_symmetric_error(self, pcoc: float) -> float:
        """
        计算对称误差
        
        如果PCOC > 1: error = PCOC - 1
        如果PCOC <= 1: error = 1/PCOC - 1
        
        :param pcoc: PCOC值
        :return: 对称误差
        """
        if pcoc > 1:
            return pcoc - 1
        else:
            return 1.0 / pcoc - 1
