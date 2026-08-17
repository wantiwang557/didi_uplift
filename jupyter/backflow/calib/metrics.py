class CalibrationMetric(object):
    def __init__(self,  y_true_col, y_pred_col, bin_num_col):
        
        self.y_true_col = y_true_col
        self.y_pred_col = y_pred_col
        self.bin_num_col = bin_num_col
        
    
    def ece(self, pdf, norm='l1'):
        subset = pdf.copy()
        total_samples = subset[self.bin_num_col].sum()

        subset['ratio'] = subset[self.bin_num_col] / total_samples
        subset['diff']  = subset[self.y_pred_col] - subset[self.y_true_col]
        
        if norm == 'l1':
            subset['diff']  = subset['diff'].apply(lambda x:abs(x))
            subset['diff']  = subset['diff'] * subset['ratio']
            
            return subset['diff'].sum()
        
        elif norm == 'l2':
            subset['diff'] = subset['diff']**2
            subset['diff'] *= subset['ratio']
            
            return math.sqrt(subset['diff'].sum())
        
        elif norm == 'max':
            return subset['diff'].max()