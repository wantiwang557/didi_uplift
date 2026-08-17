import numpy as np

from mplfonts import use_font
from mplfonts.bin.cli import init

#init()
use_font('SimHei')

try:
    import importlib
    import matplotlib.pyplot as plt
except:
    pass

try:
    from . import metrics
    print("load metric func from causalml...")

except:
    from . import metrics
    print("load metric func from local...")

def plot_gaps(pdf, bucket_col, plot_cols, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for col in plot_cols:
        plt.plot(pdf[bucket_col], pdf[col], label=col)
        
    if 'gname' in kwargs:
        plt.title('group_name={}'.format(kwargs['gname']))
    
    plt.grid(linewidth=2)
    plt.legend()

    if 'title' in kwargs:
        plt.title(kwargs['title'])
    
    plt.show()

def plot_gaps_save(pdf, bucket_col, plot_cols, **kwargs):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for col in plot_cols:
        
        plt.plot(pdf[bucket_col], pdf[col], label=col)
        
    if 'gname' in kwargs:
        plt.title('group_name={}'.format(kwargs['gname']))
    
    plt.grid(linewidth=2)
    plt.legend()

    if 'title' in kwargs:
        plt.title(kwargs['title'])
    
    plt.savefig('/nfs/volume-385-2/lite-fea-workspace/yyl_workspace/models/plot.png')

# ite_names: 与treatments相对应的增量治疗效应（ITE）名称
def draw_auuc(prediction_pdf, treatments, ite_names, label_col='pv_label',
             curve_type='qini',
              real_treatment='real_treatment',
              control_group_name='control_group',
             show_curve=False):
    
    metric_results = [] # 这将存储每种处理的指标结果

    for treatment, ite_name in zip(treatments, ite_names): # 循环遍历干预和ITE名称
        #if treatment == control_group_name:
        #    continue

        # 从prediction_pdf中选择实际干预是当前干预或控制组的行。
        subset = prediction_pdf[prediction_pdf[real_treatment].isin([treatment, control_group_name])]
        #  为子集添加一个新列exp_group，其中实验组为1，控制组为0。
        subset['exp_group'] = subset[real_treatment].apply(lambda x:1 if x!=control_group_name else 0)

        # 为子集添加一个新列label，其中非零结果被视为1，零结果被视为0。
        subset['label'] = subset[label_col]
        #subset['label'] = subset[label_col].apply(lambda x:1 if x!=0 else 0)
        
        # 计算并绘制AUUC:
        if curve_type == 'auuc':
            lift = metrics.get_cumlift(subset[['exp_group','label', ite_name]], # 从subset数据框中选择特定的列。这里使用了列表切片来选取'exp_group'、'label'和ite_name这三列
                               outcome_col='label', 
                               treatment_col='exp_group',
                               random_seed=4) # 返回一个名为lift的对象，它可能是一个DataFrame或Series，包含提升曲线的值

            gain = lift.mul(lift.index.values, axis=0)
            gain = gain.div(np.abs(gain.iloc[-1, :]), axis=1)
            
            if show_curve:
                gain.plot()

            res = gain.sum() / gain.shape[0]
            print(res)
            print("**"*20)

            metric_results.append(res)
        
        elif curve_type == 'qini':
            lift = metrics.qini_score(subset[['exp_group','label', ite_name]], 
                                      outcome_col='label', 
                                      treatment_col='exp_group', 
                                      
                            ) + 0.5
            print(lift)
            metric_results.append(lift)

            if show_curve:
                metrics.plot(subset[['exp_group','label', ite_name]], 
                         kind='qini', tmle=False, 
                         outcome_col='label', treatment_col='exp_group',
                         )
    return metric_results
