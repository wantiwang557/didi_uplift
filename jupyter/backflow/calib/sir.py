import datetime,json
import pandas as pd

from pyspark.sql.functions import col,desc, udf
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import StringType
from pyspark.sql import DataFrame
from pyspark.sql import functions as func

class PiecewiseFunc:
    def __init__(self, ):
        self.x = list()
        self.y = list()

    def interpolate(self, value):
        self._check_param(self.x, self.y)
        # 插值方法的实现
        x, y = self.x, self.y
        output = None
        for idx, i in enumerate(x):
            if i > value:
                x_1 = x[idx - 1]
                x_2 = x[idx]
                y_1 = y[idx - 1]
                y_2 = y[idx]
                V = (value - x_1) / (x_2 - value)
                output = (y_1 + V * y_2) / (V + 1)
                break
        output = y[-1] if output is None else output
        return output

    def load_param(self, params):
        x, y = params["x"], params["y"]
        self._check_param(x, y)
        self.x, self.y = x, y

    def _check_param(self, a, b):
        assert len(a) == len(b), \
            "[Error] 参数长度不一致: len(a)={},len(b)={}".format(len(a), len(b))
        assert len(a), "[Error] 参数为空: len(a)={},len(b)={}".format(len(a), len(b))


class SIR_Model:
    # 定义校准模型
    def __init__(self, binning_method, BinSize, **kwargs):
        '''
        模型参数：points
        fit：
        transform：

        binning_method:
            BinEM: 固定样本数目的分桶方案，按ADX划分
            Sweep: 动态分桶方案，按adx划分
        '''
        self.binning_method = binning_method
        self.config = kwargs
        self.bin_size = BinSize
        self.f_ = PiecewiseFunc()

    def fit(self, X, Y):
        assert len(X) == len(Y), \
            "[Error] 输入数据长度不一致: len(X)={},len(y)={}".format(len(X), len(Y))
        # 不同分桶方式
        if self.binning_method == "BinEM":
            binned_data = self._SIR_Binning(T=zip(X, Y), N=self.bin_size)
        elif self.binning_method == "Sweep":
            pass
            # binned_data = self._Sweep_Binning(X, Y)
        f_points = self._SIR(binned_data)
        self.f_.x = [x for (x, y) in f_points]
        self.f_.y = [y for (x, y) in f_points]

    def transform(self, X):
        output = []
        # 根据model_type 选择模型
        # if self.model_type == "BinEM_ADX":
        for x in X:
            output.append(self.f_.interpolate(x))
        return output

    def load_param(self, param):
        # _check_param(param,self.model_type)
        self.f_ = PiecewiseFunc()
        self.f_.load_param(param)


    def _SIR_Binning(self, T, N=1):
        '''
        Smooted Isotonic Regression

        :param T: traing set, T = {(x_i,y_i)|i<=N}
        :param N: bin size
        :return: mapping function f(x)
        '''
        # Binning
        L = sorted(T, key=lambda x: x[0])
        x, y = [i[0] for i in L], [i[1] for i in L]
        BL = []
        K = len(L) // N  # 桶数
        for k in range(K):
            s_min, s_max = N * k, N * (k + 1)
            l_k = min(x[s_min:s_max])
            u_k = max(x[s_min:s_max])
            v_k = sum(y[s_min:s_max]) / N
            c_k = N
            BL.append([l_k, u_k, v_k, c_k])
        return BL

    def _SIR(self, BL):
        '''
        SIR 算法的第2、3阶段
        :param BL:  [ (l_1,u_1,v_1,c_1), (l_2,u_2,v_2,c_2),... ]  (min(pred),max(pred),mean(label),binSize)
        :return: 分段函数的端点
        '''
        # Pair Adjacent Violator for Bins
        IBL = []
        for i in range(len(BL)):
            l, u, v, c = BL[i][0], BL[i][1], BL[i][2], BL[i][3]
            if not len(IBL):
                IBL.append((l, u, v, c))
                continue
            l_t, u_t, v_t, c_t = IBL[-1]

            while v <= v_t:
                l = l_t
                v = (v * c + v_t * c_t) / (c + c_t)
                c = c + c_t
                IBL.pop(-1)

                # 防止IndexError
                if not len(IBL):
                    break

                l_t, u_t, v_t, c_t = IBL[-1]
            IBL.append((l, u, v, c))
        # l,u,v,c: 校准后分桶的 pred起点，pred终点，cali值，样本数目

        '''
        # Interpolation
        # 论文中方案
        # ML = []
        # for i in range(len(IBL)-1):
        #     m_l=(IBL[i][0]+IBL[i][1])/2
        #     m_r=(IBL[i+1][0]+IBL[i+1][1])/2
        # a = (IBL[i+1][2] - IBL[i][2])/(m_r-m_l)
        # b = IBL[i][2]-a*m_l
        # ML.append(m_l,m_r,a,b)
        '''

        # 存储校准分段函数的端点
        points = [((l + u) / 2, v) for l, u, v, c in IBL]
        points.insert(0, (0., 0.))
        points.append((1., 1.)) # 最后直接以到对角线点的插值结果
        return points


class SIRWrapper():
    def __init__(self, model_type, model_name='v100x_cpw',**kwargs):
        self.model_type = model_type
        self.save_table = ''
        self.config = kwargs
        
        
        self.param = None
        self.model = None
        
        
    def fit(self, X, y, binsize=1):
        '''
        对输入的数据排序，用于训练校准模型的参数
        :param train_data: 训练df格式训练样本集
        :return: 模型参数，训练集的校准结果
        '''
        model = SIR_Model(binning_method=self.model_type, BinSize=binsize)
        model.fit(X, y)
        
        param = {"x": model.f_.x, "y": model.f_.y}
        
        self.param = param
        self.model = model
        
        return self
    
    def transform(self, df:DataFrame, 
                  input_col='weighted_prob',
                  output_col='calib_prob'):
        @udf
        def __apply_parameters(x):
            return self.model.f_.interpolate(x)
        
        df = df.withColumn(output_col, __apply_parameters(func.col(input_col)))
        
        return df