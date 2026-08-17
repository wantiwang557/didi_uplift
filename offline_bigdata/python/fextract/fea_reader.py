# from datetime import datetime, timedelta
# import tensorflow as tf


# assert tf.__version__.startswith("2."), "Tensorflow版本过低"
# if int(tf.__version__.split(".")[1]) >= 4:
#     import tensorflow_io as tfio
    


# class LiteFeTFRecords(object):

#     def __init__(self,
#                  hdfs_pattern,
#                  start_dt,
#                  end_dt,
#                  label_size=-1,
#                  token_size=-1,
#                  dt_fmt="%Y%m%d"):

#         _hdfs_pattern = hdfs_pattern.strip()
#         if not _hdfs_pattern.startswith("hdfs://"):
#             _hdfs_pattern = "hdfs://" + _hdfs_pattern

#         self.hdfs_pattern = _hdfs_pattern
#         self.label_size = label_size
#         self.token_size = token_size

#         self.start_dt = datetime.strptime(start_dt.strip(), dt_fmt)
#         self.end_dt = datetime.strptime(end_dt.strip(), dt_fmt)

#     def _get_hdfs_path_list(self):

#         tfr_path_list = []

#         days = (self.end_dt - self.start_dt).days

#         for i in range(days + 1):

#             _pattern = (
#                     self.start_dt + timedelta(days=i)
#             ).strftime(
#                 self.hdfs_pattern
#             )

#             tfr_path = tf.io.gfile.glob(_pattern)
#             tfr_path_list.extend(tfr_path)

#         return tfr_path_list

#     def get_dataset(self,
#                     epoch=1,
#                     batch_size=1,
#                     parallel=1,
#                     shuffle_size=0,
#                     drop_remainder=True,
#                     return_token=False,
#                     token_dtype=str):

#         feature_dict= {
#             "label":   tf.io.FixedLenFeature([1], tf.string),
#             "feature": tf.io.FixedLenFeature([1], tf.string),
#             "token":   tf.io.FixedLenFeature([1], tf.string),
#         }

#         _token2str, _token_dtype = {
#             int: (False, tf.int64),
#             float: (False, tf.float32),
#             str: (True, tf.string)
#         }[token_dtype]

#         # TFRecords 解包函数
#         def _parse_single_example(example):

#             feature_map = tf.io.parse_single_example(example, feature_dict)

#             # label
#             _labels = tf.strings.split(feature_map["label"], ",").values
#             _labels = tf.strings.to_number(_labels, tf.float32)
#             label = tf.reshape(_labels, [self.label_size])

#             # features
#             libsvm_kv = tf.strings.split(feature_map["feature"], ",").values
#             libsvm_kv = tf.strings.split(libsvm_kv, ":").values
#             feature = tf.strings.to_number(libsvm_kv, tf.float32)

#             # token
#             if return_token:
#                 _token = tf.strings.split(feature_map["token"], ",").values

#                 if not _token2str:
#                     _token = tf.strings.to_number(_token, _token_dtype)

#                 token = tf.reshape(_token, [self.token_size])

#                 res = (feature, label, token)

#             else:
#                 res = (feature, label)

#             # keras dataset format :
#             # (inputs, targets, sample_weights) or (inputs, targets)
#             return res

#         # 获取文件列表
#         tfr_path_list = self._get_hdfs_path_list()

#         dataset = tf.data.TFRecordDataset(
#             filenames=tfr_path_list,
#             num_parallel_reads=parallel
#         ).map(
#             map_func=_parse_single_example,
#             num_parallel_calls=parallel
#         ).batch(
#             batch_size=batch_size,
#             drop_remainder=drop_remainder
#         ).repeat(
#             epoch
#         ).prefetch(
#             buffer_size=tf.data.experimental.AUTOTUNE
#         )

#         if shuffle_size > 0:
#             dataset = dataset.shuffle(shuffle_size)

#         return dataset


# class LiteFeLibsvmText(object):

#     def __init__(self,
#                  local_pattern,
#                  label_size=-1,
#                  token_size=-1):

#         self.local_pattern = local_pattern
#         self.label_size = label_size
#         self.token_size = token_size

#     def get_dataset(self,
#                     epoch=1,
#                     batch_size=1,
#                     parallel=1,
#                     shuffle_size=0,
#                     drop_remainder=False,
#                     return_token=False,
#                     token_dtype=str):

#         _token2str, _token_dtype = {
#             int: (False, tf.int64),
#             float: (False, tf.float32),
#             str: (True, tf.string)
#         }[token_dtype]

#         # libsvm text 解包函数
#         def _decode_single_libsvm(line):

#             # label_feature_token & token split
#             lab_fea_token = tf.strings.split(line, "#")

#             # label_feature
#             libsvm_str = tf.strings.strip(lab_fea_token[0])
#             libsvm_str = tf.strings.split(libsvm_str, " ")

#             # label
#             _label = tf.strings.to_number(libsvm_str[0:1], tf.float32)
#             label = tf.reshape(_label, [self.label_size])

#             # feature
#             _feature = tf.strings.split(libsvm_str[1:], ":").values
#             # _feature = tf.reshape(_feature, [PADDING_SIZE * 2])
#             feature = tf.strings.to_number(_feature, tf.float32)

#             # token
#             if return_token:
#                 _token = tf.strings.strip(lab_fea_token[1])
#                 _token = tf.strings.split(_token, "\t")

#                 if not _token2str:
#                     _token = tf.strings.to_number(_token, _token_dtype)

#                 token = tf.reshape(_token, [self.token_size])

#                 res = (feature, label, token)

#             else:

#                 res = (feature, label)

#             # keras dataset format :
#             # (inputs, targets, sample_weights) or (inputs, targets)
#             return res

#         # 获取文件列表
#         libsvm_path_list = tf.io.gfile.glob(self.local_pattern)

#         dataset = tf.data.TextLineDataset(
#             filenames=libsvm_path_list,
#             num_parallel_reads=parallel
#         ).map(
#             map_func=_decode_single_libsvm,
#             num_parallel_calls=parallel
#         ).batch(
#             batch_size=batch_size,
#             drop_remainder=drop_remainder
#         ).repeat(
#             epoch
#         ).prefetch(
#             buffer_size=tf.data.experimental.AUTOTUNE
#         )

#         if shuffle_size > 0:
#             dataset = dataset.shuffle(shuffle_size)

#         return dataset


# if __name__ == "__main__":

#     def test_tfrecords_reader():
#         train_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/yjx/v2/train/2025/10/22/part-*"
#         train_start_date = "20250624"
#         train_end_date = "20250909"

#         reader = LiteFeTFRecords(
#             hdfs_pattern=train_tfr_path,
#             start_dt=train_start_date,
#             end_dt=train_end_date,
#             label_size=1,
#             token_size=4
#         )

#         return reader


#     def test_libsvm_text_reader():
#         train_libsvm_path = "./my_workspace/offline_bigdata/script/*.libsvm"

#         reader = LiteFeLibsvmText(local_pattern=train_libsvm_path)

#         return reader


#     reader_object = test_tfrecords_reader
#     # reader_object = test_libsvm_text_reader

#     reader = reader_object()
#     dataset = reader.get_dataset(
#         epoch=1,               # 迭代1轮
#         batch_size=128,        # batch size
#         parallel=4,            # 读取并行度
#         shuffle_size=-1,       # 小于等于0表示不用 shuffle
#         drop_remainder=False,  # 是否丢弃不足batch size的最后一个batch
#         token_dtype=str        # token字段的数据类型 支持 int / float / str
#     )

#     for label, token, feature in dataset:
#         print("label - [dtype: {}, shape: {}]\n{}\n".format(
#             label.dtype.name, label.shape, label))

#         print("token - [dtype: {}, shape: {}]\n{}\n".format(
#             token.dtype.name, token.shape, token))

#         print("feature - [dtype: {}, shape: {}]\n{}\n".format(
#             feature.dtype.name, feature.shape, feature))
#         break


from datetime import datetime, timedelta
import tensorflow as tf

assert tf.__version__.startswith("2."), "Tensorflow版本过低"
if int(tf.__version__.split(".")[1]) >= 4:
    import tensorflow_io as tfio


class LiteFeTFRecords(object):

    def __init__(self,
                 hdfs_pattern,
                 start_dt,
                 end_dt,
                 label_size=-1,
                 token_size=-1,
                 dt_fmt="%Y%m%d"):

        _hdfs_pattern = hdfs_pattern.strip()
        if not _hdfs_pattern.startswith("hdfs://"):
            _hdfs_pattern = "hdfs://" + _hdfs_pattern

        self.hdfs_pattern = _hdfs_pattern
        self.label_size = label_size
        self.token_size = token_size

        self.start_dt = datetime.strptime(start_dt.strip(), dt_fmt)
        self.end_dt = datetime.strptime(end_dt.strip(), dt_fmt)

    def _get_hdfs_path_list(self):

        tfr_path_list = []

        days = (self.end_dt - self.start_dt).days

        for i in range(days + 1):
            _pattern = (
                    self.start_dt + timedelta(days=i)
            ).strftime(
                self.hdfs_pattern
            )

            tfr_path = tf.io.gfile.glob(_pattern)
            tfr_path_list.extend(tfr_path)

        return tfr_path_list

    def get_dataset(self,
                    epoch=1,
                    batch_size=1,
                    parallel=1,
                    shuffle_size=0,
                    drop_remainder=True,
                    return_token=False,
                    token_dtype=str):

        feature_dict = {
            "label": tf.io.FixedLenFeature([1], tf.string),
            "feature": tf.io.FixedLenFeature([1], tf.string),
            "token": tf.io.FixedLenFeature([1], tf.string),
        }

        _token2str, _token_dtype = {
            int: (False, tf.int64),
            float: (False, tf.float32),
            str: (True, tf.string)
        }[token_dtype]

        # TFRecords 解包函数
        def _parse_single_example(example):

            feature_map = tf.io.parse_single_example(example, feature_dict)

            # 调试：打印原始数据
            tf.print("=" * 80)
            tf.print("原始 label:", feature_map["label"])
            tf.print("原始 feature:", feature_map["feature"])
            tf.print("原始 token:", feature_map["token"])

            # label
            _labels = tf.strings.split(feature_map["label"], ",").values
            tf.print("切分后的 labels:", _labels)
            _labels = tf.strings.to_number(_labels, tf.float32)
            label = tf.reshape(_labels, [self.label_size])
            tf.print("转换后的 label:", label)

            # features
            libsvm_kv = tf.strings.split(feature_map["feature"], ",").values
            tf.print("按逗号切分后的 feature (前10个):", libsvm_kv[:10])
            libsvm_kv = tf.strings.split(libsvm_kv, ":").values
            tf.print("按冒号切分后的 libsvm_kv (前20个):", libsvm_kv[:20])
            tf.print("libsvm_kv 总数量:", tf.shape(libsvm_kv))
            # 检查是否有空字符串
            empty_mask = tf.equal(tf.strings.length(libsvm_kv), 0)
            tf.print("空字符串数量:", tf.reduce_sum(tf.cast(empty_mask, tf.int32)))
            if tf.reduce_any(empty_mask):
                tf.print("警告：发现空字符串！位置:", tf.where(empty_mask))
            feature = tf.strings.to_number(libsvm_kv, tf.float32)

            # token
            if return_token:
                _token = tf.strings.split(feature_map["token"], ",").values

                if not _token2str:
                    _token = tf.strings.to_number(_token, _token_dtype)

                token = tf.reshape(_token, [self.token_size])

                res = (feature, label, token)

            else:
                res = (feature, label)

            # keras dataset format :
            # (inputs, targets, sample_weights) or (inputs, targets)
            return res

        # 获取文件列表
        tfr_path_list = self._get_hdfs_path_list()

        dataset = tf.data.TFRecordDataset(
            filenames=tfr_path_list,
            num_parallel_reads=parallel
        ).map(
            map_func=_parse_single_example,
            num_parallel_calls=parallel
        ).apply(
            tf.data.experimental.ignore_errors()  # 跳过无效数据
        ).batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ).repeat(
            epoch
        ).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        if shuffle_size > 0:
            dataset = dataset.shuffle(shuffle_size)

        return dataset


class LiteFeLibsvmText(object):

    def __init__(self,
                 local_pattern,
                 label_size=-1,
                 token_size=-1):

        self.local_pattern = local_pattern
        self.label_size = label_size
        self.token_size = token_size

    def get_dataset(self,
                    epoch=1,
                    batch_size=1,
                    parallel=1,
                    shuffle_size=0,
                    drop_remainder=False,
                    return_token=False,
                    token_dtype=str):

        _token2str, _token_dtype = {
            int: (False, tf.int64),
            float: (False, tf.float32),
            str: (True, tf.string)
        }[token_dtype]

        # libsvm text 解包函数
        def _decode_single_libsvm(line):

            # label_feature_token & token split
            lab_fea_token = tf.strings.split(line, "#")

            # label_feature
            libsvm_str = tf.strings.strip(lab_fea_token[0])
            libsvm_str = tf.strings.split(libsvm_str, " ")

            # label
            _label = tf.strings.to_number(libsvm_str[0:1], tf.float32)
            label = tf.reshape(_label, [self.label_size])

            # feature
            _feature = tf.strings.split(libsvm_str[1:], ":").values
            # _feature = tf.reshape(_feature, [PADDING_SIZE * 2])
            feature = tf.strings.to_number(_feature, tf.float32)

            # token
            if return_token:
                _token = tf.strings.strip(lab_fea_token[1])
                _token = tf.strings.split(_token, "\t")

                if not _token2str:
                    _token = tf.strings.to_number(_token, _token_dtype)

                token = tf.reshape(_token, [self.token_size])

                res = (feature, label, token)

            else:

                res = (feature, label)

            # keras dataset format :
            # (inputs, targets, sample_weights) or (inputs, targets)
            return res

        # 获取文件列表
        libsvm_path_list = tf.io.gfile.glob(self.local_pattern)

        dataset = tf.data.TextLineDataset(
            filenames=libsvm_path_list,
            num_parallel_reads=parallel
        ).map(
            map_func=_decode_single_libsvm,
            num_parallel_calls=parallel
        ).batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ).repeat(
            epoch
        ).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )

        if shuffle_size > 0:
            dataset = dataset.shuffle(shuffle_size)

        return dataset


if __name__ == "__main__":

    def test_tfrecords_reader():
        # train_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/zyr/gemn_cost_v2/test/2025/11/04/part-*"
        # train_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/yjx/v2/test/2025/11/07/part-*"
        # train_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/yjx/v0/test/2025/10/20/part-*"
        train_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/yjx/v3new/test/2026/01/16/part-*"
        # train_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/zyr/gemn_cost_v1/train/2025/11/03/part-*"
        # train_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/zyr/cost_v4/test/2025/10/07/part-*"
        # train_tfr_path = "hdfs://difed/user/prod_growstrategy/tmp/zyr/v0/train/2025/09/05/part-*"
        
        # 下方date未生效，不用修改
        train_start_date = "20251103"
        train_end_date = "20251103"

        reader = LiteFeTFRecords(
            hdfs_pattern=train_tfr_path,
            start_dt=train_start_date,
            end_dt=train_end_date,
            label_size=1,
            token_size=7
        )

        return reader


    reader_object = test_tfrecords_reader
    # reader_object = test_libsvm_text_reader

    reader = reader_object()
    dataset = reader.get_dataset(
        epoch=1,  # 调试模式：只迭代1轮
        batch_size=1,  # 调试模式：batch size 设为 1
        parallel=1,  # 调试模式：并行度设为1，避免输出混乱
        shuffle_size=-1,  # 小于等于0表示不用 shuffle
        drop_remainder=False,  # 是否丢弃不足batch size的最后一个batch
        token_dtype=str,  # token字段的数据类型 支持 int / float / str
        return_token=True
    )

    print("\n" + "="*80)
    print("开始调试模式：只读取前3条数据")
    print("="*80 + "\n")

    cnt = 0
    try:
        for feature, label, token in dataset.take(3): # 只取前3条数据进行调试
            print(f"\n>>> batch {cnt}: {'='*60}")
            cnt += 1
            print("label - [dtype: {}, shape: {}]\n{}\n".format(
                label.dtype.name, label.shape, label))

            print("token - [dtype: {}, shape: {}]\n{}\n".format(
                token.dtype.name, token.shape, token))

            print("feature - [dtype: {}, shape: {}]\n{}\n".format(
                feature.dtype.name, feature.shape, feature))
            
        print("\n" + "="*80)
        print(f"成功读取 {cnt} 条数据！")
        print("="*80)
    except Exception as e:
        print("\n" + "="*80)
        print(f"在第 {cnt+1} 条数据处发生错误！")
        print("="*80)
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()