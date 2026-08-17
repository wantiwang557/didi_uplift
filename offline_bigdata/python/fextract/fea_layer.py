import tensorflow as tf


class IndexEmbedding(tf.keras.layers.Embedding):

    def __init__(self,
                 dense_fea_dim, # 稠密特征的维度，即输入的特征数量，比如说有471个稠密特征
                 embedding_dim, # 嵌入维度，指定每个稠密特征的嵌入空间维度，设为8的意思就是每个稠密特征会映射到一个8维的嵌入空间
                 sparse_fea_dim,
                 embeddings_initializer="uniform",
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):

        super(IndexEmbedding, self).__init__(
            input_dim=dense_fea_dim, # 输入维度我设成了我的特征数量
            output_dim=embedding_dim, # 输出维度我设为了8
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            input_length=input_length,
            **kwargs
        )

        self.sparse_fea_dim = sparse_fea_dim
        self.out_dense_dim = sparse_fea_dim * embedding_dim

    def build(self, input_shape=None):
        name = "index_embedding_{}_{}".format(
            self.input_dim, self.output_dim)

        self.embeddings = self.add_weight( # 为层添加权重
            shape=(self.input_dim, self.output_dim), # 权重矩阵的形状，通常是 (input_dim, output_dim) 即471,8
            initializer=self.embeddings_initializer, # 权重初始化器，用于初始化权重矩阵的值
            regularizer=self.embeddings_regularizer, # 权重正则化器，用于施加正则化约束
            constraint=self.embeddings_constraint, # 权重约束，用于施加约束条件（如非负性约束）
            experimental_autocast=False,
            name=name # 权重矩阵的名称
        )

        self.built = True

    def call(self, feature, **kwargs):
        # 其中self.embeddings维度为（470,8），
        _embedding = tf.nn.embedding_lookup(self.embeddings, feature) # feature: 这是一个索引或者一组索引，用于指定要查找的嵌入向量
        embedding = tf.reshape(_embedding, [-1, self.out_dense_dim]) # 结果会直接返回 [batch_size, fea_num * embedding_size] 的向量

        return embedding


class MaskEmbedding(tf.keras.layers.Embedding):

    def __init__(self,
                 dense_fea_dim,
                 embedding_dim,
                 sparse_fea_dim,
                 embeddings_initializer="uniform",
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):

        super(MaskEmbedding, self).__init__(
            input_dim=dense_fea_dim,
            output_dim=embedding_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            input_length=input_length,
            **kwargs
        )

        self.sparse_fea_dim = sparse_fea_dim
        self.out_dense_dim = sparse_fea_dim * embedding_dim

    def build(self, input_shape=None):
        name = "index_embedding_{}_{}".format(
            self.input_dim, self.output_dim)

        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            experimental_autocast=False,
            name=name
        )

        self.built = True

    def call(self, feature, mask, **kwargs):
        emb = tf.nn.embedding_lookup(self.embeddings, feature)

        if (len(mask.shape) == 2) and (mask.shape[-1] == self.sparse_fea_dim):
            mask = tf.reshape(mask, [-1, self.sparse_fea_dim, 1])

        _mask_emb = tf.multiply(emb, mask)
        mask_emb = tf.reshape(_mask_emb, [-1, self.out_dense_dim])

        return mask_emb
