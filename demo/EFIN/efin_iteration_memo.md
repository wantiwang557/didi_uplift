# WWT EFIN Iteration Memo

本文记录从当前 4-treatment S-learner 迭代到 EFIN 的改造点。当前目标不是一次性复刻论文全部模块，而是先保留现有 TFRecord/libsvm 数据链路、训练 loss、平均 AUUC 早停和 Hive 输出口径，逐步把模型内部结构替换为 EFIN 风格。

## 1. Self-attention 模块：保留 slot embedding

### 1.1 当前输入形式

当前 TFRecord 中的 `feature` 仍是 libsvm 风格字符串：

```text
index:value,index:value,index:value,...
```

解析后进入模型的 dense tensor 形状是：

```text
[batch, PADDING_SIZE * 2]
```

其中偶数位是全局 feature id，奇数位是 value：

```python
input_index = tf.cast(inputs[:, 0::2], tf.int64) - 1
input_value = tf.cast(inputs[:, 1::2], tf.float32)
```

所以只要 `.slots` 筛选后的 feature 顺序稳定，`input_index` 天然就是：

```text
[batch, slot_num]
```

也就是每条样本的一组 slot/token。

### 1.2 当前 S-learner 的问题

当前 `IndexEmbedding.call()` 里会直接 flatten：

```python
_embedding = tf.nn.embedding_lookup(self.embeddings, feature)
embedding = tf.reshape(_embedding, [-1, self.out_dense_dim])
return embedding
```

形状变化是：

```text
[batch, slot_num]
  -> embedding lookup
[batch, slot_num, embedding_dim]
  -> flatten
[batch, slot_num * embedding_dim]
```

这对普通 DNN、EUEN、基础 S-learner 可用，但不适合 EFIN 的 self-attention。因为 self-attention 需要显式知道：

```text
slot_1 和 slot_2 的关系
slot_i 在预测自然响应时应该关注哪些其他 slot
```

如果过早压平成一条整体向量，模型就只剩一个大样本向量，slot 粒度的特征交互载体会被抹平。

### 1.3 EFIN 第一版要改成什么

libsvm/TFRecord 格式不需要改。改动只发生在模型内部 embedding 层：

```text
input_index: [batch, slot_num]
  -> embedding lookup
slot_embeddings: [batch, slot_num, embedding_dim]
```

也就是说，一开始就直接做 slot embedding 是可行的，而且是最方便的做法。这里不是额外从原始数据里重新拆字段，而是利用当前 libsvm tensor 的固定排列：

```text
inputs[:, 0::2] = 每个 slot 的 feature id
inputs[:, 1::2] = 每个 slot 的 value
```

第一版建议新增一个方法或参数，让 `IndexEmbedding` 同时支持两种输出：

```python
def call(self, feature, flatten=True):
    slot_embeddings = tf.nn.embedding_lookup(self.embeddings, feature)
    if not flatten:
        return slot_embeddings
    return tf.reshape(slot_embeddings, [-1, self.out_dense_dim])
```

EFIN 用：

```python
slot_embeddings = self.embedding_layer(input_index, flatten=False)
```

基础 S-learner/EUEN 如果还需要复用旧结构，则继续用：

```python
flat_embeddings = self.embedding_layer(input_index, flatten=True)
```

### 1.4 Self-attention 的输入和输出

EFIN self-interaction 模块的输入是用户/上下文 slot embedding 序列：

```text
E_x = [e_1, e_2, ..., e_S]
shape = [batch, slot_num, embedding_dim]
```

如果当前使用 replace_order 特征方案：

```text
slot_num = 372
embedding_dim = 8
E_x shape = [batch, 372, 8]
```

self-attention 输出仍然是一组 slot embedding，而不是最终预测值：

```text
self_attended_x = SelfAttention(E_x)
shape = [batch, slot_num, attention_dim]
```

若 attention 输出维度仍为 8：

```text
self_attended_x shape = [batch, 372, 8]
```

含义是：每个 slot 的表示已经融合了其他 slot 的信息。

之后再接 control/natural response tower：

```text
self_attended_x
  -> flatten 或 pooling
  -> MLP
  -> control response y0
```

### 1.5 是否使用 input_value

当前 EUEN/S-learner 主要使用 `input_index`，基本没有使用 `input_value`。EFIN 第一版建议保持一致，先只用 index embedding，减少变量。

如果后续确认 libsvm 中的 value 有稳定业务含义，可以尝试：

```python
slot_embeddings = slot_embeddings * tf.expand_dims(input_value, axis=-1)
```

但这应作为单独实验，不要和 self-attention 首版混在一起。

### 1.6 结论

可以继续使用当前 libsvm/TFRecord 格式。

不需要改上游抽数，不需要把每个特征拆成独立 TFRecord 字段。

需要改的是模型内部第一层：

```text
不要 embedding 后立刻 flatten；
要先保留 [batch, slot_num, embedding_dim] 的 slot embedding 序列；
再把这组 slot token 送进 self-attention。
```

这一步是 EFIN 后续两个模块的基础：

```text
self-interaction: x 和 x 做交互，预测 natural response/control
treatment-aware interaction: treatment 和 x 做交互，预测 uplift tau_k
```

