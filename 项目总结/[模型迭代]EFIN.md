<!-- 请注意：文档内的附件与图片将在 2026/10/27 14:33:40  失效，请您在有效期内及时下载并保存至本地
Note: Attachments and images in the document will expire at 2026/10/27 14:33:40. Please download and save them locally within the valid period.-->

# [模型迭代]EFIN

### 背景

当前对于treatment的信息没有显示使用，模型对于treatment之间的差异学习不够，尝试采用EFIN模型结构显示表征干预信息；利用attention机制显示学习treatment特征和非treatment特征交互作用，找出对treatment敏感的特征.。

### conclusion

- 模型结构：在 shared-bottom 用户表征基础上，一路建模`control`分支，一路建模`uplift分支`，引入     **treatment 显式表征和 treatment-aware attention，**concat （用户全局特征，treatment表征，对干预敏感的用户特征） 输入 uplift tower 直接预测 tau；一路结合用户全局特征和干预表征建模`cost 分支`。

- 在同特征、同验证窗口下，EFIN mean AUUC 达到 0.8505，vs EUEN (0.8427)，**AUUC** **uplift +0.77pp**.

### 实验过程

####  s-learner

> /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/EFIN/s_learner_4t.py

1. **模型结构**

- 目前采用的深度模型EUEN是基于T-learner思想，采用多head结构，每个 treatment 有自己的一套 uplift head / cost head。EFIN 基于S-learner 思想，多treatment用一套共享结构，通过 treatment 条件来预测label。因此初步把模型切换成s-learner结构。

```Python
用户/上下文特征 
  -> Embedding flatten -> shared_bottom
  -> user_repr

treatment_id 
  -> treatment_embedding

concat(user_repr, treatment_embedding)
  -> 共享 outcome_tower: 
  -> y(0), y(1), y(2), y(3), y(4)  #s-learner建模label：完单量

concat(user_repr, treatment_embedding)
  -> 共享 cost_tower: 
  -> cost(1), cost(2), cost(3), cost(4)

uplift_k = y(k) - y(0)
loss = label MSE + cost MSE
```

2. **模型效果**

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786675018614/ZUKrnlXtlH1eo5-JI_Eze/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=398daaeb4610ec885aea7d6ee0f2381eb89d408317669c76448c0cbe6b30ffcf)

#### EFIN 

> /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/EFIN/EFIN_4t.py

##### **1. 模型结构**

- treatment表征：treatment id，券总数量，首张券面额，留存券面额，券包总金额。

> 比如6张5元对应：券总数量6，首张5，留存5，总金额30。

- 复现原论文结构

	- 一路是control分支：基于用户和上下文特征，采用self-attention预测y_control，

	- 一路是treatment分支，将多个treatment表征做mean pooling作为干预显示表征，结合用户embedding做treatment aware attention，得到【对当前干预敏感的用户特征embedding】，直接建模tau，uplift tower预测tau1,2,3,4。    

	- 一路cost分支：基于treatment aware后的用户表征做cost预测

	- 因为当前拉取的是RCT数据，暂不考虑用intervention constraint。

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786676446849/cAo6YVF4OLSbxxpFMq2Oy/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=a63bca89e60b385d1eb72a11897a491c730b1f160f376a332ed862f042feea79)

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786676430964/p9FVXe35qxdFgNpc5aBRH/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=24ee28b650b415b47592e78524f334cfb22a97a0b5cacc74ec6234a57b44c366)

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786676440407/ICM6U2IIKQVSUxCSUh4Lr/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=dcd21f1593855e7de4d4752034132c0afb8a4b1ca2abbf089f369f6d36707bd9)

```Python
# control分支
slot embedding
    -> self-attention (attention dim=8)
    -> flatten embedding(control_input)
    -> control_tower
    -> y_control,y_control_for_treatment = tf.stop_gradient(y_control)

# treatment分支
slot embedding + treatment_repr (mean pooling)
    -> treatment-aware attention
       # 根据当前 treatment，学习每个用户 slot 特征的重要性权重 alpha_j
    -> treatment_aware_repr(attention dim=8)
    -> uplift_tower
    -> tau_k

# treatment分支
concat(treatment_aware_repr)
    -> cost_tower
    -> cost_k


y_treat_k = y_control + tau
loss = label_loss + cost_loss
```

##### **2. 模型效果**

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786691630564/_MAwlJB-LMWqeFdMt3idR/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2f48be1d8b09b7a355bfab40af66e5ae08c2c554133f291ff4298afd7e274225)

对比s-learner，auuc平均下降4pp。

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786692027124/1aU4nZpcUSYb6_k6OEKWn/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=25dcf60aa1d0d4bdf379a0607dfc9d7f662a6c1c05c7ee9c525b79ca350ee1b3)

***Q: 模型效果下降***

- `treatment_repr` 用 mean pooling 可能损失信息。

- treatment-aware attention 的表示容量偏弱。

- cost auc效果均下降，考虑到cost_loss 可能干扰 uplift 主目标，cost loss 的梯度也会更新 treatment-aware attention，使 attention 既要解释 response uplift，又要解释 cost。

***A: 后续迭代方向***

- `treatment_repr` 由mean pooling改成concat.

- treatment-aware attention维度，由8扩大成32。

-  cost_loss 权重置 0，隔离 cost 对 uplift 排序的影响，判断treatmet aware效果是否更好。

#### EFIN 1.0

> /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/EFIN/EFIN_4t_01.py

##### **1. 模型结构**

- 一路contol分支，一路treatment分支，将多个treatment embedding concat作为干预显示表征，结合用户embedding做treatment aware attention,建模tau，cost_loss weight置0,暂时隔离 cost 对 uplift 排序的影响。

```Python
slot embedding
  -> self-attention(attention dim=8)
  -> flatten embedding(input)
  -> control_tower
  -> y_control,y_control_for_treatment = tf.stop_gradient(y_control)

slot embedding + treatment_repr(dim=8,concat)
  -> treatment-aware attention  # aij:第i个用户，在当前treatment下，第j个用户特征的重要性权重)
  -> treatment_aware_repr (attention dim = 32,input)
  -> uplift_tower
  -> tau_k

y_treat_k = stop_gradient(y_control) + tau_k
loss = label_loss（MSE）
```

##### **2. 模型效果**

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786678510899/bnRkOfbmYXrTq4Mkng6Wq/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=031d43b7971fe745ebf77398af4425c160ea91bcd635aed59ee2c98678c9a23e)

- 对比EFIN上一版，auuc提升+0.9pp

- 对比s-learner，auuc平均仍下降3pp。

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786693316041/ynBonNwM2j6ckQwIe-I8m/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=b5acd10233cf2f72e58f6f9ba3c5bb99b41b510dbf03ed485afa4754a6f2c39e)

***Q: 模型效果***

- 对比EFIN上一版，auuc提升0.9pp，说明cost采用treatment aware attention后的特征会干扰uplift分支。

- 但对比s-learner，auuc仍下降3pp。考虑：uplift tower的输入是`treatment_aware_repr` ，attention维度=32。该向量虽然通过 attention 显式建模了 treatment 与用户特征之间的交互，但本质上是对用户 slot embedding 的一次加权压缩，给 uplift tower 的输入容量偏弱，可能无法充分表达用户的基础转化倾向和复杂偏好。

***A: 后续迭代方向***

- 持续扩增attention维度会增加计算和训练成本，考虑引入treatment显示表征（帮助共享 uplift tower 区分不同券包）和用户特征，作为整体输入进uplift tower.

#### EFIN2.0

> /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/EFIN/EFIN_4t_02.py

##### **1. 模型结构**

- 复用 control 分支的 self-attention 表征作为全局用户特征，&treatment显示表征，&treatment_aware_repr 作为 uplift tower 输入。

```Python
slot embedding
    -> self-attention(attention dim=8)
    -> flatten embedding(control_input)
    -> control_tower
    -> y_control, y_control_for_treatment = tf.stop_gradient(y_control)

control_input
    -> tf.stop_gradient(control_input)
    -> tau_user_projection(Dense 256)
    -> self_user_repr 
# self_user_repr 是从 control 分支 self-attention 后的用户向量中抽出来的一份 256 维用户表征，用来补充给 uplift tower
# 通过 stop_gradient，uplift loss 不能反向更新 self-attention/control 分支。

slot embedding + treatment_repr(dim=8)
    -> treatment-aware attention  # aij: 第i个用户，在当前treatment下，第j个用户特征的重要性权重
    -> treatment_aware_repr(attention dim=32)
    -> concat(self_user_repr, treatment_aware_repr, treatment_repr)
    -> uplift_tower
    -> tau_k

y_treat_k = stop_gradient(y_control) + tau_k
loss = label_loss(MSE)
```

##### **2. 模型效果**

- 观察到 epoch1 train mean_auuc = 0.6576，较 EFIN01 明显下降，因此该方向提前终止。

***Q: 模型效果下降***

- 做了消融实验EFIN2.1，只加入treatment的显示表征，best mean auuc为0.8148，相较于EFIN1.0（0.8145）有+0.03pp的微弱提升，不显著。

- 因此考虑是加入的用户表征导致模型效果大幅下降，不应该直接复用self-attention后的用户表征，self-attention 后的输出主要服务于 control 分支的自然响应建模；同时为了避免 treatment loss 干扰 control 分支，又对该表征做了 `stop_gradient`，导致 tau 分支拿到的是一个低维且不能被 uplift 目标继续优化的用户表征信息。

***A: 后续迭代方向***

- 单独给uplift分支加入MLP，以原始的用户特征embedding作为输入，训练用户特征。

#### EFIN3.0

> /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/EFIN/EFIN_4t_04.py

##### **1. 模型结构**

- 新增一条从原始 slot embedding 出发的 tau_user_bottom 分支，将原始用户特征embedding flatten 后经过 MLP 输出为 tau_user_repr，再与 treatment_aware_repr、treatment_repr 一起 concat 输入 uplift_tower，用于预测 tau_k。

```Python
slot embedding
    -> self-attention(attention dim=8)
    -> flatten embedding(control_input)
    -> control_tower
    -> y_control


slot embedding
    -> flatten embedding
    -> tau_user_bottom: 1024 -> 512 -> 256
    -> tau_user_repr


slot embedding + treatment_repr(dim=8)
    -> treatment-aware attention
    -> treatment_aware_repr(attention dim=32)
    -> concat(tau_user_repr, treatment_aware_repr, treatment_repr)
    -> uplift_tower
    -> tau_k

y_treat_k = stop_gradient(y_control) + tau_k
loss = label_loss(MSE)
```

##### **2. 模型效果**

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786688298392/fbKw4ETC16vkv3Ln6KuFS/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=4c250b6d5467004157c02920fa7c16649f64855f1a604783bac605f4b341aa9f)

- 对比EFIN1.0，auuc平均+0.5pp

- 对比s-learner，auuc仍 - 3pp。

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786688275775/9ARyE_NCOYxjsAvFtn9e5/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=adc7ef58539732f724dc58eebe23130e23323b5d1918f884307ebdcce2fb89c7)

***Q: 模型效果***

- EFIN04相较于01有微幅提升，但仍较弱于s-learner，考虑self-attenion效果不佳，而s-learner和euen里都有shared_bottom，control和treatment分支可以一同更新embedding信息。而当前control 分支和 tau 分支的用户底座仍然是分开的，用户表征没有形成统一共享

***A: 后续迭代方向***

- 后续迭代方向是参考 S-learner 的强 shared-bottom 结构，将 raw embedding 先输入共享底层 MLP，得到统一的shared_user_repr，再分别供 control、uplift 和 cost 分支使用。

#### EFIN4.0

##### **1. 模型结构**

- 参考 S-learner 的强 shared-bottom 结构，将 raw embedding 先输入共享底层 MLP，得到统一的shared_user_repr，再分别供 control、uplift 和 cost 分支使用。

- 其结构本质为，shared bottom建立共享用户特征embedding，一路control head，一路基于干预特征对用户embedding做交互得到treatment_aware_repr，concat shared_user_repr，treatment_repr，输入uplift tower。一路cost head。

```Python
slot embedding
    -> flatten raw embedding
    -> shared_bottom: 2048 -> 1024 -> 512
    -> shared_user_repr

# control 分支
shared_user_repr
    -> control_tower: 256 -> 128 -> 64 -> 1
    -> y_control

# treatment 分支
slot embedding + treatment_repr(dim=8)
    -> treatment-aware attention -> treatment_aware_repr(attention dim=32)
    -> concat(shared_user_repr, treatment_aware_repr, treatment_repr)
    -> uplift_tower: 256 -> 128 -> 64 -> 1
    -> tau_k

# cost 分支
concat(shared_user_repr, treatment_repr)
    -> cost_tower: 256 -> 128 -> 64 -> 1
    -> cost_k

# treatment outcome
y_treat_k = stop_gradient(y_control) + tau_k
loss = label_loss + cost_loss
```

##### **2. 模型效果**

- 在同一批特征和同一验证窗口下，EFIN的 mean AUUC 达到 0.8505，高于 S-learner 的 0.8460（+0.5pp） 和 EUEN 的 0.8427(+0.77pp)，说明引入 shared-bottom 后，EFIN 的 treatment-aware attention 能在保留强用户表征能力的同时，进一步提升 uplift 排序效果。

- 分 treatment 看，EFIN 在 T1、T3、T4 上均取得最优，分别较 S-learner+ 0.91pp、1.51pp、0.76pp。

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786689841476/4kvOfU4xGj5ohTej20EHV/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c1bfb9c492fe712da3913193317ee4d11947b596d73f0047e226e0347c599128)

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786440896732/wxHWP5jVXdC_QFN78PM7R/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c06afc6ec9c5b9c4d68be0d2536f95c6055f44a6c5fbb56b8989dc5b35cdfaac)

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1786689934684/pJQSFvdG2ndNhxeIU3_Dw/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T063340Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=3f06a8f92746cfc05bd9bc0c93762cae15952d20fcabe99d4f18f1f1bb4af925)

### 附录

- efin源码：[https://github.com/dgliu/KDD23_EFIN/blob/main/models/architecture.py](https://github.com/dgliu/KDD23_EFIN/blob/main/models/architecture.py)

- 论文：[https://arxiv.org/pdf/2306.00315](https://arxiv.org/pdf/2306.00315)