<!-- 请注意：文档内的附件与图片将在 2026/10/27 14:29:49  失效，请您在有效期内及时下载并保存至本地
Note: Attachments and images in the document will expire at 2026/10/27 14:29:49. Please download and save them locally within the valid period.-->

# [模型迭代]EEUEN


## 背景

当前短信外呼发券场景都有触达链路，用户被分到treatment组后可能会因为短信发送失败，外呼未接通等原因没有真正曝光。 目前采用的EUEN 直接学习 treatment 对转化的整体增量时，未成功曝光的样本会稀释 treatment 的真实效果，使模型难以区分“触达失败”和“曝光后无效”。

因此，本轮将 EUEN 迭代为 EEUEN，在原有 uplift/cost 多任务结构上新增 exposure head，显式预测用户成功曝光概率并对最终uplift进行加权，以更贴合短信、外呼这类触达型干预场景。

同时，由于当前模型不是天级别更新，训练样本的时效性也可能影响线上效果。因此本轮实验还同步比较两组训练窗口：一组沿用v2版时间窗口，另一组在此基础上加入近期样本，在同一窗口测试集上评估模型效果，评估增加训练样本是否能带来额外收益和后续训练样本的更新频率。

## conclusion

基于三组离线样本数据进行训练，对比eeuen vs euen 发现：

- 深度模型效果：eeuen 四组auuc均高于euen，mean auuc 在三期数据中分别 +2.65pp，+6.39pp，+7.71pp

- 运筹仿真效果：在约束cac\[3.5,6.0)下，eeuen在同一cac下的cvr基本均高于euen。

## 迭代过程

### EEUEN1.0

#### 1. 模型结构

- 新增exposure head，用 exp_label 监督用户是否成功曝光

> send_success_cnt 或 outcall_sucess_cnt任一大于0即为曝光成功，exp_label=1

- uplift分支为曝光后的增益，因此最终用来排序的uplift分数： delta_cvr_k = p_exp \* tau_e_k，即为uplift分支输出的增量和曝光概率的结合。

- 损失函数loss = outcome_loss(mse)+ cost_loss(mse) + exp_loss(bce)

#### 2. 模型效果

> **eeuen vs euen：**

> 【v1】 train：0511-0617                            test：0623-0706

> 【v2】 train：0511-0617                            test：0723-0806

> 【v3】 train：0511-0617+0623-0722     test：0723-0806

- 对于v1训练窗口期，eeuen四个treatment的auuc均高于euen，离线mean auuc相对+2.65pp

- 对于v2训练窗口期，eeuen四个treatment的auuc均高于euen，离线mean auuc相对+6.39pp

- 对于v3训练窗口期，eeuen四个treatment的auuc均高于euen，离线mean auuc相对+7.71pp

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1787737399519/tU7Db_UyXfC_Pg78ULp6z/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T062949Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=381598f5f6f5a1ec1c8d68d3e52ef4c090e98c31f02c54bf92358174008f0d10)

#### 3. 运筹仿真

- 基于选取的best_stage参数，对比不同训练窗口下eeuen vs euen的效果：

在约束cac\[3.5,6.0)下，eeuen在同一cac下的cvr基本均高于euen。

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1787817537080/sdw3xL4XAiEQuaWi_VQke/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T062949Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=48c3f8ee8ab52773c767bae1d9558f0b0fc9c73f006db83e6d409c13e829ca51)

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1787816346176/-76g1aYgKaoHx0KmrY0SC/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T062949Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=70e24981c0f5ab43e3b7f7546b766dc1f2e106f29465d348b558b4f6f59de3d6)

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1787816365157/dbNun0mEk5QePnlO51Eoo/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T062949Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=ced3b77d6a9397b4d3c3fb4f0299913dd064f80ad701e0c50e92f59212b364ff)

### EEUEN2.0

#### 1.模型结构

- 考虑到不同treatment间的触达成功率有所差异，因此把1.0共享的exposure_head分成四个exposure_head独立学习各个treatment的触达率。最终用来排序的uplift分数： delta_cvr_k = exposure_prob_k \* tau_e_k

#### 2. 模型效果

> **eeuen1.0 vs eeuen2.0**

> 【v1】 train：0511-0617                            test：0623-0706

> 【v2】 train：0511-0617                            test：0723-0806

- eeuen 2.0拆分成四个exposure head之后，和eeuen1.0对比，在不同测试期上效果不稳定。

	- v1期，mean auuc +0.63pp

	- v2期，mean auuc -1.14pp

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1787818311678/lSrVdUsiJeuH-Mcx3LrlE/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T062949Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d9f02c0886c2d8e77019c172cb2cd7a0195ab71b91293baf1d9eb2a720b048d5)

### **样本量增加收益：**

> 【v2】 train：0511-0617                            test：0723-0806

> 【v3】 train：0511-0617+0623-0722     test：0723-0806

#### **1.模型效果**

	- 对于eeuen模型，训练样本量增加后，mean auuc +1.96pp

	- 对于euen模型，训练样本量增加后，mean auuc +0.64pp

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1787213011958/My88HDCqlt4UQs2sNpvnW/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T062949Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=d7e41ef33f2af8ca8db79c6d85e30178b2453c86478f6a607d15b712bdcda0c4)

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1787213171079/bdtOaNH0dLzsWoI1PS7-u/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T062949Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=c10e8c4faa50366275f7bae41c3fc6656761a98d6e47048403ecbcf9b73815a7)

#### 2.运筹仿真

- 基于选取的best_stage参数，对比不同增加训练集数据并拉齐test下的仿真效果：v3 相比 v2 没有稳定带来收益提升。

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1787821633874/Ricx58ZpL2JkG9FyqW7_F/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T062949Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=5bf74fb0ad1fafe7f34b4ea9f7d443baab23925b5bd7e17bf0d6fbac6610eeaf)

![](https://s3-ep-inter.didistatic.com/didoc2-upload-file-prod/images/1787821681322/6f-q0Hbe0Y2wP85q64dAj/image.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKDD00000000000EBLCTZNRBJHX6ZS%2F20261026%2Fgzep%2Fs3%2Faws4_request&X-Amz-Date=20261026T062949Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=dc058bebad92e35906dc75020a5de42d17b205d56eeb1b9959ae973da09b4092)

**附录**

*sql路径*

- /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/sqls/v2

- /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/offline_bigdata/script/sqls/v3

- 

*eeuen模型*

- /nfs/dataset-ofs-385-143496/yangliu/lite-fea-workspace/wwt/demo/EEUEN/EEUEN_4t.py

*test hive落表*

> <https://ether.intra.xiaojukeji.com/customTask/snapshotDetail?id=3857>

- growstrategy.ut_msg_outcall_eeuen_exposure_v1_0511_0617_inference_di

- growstrategy.ut_msg_outcall_eeuen_exposure_v2_0511_0617_inference_di

- growstrategy.ut_msg_outcall_eeuen_exposure_v3_0511_0722_inference_di

表格：[==https://cooper.didichuxing.com/docs2/sheet/2209395469994==](https://cooper.didichuxing.com/docs2/sheet/2209395469994)