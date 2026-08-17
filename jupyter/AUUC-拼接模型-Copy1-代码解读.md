# AUUC-拼接模型-Copy1 代码解读

本文解读 `AUUC-拼接模型-Copy1.ipynb` 的作用、指标含义，以及每个单元格在做什么。

## 1. `caculate_auc(y_labels, y_scores)` 在做什么

这个函数是一个手写的排序 AUC，主要用于评估一个预测分数 `y_scores` 是否能把真实数值 `y_labels` 从低到高排好。

它的输入是：

```python
caculate_auc(y_labels, y_scores)
```

- `y_labels`：真实值，例如真实成本 `cost` / `cost_7d`，也可以是多档 label。
- `y_scores`：预测分数，例如 `coupon_prob` 或预测成本分。

它的核心逻辑是：

1. 找出真实 label 的所有取值，并从大到小排序。
2. 按预测分数 `y_scores` 从小到大排序样本。
3. 遍历排序后的真实 label，统计“预测分数排序是否和真实 label 高低关系一致”的样本对数量。
4. 用正确排序的样本对数量除以所有可比较样本对数量，得到 AUC。

更直白地说，它在回答：

```text
如果真实成本更高的用户，模型分数也更高，那么 AUC 会高。
如果模型分数和真实成本没关系，AUC 接近随机。
如果排序反了，AUC 会低。
```

它和 sklearn 的二分类 `roc_auc_score` 不完全一样，因为这里支持多档或连续型 label 的排序比较，更像“多级标签排序 AUC / ranking AUC”。

注意点：

- 这个函数适合评估“排序能力”，不评估数值校准。
- 它没有处理 `total_len=0` 的情况，如果所有 label 都一样会除零。
- 它当前按 `y_scores` 从小到大排序，但统计逻辑本身是在数“低分在前、高 label 在后”的一致性；使用时要确认分数方向是否符合预期。

## 2. 图片是不是 AUUC 曲线，如何理解

你给的图片标题是 `qini-curve`，严格说它是 **Qini 曲线**，不是最朴素的 AUUC 曲线。但 Qini、uplift curve、AUUC 是同一家族的 uplift 模型评估图，都在评估“按模型分数排序圈人后，真实增量收益累积得快不快”。

图片里每个子图对应一个 treatment：

```text
treatment: 1 | Qini: 0.31682 | ite: 0.12353
treatment: 2 | Qini: 0.32707 | ite: 0.04213
...
```

常见线条含义：

- `random`：随机排序基线。随机圈人时，增量收益按人群比例线性增长。
- `predict`：按模型预测 uplift 从高到低排序后的累计增量收益。
- `actual`：某种理想或真实排序参考，通常代表更接近真实 uplift 排序后的曲线。
- `treatment_control_ratios`：右轴，表示累计人群里 treatment/control 样本比例是否平衡。

怎么读这张图：

1. 横轴 `Ite ranks`：按预测 ITE/uplift 排序后，从高分到低分逐步纳入的人数。
2. 纵轴 `Qini`：累计增量收益，通常可以理解为 treatment 相对 control 多带来的转化或完单。
3. `predict` 曲线越早、越快高于 `random`，说明模型越能把“高增益人群”排在前面。
4. `Qini-Score` 或曲线面积越大，排序越好。
5. 黑线如果波动很大，说明 treatment/control 比例不稳定，曲线可信度会受影响。

AUUC 曲线的理解也类似：

```text
按模型 uplift 分数从高到低排序
取 top 1%、top 5%、top 10%...
分别计算这些人里的真实 treatment-control 增量
把这些累计增量连成曲线
曲线下面积就是 AUUC
```

## 3. 为什么有些用 AUUC，有些用 AUC

因为要评估的对象不同。

### AUUC 用于评估 uplift 排序

AUUC 评估的是“干预增量”的排序能力。

例如：

```text
left63 = 发 6 张 3 元券相对空白的预测增量完单
```

这时要看：

```text
模型认为 left63 高的人，真实发券相对空白是否真的增量更大？
```

所以用 AUUC / Qini / uplift curve。

### AUC 用于评估普通排序

AUC 评估的是普通 label 的排序能力。

例如：

```text
预测成本 cost_1
真实成本 cost_7d
```

这时问题是：

```text
预测成本高的人，真实成本是否也高？
```

这不是 treatment-control 增量问题，只是普通排序问题，所以用 AUC / xAUC / ranking AUC。

### 对应到这份 notebook

- `vis.draw_auuc(...)`：评估 uplift 模型，使用 AUUC。
- `caculate_auc(...)`：评估成本或概率分数的普通排序，使用 AUC。
- `left63 / left65 / left853 / left*_minus_left63`：用 AUUC。
- `cost / cost_7d / cost_1 / cost_2 / cost_3 / coupon_prob`：常用 AUC 或排序分桶检查。

## 4. Notebook 单元格逐个说明

### Cell 0：启动 Spark 环境

配置 Spark2/Spark3 的 yarn 参数、executor 数量、内存、队列、TF 环境包、tfrecord jar 等，然后通过 `dmpax.SparkBaseApp()` 初始化 Spark。

输出：

- Spark session：`spark`
- HDFS helper：`hdfs`
- Spark 版本打印

用途：后续所有 `spark.sql(...)` 都依赖这个单元格。

### Cell 1：导入评估和运筹依赖

导入 PySpark、Pandas、Numpy、Matplotlib、Seaborn、OR-Tools、sklearn AUC，以及 `backflow` 里的实验、校准、可视化、仿真工具。

关键对象：

- `vis.draw_auuc`：后续 AUUC 曲线评估核心函数。
- `pywraplp`：后续运筹优化使用的 OR-Tools solver。

### Cell 2：重复导入基础依赖

再次导入常用包，和 Cell 1 有较多重复。看起来是为了在不同执行上下文下保证依赖存在。

作用：环境兜底，无核心业务逻辑。

### Cell 3：定义 `caculate_auc`

定义手写排序 AUC 函数。用于评估预测分数对真实多档 label 的排序能力，常见用途是成本预测排序。

输出：一个 AUC 数值。

### Cell 4：Markdown 标题 `euen_base`

标记下面开始评估 `euen_base` 或相关模型版本。

### Cell 5：取 RCT label 并拼接 EUEN 推理表

定义 `QUERY_ALL_FMT`，从真实 label 表取实验样本，再 left join 模型推理表。

真实 label 字段包括：

- `pid`, `oneid`
- `group_name_eve`
- `uv_label`, `pv_label`
- `uv_label_7d`, `pv_label_7d`
- `cost`, `cost_7d`
- `dt`

模型预测字段包括：

- `uplift_1/4/5/6`
- `cost_1/4/5/6`

然后执行 SQL，转成 Pandas：`v117 = all_rct_data_df.toPandas()`。

### Cell 6：检查 `BASE_FMT_v1` 的 uplift/cost 单调性

过滤掉模型未匹配的 `999`，然后检查三类比例：

- CVR/uplift 是否满足 `left63 >= 0`, `left853 > left63`, `left65 > left853`
- Cost 是否满足 `cost_1 >= 0`, `cost_2 > cost_1`, `cost_3 > cost_2`
- 两者是否同时满足

输出：打印满足条数、比例。

### Cell 7：检查 `BASE_FMT_v4` 的 uplift/cost 单调性

逻辑与 Cell 6 一样，但评估对象换成 `BASE_FMT_v4`。

用途：比较另一个模型版本的业务合理性。

### Cell 8：检查 `BASE_FMT_v2` 的 uplift 单调性

只检查 CVR/uplift 单调性，不检查 cost。

用途：适用于 `BASE_FMT_v2` 没有完整 cost 字段或当前只关心 uplift 的情况。

### Cell 9：构建 `BASE_FMT_v1`

从 `dm.ut_alg_flowback_msg_outcall_label_di` 取 2026-05-06 到 2026-05-11 的实验样本，拼接：

```text
growstrategy.ut_mkt_backflow_multi_treat_euen_v216_xnetwork12_new_uplift_epoch2_1_model_inference_di
```

字段口径：

- `uplift_1 -> left63`
- `uplift_2 -> left853_minus_left63`
- `uplift_3 -> left65_minus_left63`
- `left65 = left63 + left65_minus_left63`
- `left853 = left63 + left853_minus_left63`
- `cost_1/2/3`

这是“基底 + 残差拼接”的典型单元格。

### Cell 10：成本/概率排序 AUC

筛选 `group_name_eve == 留存6张3元` 的样本，用：

```python
caculate_auc(left6_df['cost_7d'], left6_df['coupon_prob'])
```

评估 `coupon_prob` 对真实 7 日成本或相关 label 的排序能力。

注意：`cost_7d_df` 不在当前 cell 中定义，依赖前面或外部执行状态。

### Cell 11：评估高券相对基底券的 AUUC

```text
treatment = 3
control_group_name = 1
score = left65_minus_left63
```

含义：评估 `留存6张5元` 相对 `留存6张3元` 的额外增量排序能力。

### Cell 12：评估 8 元券相对基底券的 AUUC

```text
treatment = 2
control_group_name = 1
score = left853_minus_left63
```

含义：评估 `8元留存5张3` 相对 `留存6张3元` 的额外增量排序能力。

### Cell 13：评估基底券相对空白的 AUUC

```text
treatment = 1
control_group_name = 0
score = left63
```

含义：评估 `留存6张3元` 相对空白组的 uplift 排序能力。

### Cell 14：另一次成本/概率排序 AUC

筛选 `BASE_FMT_v2` 中 `留存6张3元` 样本，计算：

```python
caculate_auc(left6_df['cost'], left6_df['coupon_prob'])
```

这里用的是当日 `cost`，不是 `cost_7d`。

### Cell 15：Markdown 标题 `因果森林`

下面开始评估因果森林或 `dm.ut_mkt_backflow_model_inference_di` 相关模型。

### Cell 16：构建 `BASE_FMT_v2`

从 label 表取真实实验样本，拼接 `dm.ut_mkt_backflow_model_inference_di`。

典型字段：

- `prediction[0] -> left63`
- `prediction[1] -> left853`
- `prediction[2] -> left65`
- `left65_minus_left63 = left65 - left63`
- `left853_minus_left63 = left853 - left63`
- 额外拼接 `coupon_prob`

用途：评估一个直接输出多个 treatment uplift 的模型。

### Cell 17：重新构建 `BASE_FMT_v2`

和 Cell 16 类似，是另一个时间段或模型配置版本的 `BASE_FMT_v2`。用途是覆盖前面的 `BASE_FMT_v2`，便于继续评估。

### Cell 18：计算分桶 treatment 比例差异

调用：

```python
calculate_diff_rate(BASE_FMT_v2, ['left63'], [0, 6])
```

按 `left63` 分 10 桶，看 `real_treatment` 为 0 和 6 的样本比例在不同桶中是否稳定。

### Cell 19：评估 `left65` 的 AUUC

用 `left65` 作为预测分数，评估对应 treatment 相对 control 的 uplift 排序能力。

### Cell 20：评估残差 uplift 的 AUUC

同时评估：

- `left853_minus_left63`
- `left65_minus_left63`

且 `control_group_name=1`，表示高券相对基底券的增量评估。

### Cell 21：评估 treatment 5 的残差 AUUC

用 `left65_minus_left63` 作为分数，评估 `real_treatment=5` 相对指定 control 的 AUUC。

注意：这里 treatment 编码和前面 1/2/3 体系不同，说明 notebook 混合了多个实验版本。

### Cell 22：评估 treatment 6 的基底 AUUC

用 `left63` 作为分数，评估 `real_treatment=6`。

注意：同样属于另一套 treatment 编码。

### Cell 23：重新构建早期时间段 `BASE_FMT_v2`

时间改成 2026-04-01 到 2026-04-07，用于逐天或历史模型评估。

### Cell 24：Markdown 标题 `xnetwork`

下面开始评估 xnetwork 版本。

### Cell 25：检查 `BASE_FMT_v4` 的完整业务合理性

过滤 `999` 后，检查：

- uplift 是否递增
- cost 是否递增
- 两者是否同时满足

这是 xnetwork 模型版本的 sanity check。

### Cell 26：构建 `BASE_FMT_v4`

从 label 表取样本，拼接：

```text
growstrategy.ut_mkt_backflow_multi_treat_euen_v216_xnetwork11_new_uplift_epoch2_1_model_inference_di
```

字段口径：

- `uplift_1 -> left63`
- `uplift_2 -> left853`
- `uplift_3 -> left65`
- `left65_minus_left63 = left65 - left63`
- `left853_minus_left63 = left853 - left63`
- `cost_1/2/3`

这是“直接输出绝对 uplift，再反向计算残差”的口径。

### Cell 27：评估 `left63` AUUC

评估 `留存6张3元` 相对空白组的排序能力。

### Cell 28：评估 `left853` AUUC

评估 `8元留存5张3` 相对空白组的排序能力。

### Cell 29：评估 `left65` AUUC

评估 `留存6张5元` 相对空白组的排序能力。

### Cell 30：评估 xnetwork 残差 AUUC

评估：

- `left853_minus_left63`
- `left65_minus_left63`

control 设置为 1，表示相对基底券的增量排序。

### Cell 31：评估 `BASE_FMT_v133` 的 treatment 5

用 `left65_minus_left63` 评估另一个模型版本 `BASE_FMT_v133`。

注意：`BASE_FMT_v133` 不在当前附近定义，依赖 notebook 运行历史。

### Cell 32：评估 `BASE_FMT_v133` 的 treatment 6

用 `left63` 评估另一个模型版本的基底 uplift。

### Cell 33：Markdown 标题 `逐天euen`

下面开始评估逐天 EUEN 模型。

### Cell 34：构建 `BASE_FMT_v7`

取 2026-04-01 到 2026-04-22 的 label 数据，拼接逐天 EUEN 模型推理表。

字段口径倾向于：

- 输出 `left63`
- 输出残差 `left65_minus_left63`
- 输出残差 `left853_minus_left63`
- 再拼接出绝对 uplift

### Cell 35：查看 `BASE_FMT_v132` 行列规模

输出 `BASE_FMT_v132.shape`，用于确认样本量。

注意：`BASE_FMT_v132` 依赖前面或外部执行状态。

### Cell 36：查看 treatment 分布

输出：

```python
BASE_FMT_v132['real_treatment'].value_counts(dropna=False)
```

用于检查不同实验组样本量是否足够、是否有空值。

### Cell 37：定义 `calculate_diff_rate`

按预测分数分 10 个分位桶，统计指定 treatment 集合在每个桶里的占比，并计算最大占比差 `diff_rate`。

用途：检查分桶后 treatment/control 分布是否过度偏斜。

### Cell 38：调用 `calculate_diff_rate`

按 `left63` 分桶，检查 `real_treatment` 为 0 和 6 的分布差异。

### Cell 39：由残差恢复绝对 uplift

计算：

```python
BASE_FMT_v7['left65'] = BASE_FMT_v7['left63'] + BASE_FMT_v7['left65_minus_left63']
BASE_FMT_v7['left853'] = BASE_FMT_v7['left63'] + BASE_FMT_v7['left853_minus_left63']
```

用途：把“基底 + 残差”模型输出转成每个券档相对空白的绝对 uplift。

### Cell 40：评估 `BASE_FMT_v132` treatment 5 残差 AUUC

用 `left65_minus_left63` 评估 treatment 5。

### Cell 41：评估 `BASE_FMT_v132` treatment 6 基底 AUUC

用 `left63` 评估 treatment 6。

### Cell 42：Markdown 标题 `base_dfcl`

下面开始评估 dfcl 版本模型。

### Cell 43：构建 `BASE_FMT_v131` for dfcl

从 `growstrategy.ut_alg_flowback_msg_label_v2_di` 取样本，拼接 dfcl 版本模型推理表。

输出字段包括：

- `uv_label`
- `pv_label_7d`
- `cost_7d`
- `left63`
- `prob555_minus_left63`
- `left65_minus_left63`

用途：评估 dfcl 模型的不同 uplift 字段。

### Cell 44：再次定义 `calculate_diff_rate`

和 Cell 37 重复。可能是 notebook 分段运行时的函数兜底。

### Cell 45：对 `BASE_FMT_v131` 做分桶差异检查

按 `left63` 分桶，检查 treatment 0 和 6 分布差异。

### Cell 46：评估 `prob555_minus_left63` AUUC

评估某个券档相对 `left63` 的残差 uplift。

### Cell 47：评估 `left65_minus_left63` AUUC

评估 6 张 5 元相对基底券的残差 uplift。

### Cell 48：评估 `left63` AUUC

评估基底券相对 control 的 uplift 排序能力。

### Cell 49：Markdown 标题 `zlin_0.05`

下面开始评估 zlin 版本模型。

### Cell 50：构建 zlin 版本 `BASE_FMT_v131`

取真实 label 数据，拼接：

```text
growstrategy.ut_mkt_backflow_multi_treat_euen_v212_ziln124_new_2_new_uplift_model_inference_di
```

用途：评估 zlin 版本的 uplift 输出。

### Cell 51：再次定义 `calculate_diff_rate`

与 Cell 37/44 重复，供当前段落使用。

### Cell 52：对 zlin 版本做分桶差异检查

按 `left63` 分桶，检查 treatment 0 和 6 的分布差异。

### Cell 53：评估 `prob555_minus_left63` AUUC

评估 zlin 版本中某个残差 uplift 字段。

### Cell 54：评估 `left65_minus_left63` AUUC

评估 zlin 版本中高券相对基底券的残差 uplift。

### Cell 55：评估 `left63` AUUC

评估 zlin 版本中基底 uplift 的排序能力。

### Cell 56 到 Cell 65：空单元格

这些单元格没有代码，可能是 notebook 编辑过程中留下的空白。

### Cell 66：检查特征表字段非空数

执行 Spark SQL：

```sql
select
  count(dm_user_base_oneid_info_last_use_htm) as use_htm_cnt,
  count(dm_user_deal_pub_rfm_bus_score_rank) as bus_score_cnt
from dm.ut_alg_flowback_msg_outcall_feature_train_di
where dt = '2026-03-03'
```

用途：检查特征表中某些字段是否有值，辅助排查特征缺失。

### Cell 67 到 Cell 68：空单元格

没有代码。

### Cell 69：定义运筹策略全套函数

这是一个很大的工具函数单元格，核心不是 AUUC，而是策略运筹。

主要函数：

- `smart_generate_dt_list`：生成日期窗口。
- `calculate_delta_metrics`：计算某券组相对 control 的真实增量 CAC 和增量 CVR。
- `process_df`：按桶做样本放缩，得到 `zoom_finish`, `zoom_cost`, `zoom_delta_finish`, `zoom_delta_cost`。
- `data_prepare`：把桶级收益和成本转成 OR-Tools 输入向量 `Y` 和 `C`。
- `ortools_coupon`：用整数规划求解每个桶选哪种券，目标最大化增量完单，约束是成本不能超过 `set_cost * 增量完单`。
- `or_convert`：把 solver 的 0/1 解转回 `{bucket: coupon}`。
- `mock_use`：用训练集或验证集 RCT 聚合数据模拟策略效果。
- `mock_use_simulation`：按策略结果重新合并桶后做仿真。
- `split_small_data` / `split_small_data_debug`：把样本充足和样本不足的分桶组合拆开。
- `generate_bucket_mapping_final`：为多阶段分桶生成 bucket id。
- `simulate_and_plot` / `simulate_and_plot_all`：遍历不同 CAC 约束，画训练集/测试集上的策略曲线。
- `assign_bucket_num`：给每条样本匹配 bucket id，匹配不到为 -1。
- `main_try`：运筹主入口，完成分桶、聚合、求解、仿真、画图。
- `show_combination`：展示多阶段分桶的大/小样本组合。
- `calculate_percentile_thresholds`：用训练集计算分位点阈值。
- `assign_quantile_buckets`：按训练集阈值给数据打分位桶。
- `move_threshold_main`：滑动窗口计算训练/验证分桶。
- `plot_multi_results`：比较两套策略曲线。

输出：

- `bucket_mapping`
- `processed_data`
- `next_processed_data`
- `discount_res_list`
- `train_list`
- `test_list`
- `tongbu_list`
- `tongbu_all_list`
- 训练集/测试集策略曲线图

### Cell 70：生成运筹日期窗口

设置：

```python
EXE_DT = '2026-04-13'
__val_days = 14
CHECK_DTS = smart_generate_dt_list(...)
```

用途：为后续策略运筹生成训练/验证日期窗口。

### Cell 71：空单元格

没有代码。

## 5. 总结

这份 notebook 的主要输出包括：

1. 多个模型版本的 AUUC 数值和曲线。
2. 基底 uplift 与残差 uplift 的 AUUC。
3. 成本或概率分数的排序 AUC。
4. uplift/cost 单调性比例报告。
5. 预测分桶后的 treatment 分布差异表。
6. 后半段策略运筹的 CAC-增量完单曲线和每桶发券结果。

它不是单一模型训练文件，而是一个“模型推理结果拼接 + 离线评估 + 策略运筹试验”的 notebook。
