# Recommendation Project

## 1. 实验要求中文翻译

原始要求的核心内容如下：

- 任务目标：预测 `Test.txt` 中所有用户-物品对 `(u, i)` 的评分。
- 可用数据：
  - `Train.txt`：用于训练模型；
  - `Test.txt`：用于最终测试预测；
  - `ResultForm.txt`：结果文件格式示例；
  - `DataFormatExplanation.txt`：数据格式说明。
- 报告中至少应包含：
  - 数据集基本统计信息，例如用户数、物品数、评分数等；
  - 算法细节；
  - 推荐算法的实验结果，例如 `RMSE`、训练时间、空间消耗；
  - 算法的理论分析和/或实验分析。
- 不能使用 `Test.txt` 参与训练或调参。
- `Test.txt` 中没有真实评分，因此不能计算 `Test RMSE`，也不能声称某个算法“在 Test.txt 上最好”。

## 2. 项目结构说明

```text
recommendation_project/
├── data/
│   ├── Train.txt
│   ├── Test.txt
│   ├── ResultForm.txt
│   └── DataFormatExplanation.txt
├── src/
│   ├── baseline.py
│   ├── ensemble.py
│   ├── item_cf.py
│   ├── matrix_factorization.py
│   ├── utils.py
│   ├── run_experiment.py
│   ├── compare_cv_folds.py
│   ├── tune_ensemble.py
│   └── predict_final.py
├── results/
│   ├── experiment_results.csv
│   ├── compact_storage_comparison.csv
│   ├── compact_storage_comparison_folds.csv
│   ├── exploration_best_blend_fine.csv
│   └── final_result.txt
├── README.md
└── requirements.txt
```

各文件作用：

- `baseline.py`：全局平均分 + 用户偏置 + 物品偏置。
- `ensemble.py`：线性融合 ItemCF 与 MatrixFactorization 的预测结果。
- `item_cf.py`：基于物品的协同过滤，使用相似物品的加权残差做预测。
- `matrix_factorization.py`：带偏置项的矩阵分解，使用 SGD 训练。
- `utils.py`：数据读取、数据划分、RMSE、结果写出等公共函数。
- `run_experiment.py`：统一实验入口，用于验证集评估与结果对比。
- `compare_cv_folds.py`：比较 `3`、`5`、`10` 折交叉验证的评估稳定性和计算开销。
- `tune_ensemble.py`：融合权重细粒度搜索脚本。
- `optimized_ensemble.py`：优化后三模型融合模型，最终提交结果使用该模型生成。
- `predict_final.py`：最终预测入口，可通过 `--model optimized_ensemble` 显式生成最终提交文件。

## 3. 数据文件说明

数据格式由 `DataFormatExplanation.txt` 给出：

- `Train.txt`
  - 第一行块头格式：`user_id|rating_count`
  - 接下来的 `rating_count` 行格式：`item_id score`
- `Test.txt`
  - 第一行块头格式：`user_id|item_count`
  - 接下来的 `item_count` 行格式：`item_id`

本项目将训练集和测试集都读取成扁平的 `(user, item)` 或 `(user, item, rating)` 列表，方便统一训练和预测。

说明：

- 当前数据目录下的 `ResultForm.txt` 更像“格式示例”而不是完整模板，因此最终输出时脚本会优先按它解析；若条目数不匹配，则自动退化为按 `Test.txt` 的用户分块结构输出，格式仍然符合要求。

## 4. 四种算法思路

### 4.1 Baseline

预测公式：

```text
r_hat(u, i) = mu + b_u + b_i
```

其中：

- `mu` 是全局平均分；
- `b_u` 是用户偏置；
- `b_i` 是物品偏置。

训练时使用带正则化的交替更新。该方法实现简单、训练快，并且能为未知用户或未知物品提供稳定 fallback。

### 4.2 Item-based Collaborative Filtering

主要流程：

1. 先训练一个 `BaselineModel` 作为基础预测器；
2. 对每个已知评分计算残差：`residual = rating - baseline_prediction`；
3. 构建用户-物品残差矩阵；
4. 计算物品之间的相似度；
5. 对目标物品选取 `top_k` 个最相关邻居，使用加权残差修正 baseline 预测值。

预测公式可以写成：

```text
r_hat(u, i) = baseline(u, i) + sum(sim(i, j) * residual(u, j)) / sum(|sim(i, j)|)
```

当目标用户没有可用近邻，或者用户/物品未知时，自动回退到 baseline 预测。

### 4.3 Matrix Factorization

预测公式：

```text
r_hat(u, i) = mu + b_u + b_i + p_u^T q_i
```

其中：

- `p_u` 是用户隐向量；
- `q_i` 是物品隐向量；
- `b_u` 和 `b_i` 分别是用户偏置与物品偏置。

训练时用随机梯度下降 `SGD` 更新参数，并固定随机种子为 `2026`。对于未知用户或未知物品，只使用已知偏置项和全局平均分进行 fallback。

### 4.4 Ensemble

融合模型将 `ItemCF` 和 `MatrixFactorization` 的预测结果做线性加权：

```text
r_hat_ensemble(u, i) = alpha * r_hat_itemcf(u, i) + (1 - alpha) * r_hat_mf(u, i)
```

其中 `alpha` 是 `ItemCF` 的权重。本项目先通过 `5` 折交叉验证粗略搜索 `alpha = 0.0, 0.1, ..., 1.0`，再在 `0.45` 到 `0.65` 区间内按 `0.01` 步长细扫。当前最优权重为 `alpha = 0.55`，即 `55% ItemCF + 45% MatrixFactorization`。融合模型的目的不是引入新的特征，而是利用两个单模型误差结构的互补性改善最终预测。

## 5. 实验流程与如何运行实验

本项目现在采用的标准流程就是下面这 7 步：

1. 只使用 `Train.txt` 做模型选择，不碰 `Test.txt` 的真实答案。
2. 将 `Train.txt` 按用户评分做 `5` 折交叉验证。
3. 四个算法分别在 `5` 个折上重复训练与验证。
4. 比较四种算法的平均验证集 `RMSE`。
5. 选出平均验证集 `RMSE` 最低的算法。
6. 用完整 `Train.txt` 重新训练这个算法。
7. 对 `Test.txt` 做预测，生成 `results/final_result.txt`。

这里特别强调：

- `Test.txt` 没有真实评分，所以不能计算 `Test RMSE`；
- 模型优劣只能基于 `Train.txt` 内部划分出的验证集，或者基于 `K` 折交叉验证来比较；
- 最终提交文件只是预测结果文件，不是测试集评估结果文件。

先进入项目根目录：

```bash
cd recommendation_project
```

运行统一实验：

```bash
python3 src/run_experiment.py
```

该脚本会完成：

- 读取 `Train.txt`；
- 输出数据集统计信息；
- 按用户评分构造 `5` 折交叉验证；
- 每次使用 `4` 折训练、`1` 折验证；
- 对 `Baseline`、`ItemCF`、`MatrixFactorization`、`Ensemble` 分别重复 `5` 次训练与验证；
- 记录每一折的 `RMSE`，并计算平均 `RMSE` 与标准差；
- 统计平均训练时间、平均预测时间、平均近似空间消耗；
- 输出终端对比表；
- 保存结果到 `results/experiment_results.csv`。

## 6. 如何生成最终预测结果

先运行实验脚本，得到交叉验证结果：

```bash
python3 src/run_experiment.py
```

然后生成最终预测结果。本次最终提交使用优化后三模型融合模型：

```bash
python3 src/predict_final.py --model optimized_ensemble
```

也可以选择其他模型进行对照：

```bash
python3 src/predict_final.py --model itemcf
python3 src/predict_final.py --model baseline
python3 src/predict_final.py --model mf
python3 src/predict_final.py --model ensemble
python3 src/predict_final.py --model auto
```

脚本流程：

- 若使用 `--model optimized_ensemble`，使用报告中最终确定的三模型融合；
- 若使用 `--model auto`，先从基础实验的 `experiment_results.csv` 中找出平均验证集 `RMSE` 最低的模型；
- 使用完整 `Train.txt` 训练该模型或你显式指定的模型；
- 读取 `Test.txt` 中所有待预测用户-物品对；
- 预测评分；
- 将评分截断到训练集最小值和最大值之间；
- 保留 4 位小数；
- 输出到 `results/final_result.txt`。

## 7. 输出文件说明

- `results/experiment_results.csv`
  - 保存四种模型在 `5` 折交叉验证中的每折 `RMSE`、平均 `RMSE`、`RMSE` 标准差、平均训练时间、平均预测时间、平均近似空间消耗。
- `results/final_result.txt`
  - 保存最终提交用的预测结果，格式与题目要求一致。

## 8. 报告中可引用的指标说明

报告中建议明确写出下面这句话：

> `Test.txt` 不包含真实评分，因此本项目不能计算 Test RMSE。算法选择完全基于 Train.txt 上的 5 折交叉验证结果，最终使用平均验证集 RMSE 最低的模型在完整 Train.txt 上重新训练后，对 Test.txt 进行预测。

当前一次 `5` 折交叉验证运行得到的结果如下，可直接作为报告初稿中的对比数据：

| model | mean_rmse | std_rmse | avg_train_seconds | avg_predict_seconds | avg_memory_mb |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 17.524633 | 0.036371 | 0.4774 | 0.0049 | 0.7462 |
| ItemCF | 17.112610 | 0.045719 | 4.9639 | 0.8701 | 15.1663 |
| MatrixFactorization | 17.193329 | 0.027584 | 10.8121 | 0.0210 | 0.9526 |
| Ensemble | 16.952308 | 0.035910 | 16.1105 | 0.6878 | 16.1189 |

进一步的多随机种子验证中，最终 ItemCF、48 维 MatrixFactorization 与 UserCF 三模型融合的平均 RMSE 为 `16.859807`，核心结构占用约为 `31.3313 MB`。该结果对应 `results/compact_storage_comparison.csv` 与 `results/compact_storage_comparison_folds.csv`。

这些数值也已经保存到 `results/experiment_results.csv`。报告中可以直接引用下面这些指标：

- `fold_i_rmse`
  - 第 `i` 折上的验证误差。
- `mean_rmse`
  - `5` 折平均验证误差，越小越好，也是模型选择的主要依据。
- `std_rmse`
  - `5` 折结果波动大小，越小说明模型更稳定。
- `avg_train_seconds`
  - 单折平均训练耗时，用于比较训练效率。
- `avg_predict_seconds`
  - 单折平均预测耗时，用于比较推理效率。
- `avg_memory_mb`
  - 程序根据主要模型参数和核心数据结构估算的近似空间消耗，可用于课程报告中的空间复杂度实验说明。

如果需要在报告中解释实验设计，可以写成：

- 所有模型均只使用 `Train.txt` 进行 `5` 折交叉验证；
- 随机种子固定为 `2026`；
- 不使用 `Test.txt` 参与训练或调参；
- 当前基础实验中，平均验证集 `RMSE` 最低的模型是二模型 `Ensemble`；
- 扩展调优后，最终提交采用优化后三模型融合 `OptimizedEnsembleModel`；
- 当前 `results/final_result.txt` 已按 `python3 src/predict_final.py --model optimized_ensemble` 的逻辑生成，因此它对应的是在完整 `Train.txt` 上重新训练后的优化融合模型预测结果。
