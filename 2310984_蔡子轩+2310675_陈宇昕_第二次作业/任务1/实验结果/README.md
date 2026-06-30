# 任务1实验结果说明

本目录保存任务 1 推荐系统实验过程中产生的主要结果文件。

| 文件名 | 说明 |
| --- | --- |
| `final_result.txt` | 最终提交预测结果，格式与 `ResultForm.txt` 一致，对 `Test.txt` 中 9982 个用户-物品对给出预测评分。 |
| `final_model_metrics.csv` | 最终模型及主要对比模型的评估指标汇总。 |
| `experiment_results.csv` | 主实验对比结果，用于比较不同推荐算法的整体表现。 |
| `compact_storage_comparison.csv` | 稀疏/紧凑存储方案的实验对比结果。 |
| `compact_storage_comparison_folds.csv` | 紧凑存储方案在交叉验证各折上的结果。 |
| `cv_fold_count_results.csv` | 不同交叉验证折数设置下的实验结果。 |
| `ensemble_weight_fine_results.csv` | 集成模型权重细调实验结果。 |
| `exploration_best_blend_fine.csv` | 融合权重进一步搜索得到的较优组合记录。 |
| `itemcf_parameter_results.csv` | ItemCF 参数实验结果。 |
| `mf_residual_narrow_cv_results.csv` | 矩阵分解残差模型在窄范围参数搜索下的交叉验证结果。 |

其中，`final_result.txt` 是评分提交所需的最终结果文件，其余 CSV 文件用于支撑实验报告中的参数选择、模型比较和结果分析。
