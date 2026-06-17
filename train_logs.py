"""
V2 训练日志生成器

用法：
    from train_logs import generate_train_log

    generate_train_log(
        log_name="train_log_20260617_1430",
        config=config_dict,           # 训练配置参数
        start_time=start_dt,
        end_time=end_dt,
        results_df=results_df,        # 模型性能排名DataFrame
        all_metrics=all_metrics_list, # 原始评估指标列表
        shap_info=shap_dict,          # SHAP分析结果
        calibrated_models=calib_list, # 参与校准的模型名列表
    )
"""
import os
from datetime import datetime


def generate_train_log(log_name, config=None, start_time=None, end_time=None,
                       results_df=None, all_metrics=None, shap_info=None,
                       calibrated_models=None):
    """生成 V2 训练日志 Markdown 文件

    参数:
        log_name:          日志文件名（不含路径和扩展名）
        config:            dict，训练配置参数（neg_ratio, optuna_trials, use_gpu等）
        start_time:        datetime，训练开始时间
        end_time:          datetime，训练结束时间
        results_df:        DataFrame，模型性能排名（含 排名/模型名称/F1/AUC等）
        all_metrics:       list[dict]，各模型原始评估指标
        shap_info:         dict，SHAP 分析结果 {"top5": [...], "importance_file": "..."}
        calibrated_models: list，参与概率校准的模型名列表
    """
    if start_time is None:
        start_time = datetime.now()
    if end_time is None:
        end_time = datetime.now()

    duration = end_time - start_time
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    date_str = start_time.strftime("%Y%m%d")

    # 确保日志目录存在
    root_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root_dir, "train_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{log_name}.md")

    lines = []

    # 一、训练基本信息
    lines.append("# 训练日志\n")
    lines.append("---\n")
    lines.append("## 一、训练基本信息\n")
    lines.append(f"- **Pipeline**: V2（易发性·暴露度二阶段分离）")
    lines.append(f"- **训练开始时间**：{start_str}")
    lines.append(f"- **训练结束时间**：{end_str}")
    lines.append(f"- **总训练时长**：{duration}")

    # 训练配置
    if config:
        lines.append("")
        lines.append("### 训练配置\n")
        config_items = {
            'neg_ratio': '负样本比例',
            'optuna_trials': 'Optuna调参次数',
            'use_gpu': 'GPU加速',
            'buffer_dist': '缓冲区距离(km)',
            'hybrid_ratio': '混合负样本真实数据占比',
            'quality_check': 'RF质量检验',
            'run_shap': 'SHAP分析',
        }
        for key, label in config_items.items():
            if key in config:
                val = config[key]
                if isinstance(val, bool):
                    val = '开启' if val else '关闭'
                lines.append(f"- **{label}**：{val}")
        lines.append(f"- **日志目录**：`{log_dir}`")
        lines.append(f"- **模型目录**：`models_v2/`")
        lines.append(f"- **结果目录**：`results_v2/`")

    # 训练数据信息
    if config and 'n_pos' in config:
        lines.append("")
        lines.append("### 训练数据\n")
        lines.append(f"- **总样本**：{config.get('n_total', '?')} 条")
        lines.append(f"- **正样本(1)**：{config.get('n_pos', '?')} 条")
        lines.append(f"- **负样本(0)**：{config.get('n_neg', '?')} 条")
        lines.append(f"- **易发性特征**：{config.get('n_features', '?')} 个")
        lines.append(f"- **训练集形状**：{config.get('train_shape', '?')}")
        lines.append(f"- **测试集形状**：{config.get('test_shape', '?')}")

    lines.append("")

    # 二、模型性能对比
    lines.append("---\n")
    lines.append("## 二、模型性能对比\n")

    if results_df is not None and len(results_df) > 0:
        # 按 F1 值降序排列
        if 'F1值' in results_df.columns:
            df_sorted = results_df.sort_values('F1值', ascending=False).reset_index(drop=True)
        else:
            df_sorted = results_df

        # 确保有排名列
        if '排名' not in df_sorted.columns:
            df_sorted.insert(0, '排名', range(1, len(df_sorted) + 1))

        lines.append(f"共 {len(df_sorted)} 个模型，按 **F1值** 降序排列\n")

        # 选择展示列
        col_map = {
            '排名': '排名',
            '模型名称': '模型名称',
            '准确率': '准确率',
            '精确率': '精确率',
            '召回率': '召回率',
            'F1值': 'F1值',
            '特异性': '特异性',
            'AUC-ROC': 'AUC-ROC',
            'AUC-PR': 'AUC-PR',
            '最优阈值': '最优阈值',
        }
        show_cols = [c for c in col_map.keys() if c in df_sorted.columns]
        display_names = [col_map[c] for c in show_cols]

        # Markdown 表头
        header = '| ' + ' | '.join(display_names) + ' |'
        sep = '|' + '|'.join([':---:' if c != '模型名称' else ':---' for c in show_cols]) + '|'
        lines.append(header)
        lines.append(sep)

        # 表体
        for _, row in df_sorted.iterrows():
            vals = []
            for c in show_cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append('| ' + ' | '.join(vals) + ' |')

        lines.append("")

        # 最佳模型
        best = df_sorted.iloc[0]
        best_name = best.get('模型名称', 'N/A')
        best_f1 = best.get('F1值', 0)
        best_auc = best.get('AUC-ROC', 0)
        best_pr = best.get('AUC-PR', 0)
        best_th = best.get('最优阈值', '—')
        lines.append(f"**🏆 最佳模型**: `{best_name}`  "
                     f"(F1={best_f1:.4f}, AUC-ROC={best_auc:.4f}, "
                     f"AUC-PR={best_pr:.4f}, 阈值={best_th})\n")
    else:
        lines.append("> 模型性能数据为空\n")

    lines.append("")

    # 三、各模型详细指标
    lines.append("---\n")
    lines.append("## 三、各模型详细指标\n")

    if all_metrics and len(all_metrics) > 0:
        for m in all_metrics:
            name = m.get('模型名称', '?')
            f1 = m.get('F1值', 0)
            auc = m.get('AUC-ROC', 0)
            pr = m.get('AUC-PR', 0)
            rec = m.get('召回率', 0)
            prec = m.get('精确率', 0)
            spec = m.get('特异性', 0)
            acc = m.get('准确率', 0)
            th = m.get('最优阈值', '—')

            lines.append(f"- **{name}**: F1={f1:.4f}, AUC={auc:.4f}, PR={pr:.4f}, "
                         f"召回率={rec:.4f}, 精确率={prec:.4f}, 特异性={spec:.4f}, "
                         f"准确率={acc:.4f}, 阈值={th}")
    else:
        lines.append("> 无详细指标数据\n")

    lines.append("")

    # 四、SHAP 分析摘要
    if shap_info:
        lines.append("---\n")
        lines.append("## 四、SHAP 特征重要性分析\n")

        if 'best_model' in shap_info:
            lines.append(f"- **分析模型**: {shap_info['best_model']}")

        if 'top5' in shap_info and len(shap_info['top5']) > 0:
            lines.append("")
            lines.append("**Top 5 重要特征**:\n")
            for i, feat in enumerate(shap_info['top5'], 1):
                lines.append(f"  {i}. `{feat}`")

        if 'importance_file' in shap_info:
            lines.append(f"\n- **完整SHAP值**: `{shap_info['importance_file']}`")

        lines.append("")

    # 五、参与校准的模型
    if calibrated_models:
        lines.append("---\n")
        lines.append("## 五、概率校准\n")
        lines.append(f"共 {len(calibrated_models)} 个模型参与 CalibratedClassifierCV(sigmoid) 校准：\n")
        for i, name in enumerate(calibrated_models, 1):
            lines.append(f"- {i}. `{name}`")
        lines.append("")

    # 写入文件
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n训练日志已保存: {log_path}")


if __name__ == "__main__":
    # 简单测试
    import pandas as pd
    import numpy as np

    dummy_config = {
        'neg_ratio': 1.0,
        'optuna_trials': 30,
        'use_gpu': False,
        'buffer_dist': 0.8,
        'hybrid_ratio': 0.3,
        'quality_check': True,
        'run_shap': True,
        'n_total': 10644,
        'n_pos': 5322,
        'n_neg': 5322,
        'n_features': 27,
        'train_shape': '(8515, 27)',
        'test_shape': '(2129, 27)',
    }

    dummy_results = pd.DataFrame({
        '模型名称': ['xgboost', 'random_forest', 'lightgbm'],
        'F1值': [0.9521, 0.9487, 0.9455],
        'AUC-ROC': [0.9876, 0.9843, 0.9821],
        'AUC-PR': [0.9765, 0.9721, 0.9689],
        '召回率': [0.9412, 0.9356, 0.9288],
        '精确率': [0.9633, 0.9621, 0.9678],
        '特异性': [0.9721, 0.9688, 0.9745],
        '准确率': [0.9567, 0.9523, 0.9512],
        '最优阈值': [0.48, 0.52, 0.46],
    })
    dummy_results.insert(0, '排名', range(1, len(dummy_results) + 1))

    dummy_shap = {
        'best_model': 'xgboost',
        'top5': ['rain_30d', 'elevation', 'ndvi', 'api', 'tpi'],
        'importance_file': 'results_v2/shap_importance.csv',
    }

    dummy_calib = ['xgboost', 'lightgbm', 'random_forest', 'catboost']

    generate_train_log(
        log_name=f"train_log_test_v2",
        config=dummy_config,
        start_time=datetime.now(),
        end_time=datetime.now(),
        results_df=dummy_results,
        all_metrics=dummy_results.to_dict('records'),
        shap_info=dummy_shap,
        calibrated_models=dummy_calib,
    )
    print("测试日志已生成，请查看 train_logs/ 目录")
