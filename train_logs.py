import os
from datetime import datetime

def train_log(log_name, dataset_info, start_time, end_time, model_names, results_df):
    """生成训练日志 .md 文件

    参数：
        log_name:  日志文件名（不含路径和扩展名）
        dataset_info:  dict，训练数据信息 {"特征数": 43, "总样本": 7315, ...}
        start_time:   datetime，训练开始时间
        end_time:     datetime，训练结束时间
        results_df:   DataFrame，模型性能对比表
    """
    duration = end_time - start_time
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")

    # 确保日志目录存在
    root_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(root_dir, "train_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{log_name}.md")

    lines = []

    # 第一部分：训练基本信息
    lines.append("# 训练日志\n")
    lines.append("---\n")
    lines.append("## 一、训练基本信息\n")
    lines.append(f"- **训练开始时间**：{start_str}")
    lines.append(f"- **训练结束时间**：{end_str}")
    lines.append(f"- **总训练时长**：{duration}")
    if dataset_info:
        lines.append("")
        lines.append("### 训练数据\n")
        for key in ["特征数", "总样本", "标签0(降雨诱发滑坡)", "标签1(其他/非滑坡)", "训练集形状", "测试集形状"]:
            if key in dataset_info:
                lines.append(f"- **{key}**：{dataset_info[key]}")

    lines.append("")

    # 第二部分：模型性能对比
    lines.append("---\n")
    lines.append("## 二、模型性能对比\n")

    if results_df is not None and len(results_df) > 0:
        # 按 AUC-PR 降序排列
        if "F1值" in results_df.columns:
            df_sorted = results_df.sort_values("F1值", ascending=False).reset_index(drop=True)
        else:
            df_sorted = results_df

        # 排名列
        if "排名" not in df_sorted.columns:
            df_sorted.insert(0, "排名", range(1, len(df_sorted) + 1))

        lines.append(f"共 {len(df_sorted)} 个模型，按 F1值 降序排列\n")
        # 选取核心展示列（按参考格式美化列名）
        col_map = {
            "排名": "排名",
            "模型名称": "模型名称",
            "准确率": "准确率",
            "精确率": "精确率",
            "召回率": "召回率",
            "F1值": "F1 值",
            "特异性": "特异性",
            "AUC-ROC": "AUC-ROC",
            "AUC-PR": "AUC-PR",
            "最优阈值": "最优阈值",
        }
        show_cols = [c for c in col_map.keys() if c in df_sorted.columns]
        display_names = [col_map[c] for c in show_cols]

        # 生成 Markdown 表格
        lines.append(f"| {' | '.join(display_names)} |")
        lines.append(f"|{'|'.join([':---:' if c != '模型名称' else ':---' for c in show_cols])}|")

        for _, row in df_sorted.iterrows():
            vals = []
            for c in show_cols:
                v = row[c]
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append(f"| {' | '.join(vals)} |")

        lines.append("")

        # 最佳模型
        best = df_sorted.iloc[0]
        lines.append(f"**最佳模型**: {best.get('模型名称', 'N/A')}  "
                     f"(F1={best.get('F1值', 0):.4f},"
                     f"AUC-ROC={best.get('AUC-ROC', 0):.4f},"
                     f"AUC-PR={best.get('AUC-PR', 0):.4f}, "
                     f"阈值={best.get('最优阈值', 0):.3f})\n")
    else:
        lines.append("> 模型性能数据为空\n")

    # 第三部分：模型详细参数
    lines.append("---\n")
    lines.append("## 三、模型列表\n")
    lines.append(f"共 {len(model_names)} 个模型参与集成：\n")
    for i, name in enumerate(model_names, 1):
        lines.append(f"- {i}.`{name}`")
    lines.append("")

    # 写入文件
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n训练日志已保存到: {log_path}", flush=True)

if __name__ == "__main__":
    dataset_info = {}
    start_time = datetime.now()
    end_time = datetime.now()
    model_names = []
    results_df = []
    log_name = f"train_log_test"
    train_log(log_name, dataset_info, start_time, end_time, model_names, results_df)