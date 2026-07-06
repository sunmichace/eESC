import subprocess
import re
import shutil
import os
import time
from datetime import datetime

# ================= 配置区 =================
SCRIPT_NAME = "CNNPredictor.py"  # 你要运行的脚本名称
SOURCE_WEIGHT = "improved_best_model_fold3.pth"  # 原脚本生成的权重文件名
BEST_WEIGHT = "best.pth"  # 保存最佳权重的文件名
LOG_FILE = "training_history.log"  # 记录日志的文件名
MAX_RUNS = 50  # 最大循环执行次数

# 选择以哪个指标作为保存标准
# 如果以 RMSE 为准，则是越小越好；如果以 R2 为准，则是越大越好。
# 这里默认以 RMSE 为准 (越小越好)
OPTIMIZE_METRIC = "R2"


# ==========================================

def main():
    # 初始化最佳指标
    best_score = float('inf') if OPTIMIZE_METRIC == "RMSE" else float('-inf')

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始自动化执行，优化目标: {OPTIMIZE_METRIC}\n")
        f.write("=" * 60 + "\n")

    for run_idx in range(1, MAX_RUNS + 1):
        print(f"\n{'=' * 20} 第 {run_idx} 次运行开始 {'=' * 20}")

        # 启动子进程运行脚本，并将 stderr 合并到 stdout
        process = subprocess.Popen(
            ["python", SCRIPT_NAME],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        full_output = []
        # 实时读取并打印输出，这样你可以看到训练进度
        for line in process.stdout:
            print(line, end="")
            full_output.append(line)

        process.wait()  # 等待脚本执行完毕
        output_str = "".join(full_output)

        # 使用正则表达式提取 Avg RMSE 和 Avg R2
        # 匹配类似于 "Avg RMSE: 1.2053" 这样的字符串
        rmse_match = re.search(r"Avg RMSE:\s*([0-9.]+)", output_str)
        r2_match = re.search(r"Avg R2:\s*([0-9.]+)", output_str)

        if rmse_match and r2_match:
            current_rmse = float(rmse_match.group(1))
            current_r2 = float(r2_match.group(1))

            # 根据配置决定当前分数
            current_score = current_rmse if OPTIMIZE_METRIC == "RMSE" else current_r2

            # 判断是否是更好的结果
            is_better = (current_score < best_score) if OPTIMIZE_METRIC == "RMSE" else (current_score > best_score)

            # 准备日志信息
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_line = f"[{timestamp}] 第 {run_idx} 次运行结束 | RMSE: {current_rmse:.4f}, R2: {current_r2:.4f}"

            if is_better:
                print(f"\n>>> 发现新突破！{OPTIMIZE_METRIC} 从 {best_score} 优化到了 {current_score} <<<")
                best_score = current_score
                log_line += f"  *** 新纪录 ({OPTIMIZE_METRIC}: {best_score}) ***"

                # 覆盖保存最佳权重
                if os.path.exists(SOURCE_WEIGHT):
                    shutil.copy(SOURCE_WEIGHT, BEST_WEIGHT)
                    print(f">>> 已将 {SOURCE_WEIGHT} 另存为 {BEST_WEIGHT} <<<")
                else:
                    print(f"!!! 警告: 未找到 {SOURCE_WEIGHT}，无法保存最佳权重 !!!")
            else:
                print(f"\n--- 本次未打破记录。当前最佳 {OPTIMIZE_METRIC}: {best_score} ---")

            # 写入日志文件
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")

        else:
            print(f"\n!!! 第 {run_idx} 次运行未能匹配到指标输出，脚本可能报错或强行中止了 !!!")
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 第 {run_idx} 次运行失败或未输出指标。\n")

        print(f"{'=' * 20} 第 {run_idx} 次运行结束 {'=' * 20}\n")

        # 稍微暂停几秒，防止连续启动过快导致显存未释放干净
        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，已手动中止自动化运行脚本。")
