import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')  # 忽略无关警告

def paired_data_statistical_test(csv_path):
    """
    读取无表头CSV中两列成对数据，计算均值差、t统计量、p值、95%置信区间
    
    参数:
        csv_path: CSV文件本地路径（如"E:/data.csv"）
    
    返回:
        统计结果字典
    """
    # ---------------------- 1. 读取无表头CSV并校验数据 ----------------------
    try:
        # 读取无表头CSV（header=None 表示第一行就是数据，不设为表头）
        df = pd.read_csv(csv_path, header=None)
        
        # 校验列数：必须只有2列数据
        if df.shape[1] != 2:
            raise ValueError(f"CSV文件需仅包含2列数据，当前读取到{df.shape[1]}列")
        
        # 重命名列（方便后续处理，不影响原始数据）
        df.columns = ["col1", "col2"]
        
        # 移除缺失值（成对数据缺失无意义）
        df = df.dropna()
        if len(df) < 2:
            raise ValueError("有效成对数据不足2组，无法进行t检验")
        
        print(f"✅ 数据读取成功：共{len(df)}组有效成对数据")
        print("前5组数据预览：")
        print(df.head(), "\n")
        
    except FileNotFoundError:
        print(f"❌ 错误：未找到文件，请检查路径是否正确：{csv_path}")
        return None
    except Exception as e:
        print(f"❌ 数据处理错误：{str(e)}")
        return None
    
    # ---------------------- 2. 计算核心统计指标 ----------------------
    col1 = df["col1"].values
    col2 = df["col2"].values
    
    # 计算成对差值（col1 - col2）
    differences = col1 - col2
    
    # 1. 均值差（Mean Difference）
    mean_diff = np.mean(differences)
    
    # 2. t统计量、p值（配对样本t检验，双侧检验，适配成对数据）
    t_stat, p_value = stats.ttest_rel(col1, col2)
    
    # 3. 95%置信区间（基于t分布，自由度=样本量-1）
    n = len(differences)
    df_degree = n - 1  # 自由度
    standard_error = stats.sem(differences)  # 标准误（均值的标准偏差）
    t_critical = stats.t.ppf(0.975, df_degree)  # 95%置信水平双侧临界t值
    ci_lower = mean_diff - t_critical * standard_error
    ci_upper = mean_diff + t_critical * standard_error
    
    # ---------------------- 3. 结果整理与可视化输出 ----------------------
    results = {
        "样本量": n,
        "均值差 (Mean Difference, col1-col2)": round(mean_diff, 6),
        "t统计量 (t-statistic)": round(t_stat, 6),
        "p值 (p-value, 双侧检验)": round(p_value, 6),
        "95%置信区间 (95% CI)": [round(ci_lower, 6), round(ci_upper, 6)]
    }
    
    # 打印结果（格式清晰，便于复制使用）
    print("="*60)
    print("📊 成对数据统计检验结果（配对样本t检验）")
    print("="*60)
    for key, value in results.items():
        print(f"{key:<35}: {value}")
    print("="*60)
    
    # 结果解读（通俗易理解，无需手动查表）
    print("\n🔍 结果解读：")
    alpha = 0.05  # 显著性水平
    if p_value < alpha:
        print(f"• p值 ({p_value:.6f}) < {alpha} → 拒绝原假设，两列数据总体均值存在显著差异")
    else:
        print(f"• p值 ({p_value:.6f}) ≥ {alpha} → 接受原假设，两列数据总体均值无显著差异")
    if ci_lower <= 0 <= ci_upper:
        print("• 95%置信区间包含0 → 进一步验证两列数据无显著差异")
    else:
        print("• 95%置信区间不包含0 → 进一步验证两列数据存在显著差异")
    
    return results

# ---------------------- 4. 运行函数（修改为你的CSV路径） ----------------------
if __name__ == "__main__":
    # 请将此处路径替换为你的本地data.csv文件路径（绝对路径或相对路径均可）
    CSV_FILE_PATH = "data.csv"  # 示例1：相对路径（CSV与代码在同一文件夹）
    # CSV_FILE_PATH = "E:/project/data.csv"  # 示例2：Windows绝对路径（用/或\\）
    # CSV_FILE_PATH = "/Users/xxx/project/data.csv"  # 示例3：Mac/Linux绝对路径
    
    # 执行统计检验
    paired_data_statistical_test(CSV_FILE_PATH)
