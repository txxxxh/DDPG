# data_loader.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from config import config

def preprocess_features(df):
    """
    对特征进行预处理，处理数量级差异问题

    Args:
        df: 原始数据DataFrame

    Returns:
        processed_df: 预处理后的DataFrame
    """
    df_processed = df.copy()

    # 确保所有数值列都是float类型，避免类型不匹配警告
    for col in df_processed.select_dtypes(include=[np.number]).columns:
        df_processed[col] = df_processed[col].astype('float64')

    # 定义需要取对数的大数值特征（基于列名关键词）
    log_transform_cols = []

    # 检查每列的数量级并决定预处理方式
    for col_idx, col_name in enumerate(df.columns):
        if col_idx >= len(df.columns):
            break

        col_data = df.iloc[:, col_idx]

        # 跳过非数值列（如日期时间列）
        if col_data.dtype.kind in ['M', 'O', 'S', 'U']:  # datetime, object, string, unicode
            print(f"跳过非数值列: {col_name} (类型: {col_data.dtype})")
            continue

        col_data_clean = col_data.replace([np.inf, -np.inf], np.nan).dropna()

        if len(col_data_clean) == 0:
            continue

        # 根据列名和数值特征判断预处理方式
        col_name_lower = str(col_name).lower()

        # 价格相关：取对数
        if any(keyword in col_name_lower for keyword in ['价', 'price', 'pr_']):
            if col_data_clean.min() > 0:  # 确保为正值
                transformed_data = np.log1p(col_data).astype('float64')
                df_processed.iloc[:, col_idx] = transformed_data
                log_transform_cols.append(col_name)

        # 市值相关：取对数
        elif any(keyword in col_name_lower for keyword in ['市值', 'tmv', 'market']):
            if col_data_clean.min() > 0:
                transformed_data = np.log1p(col_data).astype('float64')
                df_processed.iloc[:, col_idx] = transformed_data
                log_transform_cols.append(col_name)

        # 成交量相关：取对数
        elif any(keyword in col_name_lower for keyword in ['成交量', 'volume', 'vol']):
            if col_data_clean.min() >= 0:
                transformed_data = np.log1p(col_data).astype('float64')
                df_processed.iloc[:, col_idx] = transformed_data
                log_transform_cols.append(col_name)

        # 大比率值：使用tanh压缩
        elif any(keyword in col_name_lower for keyword in ['市盈率', 'pe', '市净率', 'pb', '市销率', 'ps']):
            # 使用tanh将比率压缩到[-1,1]区间
            transformed_data = np.tanh(col_data / 100.0).astype('float64')
            df_processed.iloc[:, col_idx] = transformed_data

        # 收益率相关：保持原样或轻微缩放
        elif any(keyword in col_name_lower for keyword in ['收益', 'return', 'ret']):
            # 收益率通常已经在合理范围内，只需处理异常值
            transformed_data = np.clip(col_data, -1.0, 1.0).astype('float64')
            df_processed.iloc[:, col_idx] = transformed_data

        # 其他数值：根据数量级决定
        else:
            col_max = col_data_clean.abs().max()  # 修改：使用pandas的abs()方法
            col_std = col_data_clean.std()

            # 如果数值很大（>1000）且方差也大，进行对数变换
            if col_max > 1000 and col_std > 100:
                if col_data_clean.min() >= 0:
                    transformed_data = np.log1p(col_data).astype('float64')
                    df_processed.iloc[:, col_idx] = transformed_data
                    log_transform_cols.append(col_name)
                else:
                    # 包含负值时使用asinh变换
                    transformed_data = np.arcsinh(col_data / col_std).astype('float64')
                    df_processed.iloc[:, col_idx] = transformed_data

            # 如果数值适中但方差很大，进行标准化
            elif col_std > 0:
                transformed_data = ((col_data - col_data.mean()) / col_std).astype('float64')
                df_processed.iloc[:, col_idx] = transformed_data

    print(f"预处理完成，对数变换的特征: {len(log_transform_cols)}个")
    if log_transform_cols:
        print(f"对数变换特征: {log_transform_cols[:5]}...")  # 只显示前5个

    return df_processed


def robust_standardize_factors(factors_array):
    """
    对因子数据进行稳健标准化

    Args:
        factors_array: (n_steps, n_assets, n_factors)

    Returns:
        standardized_factors: 标准化后的因子数据
    """
    print("进行因子数据稳健标准化...")

    standardized = np.copy(factors_array)
    n_steps, n_assets, n_factors = factors_array.shape

    # 对每个因子进行标准化
    for factor_idx in range(n_factors):
        factor_data = factors_array[:, :, factor_idx].flatten()

        # 使用中位数和MAD进行稳健统计
        median = np.median(factor_data)
        mad = np.median(np.abs(factor_data - median))  # 中位数绝对偏差

        if mad > 0:
            # 稳健标准化：(x - median) / (1.4826 * MAD)
            # 1.4826是使MAD等于正态分布标准差的校正因子
            standardized[:, :, factor_idx] = (factors_array[:, :, factor_idx] - median) / (1.4826 * mad)
        else:
            # 如果MAD为0，使用传统标准化
            std = np.std(factor_data)
            if std > 0:
                standardized[:, :, factor_idx] = (factors_array[:, :, factor_idx] - np.mean(factor_data)) / std

        # 限制极值在[-5, 5]范围内
        standardized[:, :, factor_idx] = np.clip(standardized[:, :, factor_idx], -5, 5)

    print("稳健标准化完成")
    return standardized


# ====================== 高效因子计算函数 ======================

def calculate_momentum_factor(returns_data, lookback_period=20):
    """
    计算简单动量因子 - 过去N天的累积收益率

    Args:
        returns_data: 收益率数据 (n_steps, n_assets)
        lookback_period: 回望期，默认20天

    Returns:
        momentum: 动量因子 (n_steps, n_assets)
    """
    # 数据验证
    if returns_data is None:
        raise ValueError("收益率数据不能为空")

    if len(returns_data.shape) != 2:
        raise ValueError(f"收益率数据应该是2维数组，实际形状: {returns_data.shape}")

    n_steps, n_assets = returns_data.shape
    print(f"计算动量因子: 时间步数={n_steps}, 资产数={n_assets}, 回望期={lookback_period}")

    momentum = np.zeros((n_steps, n_assets))

    # 处理异常值
    clean_returns = np.nan_to_num(returns_data, nan=0.0, posinf=0.1, neginf=-0.1)
    clean_returns = np.clip(clean_returns, -0.5, 0.5)

    # 确保lookback_period不超过时间步数
    effective_lookback = min(lookback_period, n_steps - 1)

    # 计算滚动累积收益率
    for i in range(effective_lookback, n_steps):
        try:
            momentum[i, :] = np.sum(clean_returns[i - effective_lookback:i, :], axis=0)
        except Exception as e:
            print(f"计算动量因子时出错，时间步{i}: {e}")
            momentum[i, :] = 0.0

    # 标准化到[-1, 1]区间
    momentum = np.tanh(momentum)

    print(f"动量因子计算完成，统计: min={momentum.min():.6f}, max={momentum.max():.6f}")
    return momentum


def calculate_volatility_factor(returns_data, lookback_period=20):
    """
    计算波动率因子 - 过去N天收益率的标准差

    Args:
        returns_data: 收益率数据 (n_steps, n_assets)
        lookback_period: 回望期，默认20天

    Returns:
        volatility: 波动率因子 (n_steps, n_assets)
    """
    n_steps, n_assets = returns_data.shape
    volatility = np.zeros((n_steps, n_assets))

    # 处理异常值
    clean_returns = np.nan_to_num(returns_data, nan=0.0, posinf=0.1, neginf=-0.1)
    clean_returns = np.clip(clean_returns, -0.5, 0.5)

    # 计算滚动标准差
    for i in range(lookback_period, n_steps):
        volatility[i, :] = np.std(clean_returns[i - lookback_period:i, :], axis=0)

    # 标准化处理
    volatility = np.tanh(volatility * 10)  # 放大后用tanh压缩

    return volatility


def calculate_trend_factor(returns_data, short_period=5, long_period=20):
    """
    计算趋势因子 - 短期和长期移动平均的差值

    Args:
        returns_data: 收益率数据 (n_steps, n_assets)
        short_period: 短期周期，默认5天
        long_period: 长期周期，默认20天

    Returns:
        trend: 趋势因子 (n_steps, n_assets)
    """
    n_steps, n_assets = returns_data.shape
    trend = np.zeros((n_steps, n_assets))

    # 处理异常值
    clean_returns = np.nan_to_num(returns_data, nan=0.0, posinf=0.1, neginf=-0.1)
    clean_returns = np.clip(clean_returns, -0.5, 0.5)

    # 计算累积价格（用于移动平均）
    cumulative_returns = np.cumsum(clean_returns, axis=0)

    # 计算移动平均差值
    for i in range(long_period, n_steps):
        short_ma = np.mean(cumulative_returns[i - short_period:i, :], axis=0)
        long_ma = np.mean(cumulative_returns[i - long_period:i, :], axis=0)
        trend[i, :] = short_ma - long_ma

    # 标准化到[-1, 1]区间
    trend = np.tanh(trend)

    return trend


def calculate_mean_reversion_factor(returns_data, lookback_period=10):
    """
    计算均值回归因子 - 当前价格相对于移动平均的偏离度

    Args:
        returns_data: 收益率数据 (n_steps, n_assets)
        lookback_period: 回望期，默认10天

    Returns:
        mean_reversion: 均值回归因子 (n_steps, n_assets)
    """
    n_steps, n_assets = returns_data.shape
    mean_reversion = np.zeros((n_steps, n_assets))

    # 处理异常值
    clean_returns = np.nan_to_num(returns_data, nan=0.0, posinf=0.1, neginf=-0.1)
    clean_returns = np.clip(clean_returns, -0.5, 0.5)

    # 计算累积价格
    cumulative_returns = np.cumsum(clean_returns, axis=0)

    # 计算相对于移动平均的偏离
    for i in range(lookback_period, n_steps):
        moving_avg = np.mean(cumulative_returns[i - lookback_period:i, :], axis=0)
        current_level = cumulative_returns[i, :]
        mean_reversion[i, :] = current_level - moving_avg

    # 标准化到[-1, 1]区间
    mean_reversion = np.tanh(mean_reversion)

    return mean_reversion


def calculate_cross_sectional_rank_factor(returns_data):
    """
    计算横截面排名因子 - 每个时点上各资产收益率的排名百分位

    Args:
        returns_data: 收益率数据 (n_steps, n_assets)

    Returns:
        rank_factor: 排名因子 (n_steps, n_assets)
    """
    n_steps, n_assets = returns_data.shape
    rank_factor = np.zeros((n_steps, n_assets))

    # 处理异常值
    clean_returns = np.nan_to_num(returns_data, nan=0.0, posinf=0.1, neginf=-0.1)
    clean_returns = np.clip(clean_returns, -0.5, 0.5)

    # 计算每个时点的排名百分位
    for i in range(n_steps):
        # 使用scipy.stats.rankdata计算排名
        ranks = np.argsort(np.argsort(clean_returns[i, :]))  # 双重argsort得到排名
        rank_factor[i, :] = (ranks / (n_assets - 1)) * 2 - 1  # 标准化到[-1, 1]

    return rank_factor


# ====================== 修改后的数据加载函数 ======================
def load_fund_data_from_excel(excel_path, n_assets=None, n_factors=24, return_col_index=22,
                              factor_type='momentum', asset_codes=None, random_seed=None,
                              selection_mode='sequential'):
    """
    从Excel工作簿加载基金数据（支持灵活的资产选择）

    Args:
        excel_path: Excel文件路径
        n_assets: 资产数量，如果为None则自动使用所有可用工作表
        n_factors: 因子数量，默认24
        return_col_index: 收益率列索引，默认22
        factor_type: 第24个因子类型，可选：
                    - 'momentum': 动量因子（默认）
                    - 'volatility': 波动率因子
                    - 'trend': 趋势因子
                    - 'mean_reversion': 均值回归因子
                    - 'rank': 横截面排名因子
        asset_codes: 指定的股票代码列表，如果提供则必须与n_assets数量匹配
        random_seed: 随机种子，确保随机选择结果可重现
        selection_mode: 资产选择模式
                       - 'sequential': 按顺序选择（默认，保持原有行为）
                       - 'random': 随机选择
                       - 'specified': 根据asset_codes指定选择

    Returns:
        returns: (n_steps, n_assets) 收益率数据
        factors: (n_steps, n_assets, n_factors) 因子数据
        initial_prices: 初始价格数组
        actual_n_assets: 实际加载的资产数量
        selected_assets: 实际选择的资产名称列表
    """
    import random

    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    # 读取Excel文件
    try:
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        available_sheets = len(sheet_names)
        print(f"Excel文件中找到 {available_sheets} 个工作表: {sheet_names[:5]}{'...' if len(sheet_names) > 5 else ''}")
    except Exception as e:
        raise ValueError(f"无法读取Excel文件: {e}")

    # 根据选择模式确定要加载的工作表
    if selection_mode == 'specified' or asset_codes is not None:
        # 指定资产代码模式
        if asset_codes is None:
            raise ValueError("指定模式下必须提供asset_codes参数")

        if n_assets is not None and len(asset_codes) != n_assets:
            raise ValueError(f"指定的股票代码数量({len(asset_codes)})与请求的资产数量({n_assets})不匹配")

        selected_sheets = []
        for code in asset_codes:
            # 查找匹配的工作表（支持模糊匹配）
            matching_sheets = [sheet for sheet in sheet_names if code in sheet]
            if not matching_sheets:
                # 尝试更宽松的匹配
                matching_sheets = [sheet for sheet in sheet_names if
                                   code.upper() in sheet.upper() or sheet.upper() in code.upper()]

            if not matching_sheets:
                print(f"警告: 未找到股票代码 {code} 对应的工作表，跳过")
                continue
            elif len(matching_sheets) > 1:
                print(
                    f"警告: 股票代码 {code} 匹配到多个工作表: {matching_sheets[:3]}{'...' if len(matching_sheets) > 3 else ''}")
                print(f"将使用第一个: {matching_sheets[0]}")

            selected_sheets.append(matching_sheets[0])

        actual_n_assets = len(selected_sheets)
        print(f"指定模式: 成功匹配 {actual_n_assets} 个资产")

    else:
        # 确定实际要加载的资产数量
        if n_assets is None:
            actual_n_assets = available_sheets
            print(f"未指定资产数量，将加载所有 {actual_n_assets} 个资产")
        else:
            if n_assets > available_sheets:
                print(f"警告: 请求的资产数量 {n_assets} 超过可用工作表数量 {available_sheets}")
                print(f"将加载所有可用的 {available_sheets} 个资产")
                actual_n_assets = available_sheets
            else:
                actual_n_assets = n_assets
                print(f"将加载 {actual_n_assets} 个资产")

        # 根据选择模式选择工作表
        if selection_mode == 'random':
            selected_sheets = random.sample(sheet_names, actual_n_assets)
            print(f"随机选择模式: {selected_sheets}")
        else:  # sequential (默认)
            selected_sheets = sheet_names[:actual_n_assets]
            print(f"顺序选择模式: 将加载前 {actual_n_assets} 个资产")

    # 存储数据
    all_returns = []
    all_factors_raw = []
    initial_prices = []
    loaded_assets = []

    print(f"开始加载 {len(selected_sheets)} 个资产的数据...")

    for i, sheet_name in enumerate(selected_sheets):
        try:
            print(f"正在加载资产 {i + 1}/{len(selected_sheets)}: {sheet_name}")

            # 读取工作表数据
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            # 处理日期列
            date_cols = df.select_dtypes(include=['datetime64']).columns
            if len(date_cols) > 0:
                for date_col in date_cols:
                    if date_col in df.columns:
                        df = df.drop(columns=[date_col])

            # 只保留数值列
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_cols]

            # 验证数据列数
            if df.shape[1] < 23:
                print(f"警告: 工作表 '{sheet_name}' 数据列数不足 ({df.shape[1]} < 23)，跳过该资产")
                continue

            # 数据预处理
            df_clean = df.copy()
            df_clean = df_clean.ffill().fillna(0)
            df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(0)

            # 确保数值类型
            for col in df_clean.columns:
                if not pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

            # 预处理特征
            df_processed = preprocess_features(df_clean)

            # 确保return_col_index在有效范围内
            if return_col_index >= df_processed.shape[1]:
                effective_return_col = df_processed.shape[1] - 1
                print(f"警告: 指定的收益率列索引 {return_col_index} 超出范围，使用列索引 {effective_return_col}")
            else:
                effective_return_col = return_col_index

            # 提取收益率
            returns = df_processed.iloc[:, effective_return_col].values.reshape(-1, 1)
            returns = np.nan_to_num(returns, nan=0.0, posinf=0.1, neginf=-0.1)
            returns = np.clip(returns, -0.9, 2.0)

            # 提取23个因子
            factors_raw = df_processed.iloc[:, :23].values
            factors_raw = np.nan_to_num(factors_raw, nan=0.0, posinf=1.0, neginf=-1.0)

            # 获取初始价格
            initial_price = abs(df_processed.iloc[0, 1]) if df_processed.shape[1] > 1 and not np.isnan(
                df_processed.iloc[0, 1]) else 1.0

            # 存储数据
            all_returns.append(returns)
            all_factors_raw.append(factors_raw)
            initial_prices.append(initial_price)
            loaded_assets.append(sheet_name)

            print(f"✓ 成功加载资产: {sheet_name} (数据行数: {df_processed.shape[0]})")

        except Exception as e:
            print(f"✗ 处理工作表 {sheet_name} 时出错: {e}")
            print(f"跳过该资产，继续处理下一个...")
            continue

    # 检查是否成功加载了数据
    if len(all_returns) == 0:
        raise ValueError("未能成功加载任何资产数据")

    # 更新实际加载的资产数量
    actual_loaded_assets = len(all_returns)
    if actual_loaded_assets != len(selected_sheets):
        print(f"警告: 计划加载 {len(selected_sheets)} 个资产，实际成功加载 {actual_loaded_assets} 个资产")

    print(f"成功加载的资产: {loaded_assets}")

    # 组合数据
    returns_array = np.hstack(all_returns)  # (n_steps, actual_loaded_assets)
    factors_raw_array = np.stack(all_factors_raw, axis=1)  # (n_steps, actual_loaded_assets, 23)

    # 验证数据
    print(f"收益率数组形状: {returns_array.shape}")
    print(f"原始因子数组形状: {factors_raw_array.shape}")

    if returns_array is None or factors_raw_array is None:
        raise ValueError("数据组合失败：收益率或因子数据为空")

    if returns_array.shape[0] == 0 or returns_array.shape[1] == 0:
        raise ValueError(f"收益率数据形状异常: {returns_array.shape}")

    print(f"计算第24个因子: {factor_type}...")

    # 根据选择计算第24个因子
    try:
        if factor_type == 'momentum':
            additional_factor = calculate_momentum_factor(returns_array)
        elif factor_type == 'volatility':
            additional_factor = calculate_volatility_factor(returns_array)
        elif factor_type == 'trend':
            additional_factor = calculate_trend_factor(returns_array)
        elif factor_type == 'mean_reversion':
            additional_factor = calculate_mean_reversion_factor(returns_array)
        elif factor_type == 'rank':
            additional_factor = calculate_cross_sectional_rank_factor(returns_array)
        else:
            print(f"未知因子类型 {factor_type}，使用默认动量因子")
            additional_factor = calculate_momentum_factor(returns_array)

        if additional_factor is None:
            raise ValueError(f"第24个因子({factor_type})计算结果为空")

        print(f"第24个因子计算成功，形状: {additional_factor.shape}")

    except Exception as e:
        print(f"计算第24个因子时出错: {e}")
        print("使用零值填充第24个因子")
        additional_factor = np.zeros((returns_array.shape[0], returns_array.shape[1]))

    # 组合所有因子
    factors_array = np.zeros((returns_array.shape[0], actual_loaded_assets, n_factors))
    factors_array[:, :, :23] = factors_raw_array  # 前23个因子
    factors_array[:, :, 23] = additional_factor  # 第24个因子

    # 对因子数据进行稳健标准化
    factors_array = robust_standardize_factors(factors_array)

    print(f"\n=== 数据加载完成 ===")
    print(f"  收益率形状: {returns_array.shape}")
    print(f"  因子形状: {factors_array.shape}")
    print(f"  实际资产数量: {actual_loaded_assets}")
    print(f"  选择模式: {selection_mode}")
    print(f"  因子数量: {n_factors} (23个现成因子 + 1个{factor_type}因子)")
    print(f"  第24个因子统计: min={additional_factor.min():.6f}, max={additional_factor.max():.6f}")

    # 验证数据形状的正确性
    assert factors_array.shape == (returns_array.shape[0], actual_loaded_assets, n_factors), \
        f"因子数据形状错误: 期望({returns_array.shape[0]}, {actual_loaded_assets}, {n_factors}), 实际{factors_array.shape}"

    print(f"  数据验证通过: 时间步数={returns_array.shape[0]}, 资产数={actual_loaded_assets}, 因子数={n_factors}")

    return returns_array, factors_array, np.array(initial_prices), actual_loaded_assets, loaded_assets




def load_data_with_config(excel_path, desired_n_assets=None, factor_type='momentum',
                         asset_codes=None, random_seed=None, selection_mode='sequential'):
    """
    根据配置加载数据并更新全局参数（增强版）

    Args:
        excel_path: Excel文件路径
        desired_n_assets: 期望的资产数量，如果为None则使用全局配置
        factor_type: 因子类型
        asset_codes: 指定的股票代码列表
        random_seed: 随机种子
        selection_mode: 选择模式 ('sequential', 'random', 'specified')

    Returns:
        返回加载的数据和实际资产数量
    """
    # 确定要加载的资产数量
    if desired_n_assets is None:
        target_n_assets = config.n_assets
    else:
        target_n_assets = desired_n_assets

    # 如果提供了asset_codes，自动切换到指定模式
    if asset_codes is not None:
        selection_mode = 'specified'
        target_n_assets = len(asset_codes)

    print(f"目标加载资产数量: {target_n_assets}")
    print(f"选择模式: {selection_mode}")
    if asset_codes:
        print(f"指定资产代码: {asset_codes}")

    # 加载数据
    returns, factors, initial_prices, actual_n_assets, selected_assets = load_fund_data_from_excel(
        excel_path=excel_path,
        n_assets=target_n_assets,
        n_factors=config.n_factors,
        return_col_index=config.return_col_index,
        factor_type=factor_type,
        asset_codes=asset_codes,
        random_seed=random_seed,
        selection_mode=selection_mode
    )

    return returns, factors, initial_prices, actual_n_assets, selected_assets



# ====================== 使用示例 ======================
"""
使用方法（可以计算不同的第24个因子）：

# 使用动量因子（默认，最快）
returns, factors, prices = load_fund_data_from_excel_fast(
    'your_data.xlsx', 
    factor_type='momentum'
)

# 使用波动率因子
returns, factors, prices = load_fund_data_from_excel_fast(
    'your_data.xlsx', 
    factor_type='volatility'
)

# 使用横截面排名因子（推荐，包含相对信息）
returns, factors, prices = load_fund_data_from_excel_fast(
    'your_data.xlsx', 
    factor_type='rank'
)

各因子特点：
- momentum: 动量因子，计算最快，捕捉趋势
- volatility: 波动率因子，捕捉风险特征
- trend: 趋势因子，捕捉短长期趋势差异
- mean_reversion: 均值回归因子，捕捉反转信号
- rank: 横截面排名因子，最有效，包含相对强弱信息
"""


