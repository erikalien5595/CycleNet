import numpy as np

class MultiTimeSeriesSimulator:
    """
    生成多元时间序列数据的模拟器，每个序列由趋势、两个季节性周期（24小时和168小时）
    以及 VAR(1) 模型产生的残差组成。
    """
    def __init__(self, T=30000, n_series=21, sigma=0.2, random_seed=42):
        """
        参数:
            T: 时间步长（默认30000）
            n_series: 序列数量（默认21）
            sigma: 噪声标准差（默认0.5）
            random_seed: 随机种子，保证结果可复现（默认42）
        """
        self.T = T
        self.n_series = n_series
        self.sigma = sigma
        np.random.seed(random_seed)

    def _seasonality_component(self, t, period, amplitude, phase):
        """生成周期性成分（正弦波）。"""
        return amplitude * np.sin(2 * np.pi * t / period + phase)

    def _traffic_seasonality_component(self, t, daily_amplitude, phase, weekday_factor=1.0, weekend_factor=0.5):
        """
        生成交通流量风格的季节性成分：
        - 基于24小时正弦波；
        - 根据 t 计算出对应的星期几，工作日采用较高振幅（weekday_factor），
          周末采用较低振幅（weekend_factor）。
        """
        # 计算对应的星期几：每天24小时，星期从0（周一）到6（周日）
        day_of_week = (t // 24) % 7
        amp_factor = np.where(day_of_week < 5, weekday_factor, weekend_factor)
        return daily_amplitude * amp_factor * np.sin(2 * np.pi * (t % 24) / 24 + phase)

    def _linear_trend(self, t, slope, intercept):
        """生成线性趋势。"""
        return slope * t + intercept

    def _sigmoid_trend(self, t, k, t0, amplitude, intercept):
        """生成 sigmoid 形状的趋势。"""
        return amplitude / (1 + np.exp(-k * (t - t0))) + intercept

    def _declining_trend(self, t, slope, intercept):
        """生成下降趋势（负斜率）。"""
        return -slope * t + intercept

    def _increase_then_decrease_trend(self, t, peak_t, max_value, intercept):
        """生成先增长后下降的趋势（分段线性趋势）。"""
        trend = np.zeros_like(t, dtype=float)
        idx_inc = t <= peak_t
        trend[idx_inc] = (max_value / peak_t) * t[idx_inc] + intercept
        idx_dec = t > peak_t
        trend[idx_dec] = max_value - (max_value / (self.T - peak_t)) * (t[idx_dec] - peak_t) + intercept
        return trend

    def _decrease_then_increase_trend(self, t, valley_t, min_value, intercept):
        """生成先下降后上升的趋势（分段线性趋势）。"""
        trend = np.zeros_like(t, dtype=float)
        idx_dec = t <= valley_t
        trend[idx_dec] = - (min_value / valley_t) * t[idx_dec] + intercept
        idx_inc = t > valley_t
        trend[idx_inc] = -min_value + (min_value / (self.T - valley_t)) * (t[idx_inc] - valley_t) + intercept
        return trend

    def _quadratic_max_trend(self, t):
        """
        生成全局类似二次函数的趋势：先上升后下降（抛物线开口向下）。
        公式：f(t) = -a*(t - h)^2 + amplitude，
        为了使得 f(0) ≈ f(T) ≈ 0，取 h = T/2，并令 a = amplitude/(h^2)。
        """
        h = self.T / 2
        amplitude = np.random.uniform(5, 10)
        a = amplitude / (h**2)
        return -a * (t - h)**2 + amplitude

    def _quadratic_min_trend(self, t):
        """
        生成全局类似二次函数的趋势：先下降后上升（抛物线开口向上）。
        公式：f(t) = a*(t - h)^2 - amplitude，
        同样取 h = T/2，并令 a = amplitude/(h^2)，使得 f(0) ≈ f(T) ≈ 0。
        """
        h = self.T / 2
        amplitude = np.random.uniform(5, 10)
        a = amplitude / (h**2)
        return a * (t - h)**2 - amplitude

    def _double_quadratic_trend(self, t):
        """
        生成一个分为两个阶段的趋势：
        第一阶段（t in [0, T1]）：使用凸型二次函数 f₁(t)，
            使得 f₁(0)=A₁, f₁(T1)=B₁, 且 A₁ > B₁；
        第二阶段（t in [T1, T]）：使用凸型二次函数 f₂(t)，
            使得 f₂(T1)=B₁, f₂(T)=B₂, 且 B₁ > B₂。
        通过拟合两个二次函数（形式为 f(t)=a*(t-h)²+c）实现，每个阶段选取适当的
        顶点 h 保证在区间内且满足 f(t)先下降后上升。
        """
        T_total = self.T
        T1 = T_total // 2  # 分段点

        # 第一阶段：[0, T1]
        t1 = np.arange(0, T1)
        A1 = np.random.uniform(5, 10)         # 第一阶段起始值
        drop1 = np.random.uniform(0.1, 0.5)         # 下降幅度
        B1 = A1 - drop1                       # 第一阶段结束值，满足 A1 > B1
        # 为保证凸型二次曲线（开口向上）且先降后升，顶点 h1 选在 (T1/2, T1)
        h1 = np.random.uniform(T1/2, T1)
        # 根据 f(0)=A1 与 f(T1)=B1 解得 a1 与 c1：
        # A1 = a1*(0 - h1)² + c1,   B1 = a1*(T1 - h1)² + c1
        a1 = (A1 - B1) / ((0 - h1)**2 - (T1 - h1)**2)
        c1 = A1 - a1*(0 - h1)**2
        trend1 = a1*(t1 - h1)**2 + c1

        # 第二阶段：[T1, T_total]
        t2 = np.arange(T1, T_total)
        A2 = B1                             # 第二阶段起点与第一阶段终点相接
        drop2 = np.random.uniform(0.1, 0.5)
        B2 = A2 - drop2                     # 第二阶段结束值，满足 A2 > B2
        # 顶点 h2 选在 ((T1 + T_total)/2, T_total)
        h2 = np.random.uniform((T1 + T_total)/2, T_total)
        a2 = (A2 - B2) / ((T1 - h2)**2 - (T_total - h2)**2)
        c2 = A2 - a2*(T1 - h2)**2
        trend2 = a2*(t2 - h2)**2 + c2

        # 合并两个阶段
        return np.concatenate([trend1, trend2])

    def _generate_random_correlation_matrix(self, n, k=3, epsilon=0.001):
        """
        利用低秩构造生成一个随机相关矩阵：
        1. 生成形状为 (n, k) 的矩阵 Q（k 可以小于 n）；
        2. 计算 R = Q Q^T，并归一化使得对角线为 1；
        3. 加入 epsilon*I 以确保正定性，再次归一化。
        """
        Q = np.random.randn(n, k)
        R = Q @ Q.T
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        R = R + epsilon * np.eye(n)
        d = np.sqrt(np.diag(R))
        R = R / np.outer(d, d)
        print(R)
        return R

    def _generate_var_residuals(self):
        """
        利用 VAR(1) 模型生成各序列间相关的残差部分，
        其中：
         - 自回归系数矩阵 A 采用正态分布，取值范围较大后再缩放确保平稳性；
         - 噪声协方差矩阵采用自定义的随机相关矩阵，从而使得不同变量间的噪声相关性有的高有的低，
           最终影响到整个 VAR 模型的变量间相关性。
        """
        T = self.T
        n_series = self.n_series

        # 1. 构造自回归系数矩阵 A，采用正态分布（可取正负值）
        A = np.random.randn(n_series, n_series) * 0.2
        # 确保 VAR 模型平稳：若 A 的最大特征值超过 0.95，则进行缩放
        eigenvalues = np.linalg.eigvals(A)
        max_eigen = np.max(np.abs(eigenvalues))
        if max_eigen >= 0.95:
            A = A * (0.95 / max_eigen)

        # 2. 生成噪声的随机相关矩阵 R，然后构造噪声协方差矩阵 cov_noise = sigma^2 * R
        R = self._generate_random_correlation_matrix(n_series, k=5, epsilon=0.1)
        cov_noise = (self.sigma ** 2) * R

        residuals = np.zeros((T, n_series))
        for t_idx in range(1, T):
            noise = np.random.multivariate_normal(mean=np.zeros(n_series), cov=cov_noise)
            residuals[t_idx] = A.dot(residuals[t_idx - 1]) + noise
        return residuals

    def simulate(self):
        """
        生成多元时间序列数据，返回一个 numpy 数组，形状为 (T, n_series)
        """
        T = self.T
        n_series = self.n_series
        t = np.arange(T)
        # 生成 VAR(1) 模型的残差部分
        residuals = self._generate_var_residuals()
        trend = np.zeros((T, n_series))
        season = np.zeros((T, n_series))
        sim_data = np.zeros((T, n_series))

        # 为每个序列生成趋势、季节性和残差
        for i in range(n_series):
            # 随机选择趋势类型
            trend_type = np.random.choice([
                'linear', 'sigmoid', 'decline',
                'increase_then_decrease', 'decrease_then_increase',
                'quadratic_max', 'quadratic_min', 'double_quadratic',
            ])
            print(i, trend_type)
            if trend_type == 'linear':
                slope = np.random.uniform(0.00001, 0.0001)
                intercept = np.random.uniform(-1, 1)
                trend_comp = self._linear_trend(t, slope, intercept)
            elif trend_type == 'sigmoid':
                k = np.random.uniform(0.0001, 0.001)
                t0 = np.random.uniform(T * 0.3, T * 0.7)
                amplitude = np.random.uniform(5, 10)
                intercept = np.random.uniform(-3, 2)
                trend_comp = self._sigmoid_trend(t, k, t0, amplitude, intercept)
            elif trend_type == 'decline':
                slope = np.random.uniform(0.00001, 0.0001)
                intercept = np.random.uniform(5, 10)
                trend_comp = self._declining_trend(t, slope, intercept)
            elif trend_type == 'increase_then_decrease':
                peak_t = np.random.randint(int(T * 0.3), int(T * 0.7))
                max_value = np.random.uniform(5, 10)
                intercept = np.random.uniform(-1, 1)
                trend_comp = self._increase_then_decrease_trend(t, peak_t, max_value, intercept)
            elif trend_type == 'decrease_then_increase':
                valley_t = np.random.randint(int(T * 0.3), int(T * 0.7))
                min_value = np.random.uniform(5, 10)
                intercept = np.random.uniform(-1, 1)
                trend_comp = self._decrease_then_increase_trend(t, valley_t, min_value, intercept)
            elif trend_type == 'quadratic_max':
                trend_comp = self._quadratic_max_trend(t)
            elif trend_type == 'quadratic_min':
                trend_comp = self._quadratic_min_trend(t)
            elif trend_type == 'double_quadratic':
                trend_comp = self._double_quadratic_trend(t)
            trend[:, i] = trend_comp
            # # 生成两个季节性成分：周期分别为24小时和168小时
            # amp1 = np.random.uniform(0.5, 1)
            # phase1 = np.random.uniform(0, 2 * np.pi)
            # seasonal1 = self._seasonality_component(t, 24, amp1, phase1)
            #
            # amp2 = np.random.uniform(1, 1.5)
            # phase2 = np.random.uniform(0, 2 * np.pi)
            # seasonal2 = self._seasonality_component(t, 168, amp2, phase2)
            # seasonal_comp = seasonal1 + seasonal2

            # 生成交通流量风格的季节性成分
            daily_amplitude = np.random.uniform(0.5, 1.5)
            phase = np.random.uniform(0, 2 * np.pi)
            seasonal_comp = self._traffic_seasonality_component(
                t, daily_amplitude, phase, weekday_factor=1.0, weekend_factor=0.5
            )
            season[:, i] = seasonal_comp

            # 将趋势、季节性和残差相加得到最终序列
            sim_data[:, i] = trend[:, i] + season[:, i]  + residuals[:, i]
        return sim_data, trend, season, residuals

# 以下为示例：如何使用该 class 生成数据
if __name__ == "__main__":
    # 设置随机种子，保证结果可复现
    random_seed = 42

    import pandas as pd
    import matplotlib.pyplot as plt
    # 创建模拟器实例，并生成数据
    simulator = MultiTimeSeriesSimulator(n_series=21, random_seed=random_seed)
    sim_data, trend, season, residuals = simulator.simulate()
    df = pd.DataFrame(sim_data)
    # 导出 CSV（核心操作）
    df.to_csv('../dataset/simdata/simdata.csv', index=False)

    # 打印生成数据的 shape
    print("生成数据的形状：", sim_data.shape)
    # 绘制前 5 个序列
    plt.figure(figsize=(12, 8))
    for i in range(sim_data.shape[1]):
    # for i in range(5, 10):
        # plt.plot(sim_data[4500:5000, i], label=f'Series {i + 1}')
        plt.plot(sim_data[:, i], label=f'Series {i}')

    plt.title(f'Generated MultiTimeSeries Data (Shape: {sim_data.shape})')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # 绘制前 5 个序列
    for i in range(11, 17):
        plt.figure(figsize=(12, 8))
        plt.plot(sim_data[:500, i], label=f'Series {i + 1}')
        # plt.plot(sim_data[:, i], label=f'Series {i}')
        plt.legend()
        plt.show()

    # 创建一个热力图
    import seaborn as sns
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.5)
    sns.heatmap(np.corrcoef(residuals.T), annot=False, cmap='coolwarm')
    plt.show()

