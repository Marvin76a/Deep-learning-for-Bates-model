import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
import matplotlib.pyplot as plt
import random
import time
import os

# ==========================================
# 1. 配置与初始化
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Config:
    def __init__(self):
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 基础参数 ---
        self.d = 100        # 资产数量 (High-dimensional)
        self.T = 1.0        # 到期时间
        self.N = 100         # 时间步数 (离散化越细，SV模拟越准)
        self.M = 1024       # 训练时的 batch size (路径数)
        self.r = 0.05       # 无风险利率
        
        # --- Bates 模型参数 (SVJ) ---
        # 1. 资产初始价格
        self.S0 = 1.0       
        # 2. 随机波动率 Heston 参数
        self.v0 = 0.04      # 初始方差
        self.kappa = 2.0    # 均值回归速度
        self.theta = 0.04   # 长期均值方差
        self.sigma_v = 0.3  # 波动率的波动率 (Vol of Vol)
        self.rho = -0.5     # 资产价格与波动率的相关系数
        # 3. 跳跃参数 (Merton Jump)
        self.lambda_ = 1.0  # 跳跃强度
        self.k = 0.1        # 跳跃幅度 (假设固定比例跳跃)
        
        # --- 衍生品参数 ---
        self.K = 1.0        # 行权价 (用于 Max Call)
        
        # --- 训练参数 ---
        self.epochs = 3000
        self.lr = 1e-3
        self.milestone_step = 1000

# ==========================================
# 2. 路径生成 (SDE Simulation)
# ==========================================

def generate_bates_paths(config, device, dW_iso, dN_tilde):
    """
    生成 Bates 模型 (SVJDM) 路径
    
    参数:
    dW_iso: [N, M, d+1] 独立的标准正态增量
            dim 0: 驱动波动率的噪声 (W^v)
            dim 1..d: 驱动资产的独立噪声 (W^S_perp)
    dN_tilde: [N, M, d] 补偿泊松增量 (仅影响资产)
    
    返回:
    X: [N+1, M, d+1] 状态路径 (前d维为S_t, 最后一维为v_t)
    """
    dt = config.T / config.N
    N, M, _ = dW_iso.shape
    d = config.d
    
    # 初始化状态容器
    # X[:, :, :d] -> 资产价格 S_t
    # X[:, :, d]  -> 波动率 v_t
    X = torch.zeros(N+1, M, d+1, device=device)
    
    # 设置初始值
    X[0, :, :d] = config.S0
    X[0, :, d] = config.v0
    
    # 提取独立噪声源
    # dW_v: [N, M, 1] (广播到所有资产，假设单一波动率因子)
    dW_v_iso = dW_iso[:, :, 0:1] 
    # dW_s_perp: [N, M, d]
    dW_s_iso = dW_iso[:, :, 1:]
    
    # 构造相关噪声 dW_s (S的驱动噪声)
    # dW^S_i = rho * dW^v + sqrt(1-rho^2) * dW^S_perp_i
    # 这保证了 Corr(dW^S_i, dW^v) = rho
    rho = config.rho
    rho_bar = np.sqrt(1 - rho**2)
    
    # 注意：dW_v_iso 需要广播到 [N, M, d]
    dW_s = rho * dW_v_iso + rho_bar * dW_s_iso
    
    for i in range(N):
        S_curr = X[i, :, :d]
        v_curr = X[i, :, d:d+1] # keep dim for broadcasting
        
        # --- 1. 模拟波动率 v_t (CIR Process) ---
        # 使用 Full Truncation 方案防止 v_t 变为负数导致 NaN
        # 方案: f(v) = max(v, 0) in drift and diffusion
        v_plus = torch.clamp(v_curr, min=0.0)
        
        # CIR 离散化
        # dv = kappa(theta - v)dt + sigma_v * sqrt(v) * dW^v
        dv = config.kappa * (config.theta - v_plus) * dt + \
             config.sigma_v * torch.sqrt(v_plus) * dW_v_iso[i]
             
        v_next = v_curr + dv
        
        # --- 2. 模拟资产价格 S_t (Jump Diffusion with SV) ---
        # dS/S = r dt + sqrt(v) dW^S + k dÑ
        # 这里使用 v_plus 确保波动率为实数
        vol_term = torch.sqrt(v_plus)
        
        drift_S = config.r * S_curr * dt
        diff_S = vol_term * S_curr * dW_s[i]
        jump_S = config.k * S_curr * dN_tilde[i]
        
        S_next = S_curr + drift_S + diff_S + jump_S
        
        # 更新状态
        X[i+1, :, :d] = S_next
        X[i+1, :, d:d+1] = v_next
        
    return X

# ==========================================
# 3. 神经网络模型 (BSDE Solver)
# ==========================================

class FeedForwardSubNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 增加一点宽度以适应高维特征
        hidden_dim = input_dim + 20 
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x: [M, input_dim]
        # BN 在 dim 1 (M) 上归一化特征
        x = self.bn0(x) 
        return self.layers(x)

class BSDE_SVJ_Solver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config.d           # 资产维数
        self.x_dim = config.d + 1   # 状态维数 (S + v)
        self.N = config.N
        self.r = config.r
        
        # 待学习的初始价格 Y0
        self.Y0 = nn.Parameter(torch.tensor(1.0)) # 初始化猜测
        
        # Z 网络: 近似梯度/扩散风险
        # 输入: X_t (d+1维)
        # 输出: d+1 维 (对应 d+1 个独立布朗运动源 dW_iso)
        # 解释: Z dot dW_iso = (nabla_u * sigma) dot dW_iso
        self.z_networks = nn.ModuleList([
            FeedForwardSubNet(self.x_dim, self.x_dim) for _ in range(self.N)
        ])
        
        # U 网络: 近似跳跃风险
        # 输入: X_t (d+1维)
        # 输出: d 维 (对应 d 个资产的独立跳跃源 dN_tilde)
        self.u_networks = nn.ModuleList([
            FeedForwardSubNet(self.x_dim, self.d) for _ in range(self.N)
        ])
    
    def forward(self, X, dW_iso, dN_tilde, config):
        """
        X: [N+1, M, d+1]
        dW_iso: [N, M, d+1] (用于 BSDE 积分的独立噪声)
        dN_tilde: [N, M, d]
        """
        dt = config.T / config.N
        M = X.shape[1]
        
        # 初始化 Y_0
        Y = torch.ones(M, device=X.device) * self.Y0
        
        for i in range(self.N):
            X_curr = X[i] # [M, d+1]
            
            # 1. 计算策略函数 Z_i, U_i
            Z_i = self.z_networks[i](X_curr) # [M, d+1]
            U_i = self.u_networks[i](X_curr) # [M, d]
            
            # 2. 计算生成元 f (Driver)
            # 在风险中性定价下, f(t, X, Y, Z, U) = -r * Y
            # 注意: 如果是不同类型的PDE，这里f的形式会变
            f_i = -self.r * Y
            
            # 3. 欧拉前向更新 Y
            # dY = -f dt + Z . dW_iso + U . dN_tilde
            # 注意: 这里 Z 与 dW_iso 点积。dW_iso 是我们在生成路径时使用的那个独立噪声源。
            # 理论上 Z 应该学习将风险敞口映射回这些独立因子。
            martingale_term_W = torch.sum(Z_i * dW_iso[i], dim=1)
            martingale_term_N = torch.sum(U_i * dN_tilde[i], dim=1)
            
            Y = Y - f_i * dt + martingale_term_W + martingale_term_N
            
        # 4. 计算终端 Loss
        # Payoff g(X_T) = (max(S_T) - K)^+  (Max Call Option)
        S_T = X[-1, :, :self.d] # 取出资产部分
        payoff = torch.clamp(torch.max(S_T, dim=1).values - config.K, min=0.0)
        
        return Y, payoff

# ==========================================
# 4. 辅助函数与主循环
# ==========================================

def train_solver(config):
    set_seed(config.seed)
    device = config.device
    
    # 实例化模型
    model = BSDE_SVJ_Solver(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    # 学习率衰减策略
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1500, 2500], gamma=0.1)
    
    loss_history = []
    y0_history = []
    dt = config.T / config.N
    
    start_time = time.time()
    print(f"开始训练 SVJDM (Bates Model) Solver...")
    print(f"配置: d={config.d}, N={config.N}, M={config.M}, Rho={config.rho}, Sigma_v={config.sigma_v}")
    
    for epoch in range(config.epochs):
        # 1. 重新采样噪声 (On-the-fly generation)
        # dW_iso: [N, M, d+1] (0 dim is vol noise, 1..d are asset noises)
        dW_iso = Normal(0, np.sqrt(dt)).sample((config.N, config.M, config.d + 1)).to(device)
        
        # dN: Poisson Jumps
        poisson_dist = Poisson(config.lambda_ * dt)
        dN = poisson_dist.sample((config.N, config.M, config.d)).to(device)
        dN_tilde = dN - config.lambda_ * dt
        
        # 2. 生成前向 SDE 路径
        with torch.no_grad(): # SDE 参数固定，不需要对 X 求导 (除非做 sensitivity analysis)
            X = generate_bates_paths(config, device, dW_iso, dN_tilde)
        
        # 3. 前向计算 BSDE
        model.train()
        Y_pred, Y_target = model(X, dW_iso, dN_tilde, config)
        
        # 4. 计算 Loss (MSE)
        loss = torch.mean((Y_pred - Y_target)**2)
        
        # 5. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 记录
        loss_val = loss.item()
        y0_val = model.Y0.item()
        loss_history.append(loss_val)
        y0_history.append(y0_val)
        
        # 打印进度
        if epoch % 100 == 0 or epoch == config.epochs - 1:
            print(f"Step {epoch:04d} | Loss: {loss_val:.4e} | Y0: {y0_val:.4f} | Time: {time.time()-start_time:.1f}s")
            
    total_time = time.time() - start_time
    print(f"训练完成. 总耗时: {total_time:.2f}s")
    print(f"最终预测价格 Y0: {model.Y0.item():.4f}")
    
    return model, loss_history, y0_history

def visualize_results(loss_hist, y0_hist, config):
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(loss_hist)
    plt.yscale('log')
    plt.title(f'Training Loss (SVJDM, d={config.d})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Y0 Convergence
    plt.subplot(1, 2, 2)
    plt.plot(y0_hist, color='orange')
    plt.title(f'Y0 Convergence: {y0_hist[-1]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Price')
    plt.grid(True)
    
    os.makedirs('figs', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'figs/loss_y0_{config.N}.png')
    # plt.show() # 注释掉以避免阻塞循环
    plt.close() # 关闭当前图形以释放内存

# ==========================================
# 5. 运行入口
# ==========================================

if __name__ == "__main__":
    cfg = Config()
    nlist=[300,500,700,900,1100,1300,1500]
    y0list=[]
    for n in nlist:
        cfg.N=n
        trained_model, losses, y0s = train_solver(cfg)
        y0list.append(y0s[-1])
        visualize_results(losses, y0s, cfg)
    plt.plot(nlist,y0list)
    plt.show()