# pricing_base.py - 基础BSDE求解器模块
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions.poisson import Poisson
import matplotlib.pyplot as plt
import random
import time

def set_seed(seed=42):
    """设置所有随机数生成器的种子以确保可重复性"""
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
        self.d = 100  # 维度
        self.T = 1.0
        self.N = 100
        self.M = 1000  # 每次蒙特卡洛路径数（训练时每 epoch 重新采样）
        self.r = 0.05
        self.lambda_ = 1.0
        self.sigma = 0.2
        self.k = 0.1
        self.K = 1
        self.X0 = torch.ones(self.d)
        self.epochs = 2000  # 可调（论文给出的实验使用 1000~4000）
        self.lr = 1e-3  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_paths(config, device, dW, dN_tilde):
    """
    X shape: [N+1, M, d]
    dW shape: [N, M, d] (increments, Normal(0, sqrt(dt)))
    dN_tilde shape: [N, M, d] (compensated Poisson increments)
    使用形式 dX = r X dt + sigma X dW + k X dÑ    (与论文中 dX = rXdt + ... + kX dÑ 保持一致)
    """
    dt = config.T / config.N
    N = dW.shape[0]
    M = dW.shape[1]
    d = dW.shape[2]
    X = torch.zeros(N+1, M, d, device=device)
    X0 = config.X0.to(device).unsqueeze(0).expand(M, -1)  # [M, d]
    X[0] = X0
    for i in range(N):
        drift = config.r * X[i] * dt  # 使用 r（因为我们用的是 dN_tilde）
        diffusion = config.sigma * X[i] * dW[i]
        jump = config.k * X[i] * dN_tilde[i]
        X[i+1] = X[i] + drift + diffusion + jump
    return X

class ZNetwork(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d+10),
            nn.ReLU(),
            nn.Linear(d+10, d+10),
            nn.ReLU(),
            nn.Linear(d+10, d)
        )
    def forward(self, x):
        return self.layers(x)  # [M, d]

class UNetwork(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d, d+10),
            nn.ReLU(),
            nn.Linear(d+10, d+10),
            nn.ReLU(),
            nn.Linear(d+10, d)
        )
    def forward(self, x):
        return self.layers(x)  # [M, d]  —— 对应每个维度的跳跃增量影响

class BSDESolver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d = config.d
        self.N = config.N
        self.r = config.r
        # 学习 Y0
        self.Y0 = nn.Parameter(torch.tensor(1.0))
        # 每个时间步一个子网络
        self.z_networks = nn.ModuleList([ZNetwork(self.d) for _ in range(self.N)])
        self.u_networks = nn.ModuleList([UNetwork(self.d) for _ in range(self.N)])
    
    def forward(self, X, dW, dN_tilde, config):
        """
        X: [N+1, M, d]
        dW: [N, M, d]
        dN_tilde: [N, M, d]
        返回: Y_pred (shape [M]), terminal payoff (shape [M])
        """
        dt = config.T / config.N
        N = X.shape[0] - 1
        M = X.shape[1]
        Y = torch.ones(M, device=X.device) * self.Y0  # 初始 Y (每条路径的相同起始参数)
        for i in range(N):
            X_i = X[i]  # [M, d]
            Z_i = self.z_networks[i](X_i)  # [M, d]
            U_i = self.u_networks[i](X_i)  # [M, d]
            # f = -r * Y  （文中 f = -r Y）
            f = -self.r * Y
            # 离散更新：Y = Y - f dt + <Z, dW> + <U, dN_tilde>
            Y = Y - f * dt + torch.sum(Z_i * dW[i], dim=1) + torch.sum(U_i * dN_tilde[i], dim=1)
        # 终端支付
        X_T = X[-1]  # [M, d]
        terminal = torch.clamp(torch.max(X_T, dim=1).values - config.K, min=0.0)
        return Y, terminal

def validate(model, config, device, num_val_paths=None):
    """验证函数：使用独立的路径"""
    dt = config.T / config.N
    if num_val_paths is None:
        num_val_paths = config.M
    with torch.no_grad():
        # 重新采样验证路径
        dW_val = Normal(0, np.sqrt(dt)).sample((config.N, num_val_paths, config.d)).to(device)
        poisson = Poisson(config.lambda_ * dt)
        dN_val = poisson.sample((config.N, num_val_paths, config.d)).to(device)
        dN_tilde_val = dN_val - config.lambda_ * dt
        X_val = generate_paths(config, device, dW_val, dN_tilde_val)
        Y_pred_val, Y_true_val = model(X_val, dW_val, dN_tilde_val, config)
        val_loss = torch.mean((Y_pred_val - Y_true_val)**2)
        val_y0 = model.Y0.item()
        return val_loss.item(), val_y0

def basic_train(config):
    """基础训练函数（来自 pricing 0.2.py）"""
    device = config.device
    print("Device:", device)
    model = BSDESolver(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    train_loss_history = []
    val_loss_history = []
    train_y0_history = []
    val_y0_history = []
    dt = config.T / config.N

    # 时间记录
    time_history = []
    start_time = time.time()
    last_milestone_time = start_time

    print("开始训练...")
    print(f"维度 d={config.d}, 路径数 M={config.M}, 时间步 N={config.N}, epochs={config.epochs}, lr={config.lr}")
    print(f"训练开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    for epoch in range(config.epochs):
        # 每个 epoch 重新采样 monte-carlo 增量并生成路径（避免过拟合固定噪声）
        dW = Normal(0, np.sqrt(dt)).sample((config.N, config.M, config.d)).to(device)
        poisson = Poisson(config.lambda_ * dt)
        dN = poisson.sample((config.N, config.M, config.d)).to(device)
        dN_tilde = dN - config.lambda_ * dt
        X = generate_paths(config, device, dW, dN_tilde)

        model.train()
        Y_pred, Y_true = model(X, dW, dN_tilde, config)
        train_loss = torch.mean((Y_pred - Y_true)**2)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_loss_history.append(train_loss.item())
        train_y0_history.append(model.Y0.item())

        # 每1000个epoch记录时间里程碑
        if (epoch + 1) % 1000 == 0:
            current_time = time.time()
            elapsed_total = current_time - start_time
            elapsed_interval = current_time - last_milestone_time
            time_history.append(elapsed_total)
            
            print(f"=== 时间里程碑 Epoch {epoch+1} ===")
            print(f"总训练时间: {elapsed_total:.2f}秒 ({elapsed_total/60:.2f}分钟)")
            print(f"最近1000个epoch用时: {elapsed_interval:.2f}秒")
            print(f"平均每epoch用时: {elapsed_interval/1000:.4f}秒")
            print(f"预计剩余时间: {(config.epochs - epoch - 1) * (elapsed_interval/1000):.2f}秒")
            
            last_milestone_time = current_time

        if (epoch + 1) % 50 == 0:
            model.eval()
            val_loss, val_y0 = validate(model, config, device)
            val_loss_history.append(val_loss)
            val_y0_history.append(val_y0)
            print(f"Epoch [{epoch+1}/{config.epochs}] Train Loss={train_loss.item():.6e} Val Loss={val_loss:.6e} Train Y0={model.Y0.item():.6f} Val Y0={val_y0:.6f}")
        elif (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}] Train Loss={train_loss.item():.6e} Train Y0={model.Y0.item():.6f}")

    # 最终时间统计
    end_time = time.time()
    total_training_time = end_time - start_time

    # 最终独立验证（使用更多路径）
    final_val_loss, final_val_y0 = validate(model, config, device, num_val_paths=5000)
    print("-" * 60)
    print("训练完成")
    print(f"训练结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"总训练时间: {total_training_time:.2f}秒 ({total_training_time/60:.2f}分钟 / {total_training_time/3600:.2f}小时)")
    print(f"平均每epoch用时: {total_training_time/config.epochs:.4f}秒")
    print(f"最终训练 Y0: {model.Y0.item():.6f}")
    print(f"最终验证 Y0 (5000 paths): {final_val_y0:.6f}")
    print(f"最终训练损失: {train_loss_history[-1]:.6e}")
    print(f"最终验证损失: {final_val_loss:.6e}")

    return model, train_loss_history, val_loss_history, train_y0_history, val_y0_history, time_history, total_training_time

def plot_basic_results(model, config, train_loss_history, val_loss_history, train_y0_history, val_y0_history):
    """绘制基础训练结果"""
    device = config.device
    dt = config.T / config.N
    
    # 可视化（最后一次随机采样）
    dW_vis = Normal(0, np.sqrt(dt)).sample((config.N, config.M, config.d)).to(device)
    dN_vis = Poisson(config.lambda_ * dt).sample((config.N, config.M, config.d)).to(device)
    dN_tilde_vis = dN_vis - config.lambda_ * dt
    X_vis = generate_paths(config, device, dW_vis, dN_tilde_vis)
    with torch.no_grad():
        Y_pred_vis, Y_true_vis = model(X_vis, dW_vis, dN_tilde_vis, config)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_loss_history, label='Train Loss')
    val_epochs = list(range(50, config.epochs + 1, 50))
    if len(val_loss_history) > 0:
        axes[0].plot(val_epochs, val_loss_history, 'ro-', label='Val Loss', markersize=4)
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].plot(train_y0_history, label='Train Y0')
    if len(val_y0_history) > 0:
        axes[1].plot(val_epochs, val_y0_history, 'ro-', label='Val Y0', markersize=4)
    axes[1].set_title("Y0 over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Y0")
    axes[1].legend()
    plt.tight_layout()

    # Y_T vs payoff 散点图
    plt.figure(figsize=(5, 5))
    y_pred = Y_pred_vis.detach().cpu().numpy()
    y_true = Y_true_vis.detach().cpu().numpy()
    plt.scatter(y_true, y_pred, s=6, alpha=0.5)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=1)
    plt.xlabel("Payoff g(X_T)")
    plt.ylabel("Predicted Y_T")
    plt.title("Y_T vs g(X_T)")

    # 示例路径（前 10 条，维度 0）
    plt.figure(figsize=(8, 4))
    t_grid = np.linspace(0.0, config.T, config.N + 1)
    X_sample = X_vis[:, :10, 0].detach().cpu().numpy()
    for m in range(X_sample.shape[1]):
        plt.plot(t_grid, X_sample[:, m], alpha=0.7)
    plt.title("Sample paths (first 10, dim 0)")
    plt.xlabel("Time")
    plt.ylabel("X_t[0]")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    config = Config()
    set_seed(config.seed)
    trained_model, train_loss, val_loss, train_y0, val_y0, time_hist, total_time = basic_train(config)
    plot_basic_results(trained_model, config, train_loss, val_loss, train_y0, val_y0)
