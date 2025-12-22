#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# RPC Visualization Script for MountainCarContinuous (Real YOLO Loop)
# Requires: gymnasium, torch, numpy, opencv-python

import os
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from multiprocessing.connection import Client

# 尝试导入 OpenCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ Warning: 'opencv-python' not found. Visualization window will not show.")

# ==========================================
# 1. 网络结构 (必须匹配 MountainCar 的 2D 输入)
# ==========================================
class NNPolicy(nn.Module):
    def __init__(self, hidden_size=0):
        super().__init__()
        
        # MountainCar 输入是 2维: [position, velocity]
        if hidden_size > 0:
            print(f"🧠 Loading MLP Policy (Input: 2 -> Hidden: {hidden_size} -> Output: 1)")
            # Input(2) -> Linear(16) -> ReLU -> Linear(1) -> Tanh
            self.net = nn.Sequential(
                nn.Linear(2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Tanh()
            )
        else:
            print(f"🧠 Loading Linear Policy (Input: 2 -> Output: 1)")
            # Input(2) -> Linear(1) -> Tanh
            self.net = nn.Sequential(
                nn.Linear(2, 1),
                nn.Tanh()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这里的输出直接就是动作，不需要像 Pendulum 那样 * 2.0
        # 因为 Tanh 输出 [-1, 1] 正好符合 MountainCarContinuous 的动作空间
        return self.net(x)

# ==========================================
# 2. RPC 通信模块
# ==========================================
class RPCClient:
    def __init__(self, host, port, authkey=b"mc-rpc"):
        self.address = (host, port)
        try:
            self.conn = Client(self.address, authkey=authkey)
            print(f"✅ Connected to YOLO Server at {host}:{port}")
        except ConnectionRefusedError:
            print(f"❌ Connection Failed! Is the server running on {port}?")
            raise

    def reset(self):
        self.conn.send(("reset", None))
        self.conn.recv()

    def infer(self, frame_bgr):
        self.conn.send(("infer", frame_bgr))
        ok, res = self.conn.recv()
        return res if ok else None

    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

# ==========================================
# 3. 辅助函数
# ==========================================
def set_weights_vector(m: nn.Module, vec: np.ndarray):
    expected = sum(p.numel() for p in m.parameters())
    if vec.size != expected:
        print(f"\n[CRITICAL ERROR] Weight Mismatch!")
        print(f"  > Model expects {expected} params (Input=2).")
        print(f"  > Loaded file has {vec.size} params.")
        print(f"  > Check --hidden-size argument.")
        return False
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n
    return True

# ==========================================
# 4. 可视化主循环
# ==========================================
def run_rpc_visualization(args):
    # 1. 加载模型
    if not os.path.exists(args.model_path):
        print(f"File not found: {args.model_path}")
        return

    data = np.load(args.model_path)
    weights = data['weights']
    
    model = NNPolicy(hidden_size=args.hidden_size)
    if not set_weights_vector(model, weights):
        return
    model.eval()

    # 2. 连接服务器
    try:
        rpc = RPCClient(args.host, args.port, authkey=args.authkey.encode('utf-8'))
    except:
        return

    # 3. 创建环境
    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    
    print(f"\n🎥 Starting Real-Loop Replay ({args.episodes} episodes)...")

    for ep in range(1, args.episodes + 1):
        seed = random.randint(0, 99999)
        obs, _ = env.reset(seed=seed)
        
        rpc.reset()
        
        # MountainCar 需要维护一个“最近有效状态”，防止丢帧导致状态为None
        last_state = np.zeros(2, dtype=np.float32)
        total_reward = 0.0
        steps = 0
        
        while True:
            # A. 获取画面
            frame_rgb = env.render()
            if frame_rgb is None: break
            
            frame_bgr = frame_rgb[..., ::-1].copy()
            
            # B. 显示画面
            if HAS_CV2:
                display_frame = frame_bgr.copy()
                # 显示实时奖励
                cv2.putText(display_frame, f"Reward: {total_reward:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("MC YOLO Client", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    env.close()
                    rpc.close()
                    return

            # C. 发送给 Server 获取 [position, velocity]
            state = rpc.infer(frame_bgr)
            
            if state is None:
                state = last_state
            else:
                last_state = state

            # D. 推理
            s_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                # 连续动作获取
                action_val = model(s_tensor).item()
                act = [action_val]
            
            # E. 环境交互
            _, reward, done, truncated, _ = env.step(act)
            total_reward += reward
            steps += 1
            
            if args.fps > 0:
                time.sleep(1.0 / args.fps)
            
            if done or truncated:
                break
        
        # MountainCar > 90 分才算解决
        status = "SUCCESS 🚩" if total_reward > 90 else "Fail"
        print(f"🎬 Episode {ep} | Seed: {seed} | Reward: {total_reward:.2f} | {status}")
        time.sleep(0.5)

    if HAS_CV2:
        cv2.destroyAllWindows()
    env.close()
    rpc.close()
    print("✨ Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to .npz file")
    
    # 结构参数
    parser.add_argument("--hidden-size", type=int, default=0, help="0 for Linear, 16 for Hidden (default: 0)")
    
    # RPC 参数 (默认端口 6001)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6001, help="Default MC port is 6001")
    parser.add_argument("--authkey", type=str, default="mc-rpc")
    
    # 播放控制
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=0)
    
    args = parser.parse_args()
    run_rpc_visualization(args)