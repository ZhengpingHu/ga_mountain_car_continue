#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# YOLO Server for MountainCar (2-Point Texture Anchor)

import os
import argparse
import numpy as np
import torch
from threading import Thread
from multiprocessing.connection import Listener
from typing import Optional

# 限制线程数，防止抢占资源
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from ultralytics import YOLO
except ImportError:
    raise RuntimeError("Please install ultralytics: pip install ultralytics")

print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

class MountainCarStateEstimator:
    def __init__(self, model_path, device="cuda:0", _loaded_model=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 1. 加载模型
        if _loaded_model:
            self.model = _loaded_model
        else:
            print(f"👁️ [Vision] Loading YOLO model: {model_path} ...")
            self.model = YOLO(model_path).to(self.device)
            
        # 2. 物理参数 (必须与生成数据时完全一致)
        self.min_pos = -1.2
        self.max_pos = 0.6
        self.screen_width = 600
        # 缩放比例: 600 px / 1.8 units = 333.333
        self.scale = self.screen_width / (self.max_pos - self.min_pos)
        
        # 3. 状态缓存
        self.last_pos = None
        self.state = None
        
        # 用于 clone (多进程支持)
        self._init_args = locals() 

    def clone(self):
        """为每个新的连接克隆一个实例"""
        args = self._init_args.copy()
        args.pop('self')
        args['_loaded_model'] = self.model # 共享显存中的模型，不重新加载
        return MountainCarStateEstimator(**args)

    def begin_episode(self):
        """新的一局开始，重置速度计算器"""
        self.last_pos = None
        self.state = np.zeros(2, dtype=np.float32) # [pos, vel]

    @torch.no_grad()
    def process_frame(self, frame_bgr):
        """
        输入: BGR 图像
        输出: [position, velocity]
        """
        # YOLO 推理 (verbose=False 关闭刷屏)
        results = self.model.predict(frame_bgr, verbose=False, device=self.device)
        
        # 如果没检测到关键点，保持上一帧状态 (Zero-Order Hold)
        if not results or not results[0].keypoints:
            return self.state 

        # 获取关键点坐标: shape [2, 2] -> [[x1, y1], [x2, y2]]
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        
        # 稳健性检查: 必须检测到 2 个点
        if kpts.shape[0] < 2:
            return self.state
            
        # [核心逻辑] 使用两个点的中点作为 X 轴锚点
        # 我们的训练数据保证了这两个点的中点 X 坐标与物理位置线性相关
        center_x = (kpts[0][0] + kpts[1][0]) / 2.0
        
        # 反推物理位置 x
        # pixel_x = (pos - min) * scale  =>  pos = (pixel_x / scale) + min
        current_pos = (center_x / self.scale) + self.min_pos
        
        # 边界截断 (防止轻微抖动超出物理范围)
        current_pos = np.clip(current_pos, self.min_pos, self.max_pos)
        
        # 计算速度 v (差分)
        if self.last_pos is None:
            vel = 0.0
        else:
            vel = current_pos - self.last_pos
            
        self.last_pos = current_pos
        self.state = np.array([current_pos, vel], dtype=np.float32)
        
        return self.state.copy()

# --- RPC Server 架构 (复用) ---
class InferenceServer:
    def __init__(self, est, host="127.0.0.1", port=6001, authkey=b"mc-rpc"):
        self.master_est = est
        self.addr = (host, port)
        self.authkey = authkey

    def _handle(self, conn):
        # print(f"[RPC] Worker connected.")
        sess = self.master_est.clone()
        try:
            while True:
                msg = conn.recv()
                if not msg: break
                cmd, payload = msg[0], msg[1] if len(msg)>1 else None
                
                if cmd == "reset":
                    sess.begin_episode()
                    conn.send((True, "ok"))
                elif cmd == "infer":
                    res = sess.process_frame(payload)
                    conn.send((True, res) if res is not None else (False, None))
                else:
                    conn.send((False, "unknown"))
        except: pass
        finally: conn.close()

    def serve(self):
        listener = Listener(self.addr, authkey=self.authkey)
        print(f"🚀 [RPC] MountainCar Vision Server running on {self.addr[0]}:{self.addr[1]}")
        print(f"Waiting for GA clients...")
        while True:
            Thread(target=self._handle, args=(listener.accept(),), daemon=True).start()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--port", type=int, default=6001)
    args = parser.parse_args()
    
    est = MountainCarStateEstimator(args.model)
    InferenceServer(est, port=args.port).serve()

if __name__ == "__main__":
    main()