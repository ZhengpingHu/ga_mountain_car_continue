#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# MountainCar GA Client: Energy Shaping (Hidden) + Raw Plotting + Natural Input + Continuous Action Fix

import os
import argparse
import random
import multiprocessing as mp
from multiprocessing.connection import Client
from typing import Optional, Tuple, Sequence, List, Dict
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from tqdm import tqdm
import pandas as pd
import math  # Needed for Energy calc

# [PLOT] 离屏绘图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# --- Boilerplate ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
print = lambda *a, **k: (__import__("builtins").print(*a, **{**k, "flush": True}))

# ----------------------------
# 1. Reproducibility
# ----------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ----------------------------
# 2. Physics Helper (Energy Calculation)
# ----------------------------
def calculate_max_energy(pos_history: List[float], vel_history: List[float]) -> float:
    max_e = -999.0
    
    for pos, vel in zip(pos_history, vel_history):
        # Height approximation (normalized roughly 0-1)
        height = math.sin(3 * pos) * 0.45 + 0.55
        
        # Kinetic Energy weighted
        kinetic = (vel ** 2) * 200.0
        
        total_e = height + kinetic
        if total_e > max_e:
            max_e = total_e
            
    return max_e

# ----------------------------
# 3. Neural Network (修正点：增加隐藏层)
# ----------------------------
class NNPolicy(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()
        # 修正：使用 Sequential 构建带隐藏层的网络
        # 结构：Input(2) -> Linear(2->16) -> ReLU -> Linear(16->1) -> Tanh
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),  # 输入层 -> 隐藏层
            nn.ReLU(),                  # 激活函数 (引入非线性)
            nn.Linear(hidden_size, 1)   # 隐藏层 -> 输出层
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return torch.tanh(out) # 保证输出在 [-1, 1] 之间，符合 MountainCarContinuous 要求

def get_weights_vector(m: nn.Module) -> np.ndarray:
    with torch.no_grad():
        return torch.cat([p.data.flatten() for p in m.parameters()]).cpu().numpy()

def set_weights_vector(m: nn.Module, vec: np.ndarray):
    offset = 0
    with torch.no_grad():
        for p in m.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(vec[offset:offset+n]).view_as(p))
            offset += n

def mutate(vec: np.ndarray, sigma: float) -> np.ndarray:
    return vec + np.random.randn(vec.size) * sigma

def uniform_crossover(p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.random.rand(len(p1)) < 0.5
    c1 = np.where(mask, p1, p2)
    c2 = np.where(~mask, p1, p2)
    return c1, c2

# ----------------------------
# 4. Seed Management
# ----------------------------
class SeedAgeScheduler:
    def __init__(self, seed_pool: Sequence[int], rng_seed: int = 0):
        self.seed_pool = list(seed_pool)
        self.ages: List[int] = [0 for _ in self.seed_pool]
        self.rng = random.Random(rng_seed)
    @classmethod
    def from_fixed_pool(cls, pool_size: int, base_seed: int, shuffle: bool, rng_seed: int):
        g = np.random.default_rng(base_seed)
        pool = g.integers(low=0, high=2**31-1, size=pool_size, dtype=np.int32).tolist()
        if shuffle:
            rr = random.Random(rng_seed); rr.shuffle(pool)
        return cls(pool, rng_seed=rng_seed)
    def select_subset(self, k: int) -> List[int]:
        n = len(self.seed_pool); idx = list(range(n)); self.rng.shuffle(idx)
        idx.sort(key=lambda i: self.ages[i], reverse=True); return [self.seed_pool[i] for i in idx[:k]]
    def update_after_generation(self, chosen_seeds: Sequence[int]):
        chosen = set(chosen_seeds)
        for i, s in enumerate(self.seed_pool): self.ages[i] = 0 if s in chosen else self.ages[i] + 1
    def get_age_of_seed(self, seed: int) -> int:
        try: return self.ages[self.seed_pool.index(seed)]
        except ValueError: return -1
    def state_dict(self) -> dict: return {"seed_pool": self.seed_pool, "ages": self.ages}
    def load_state_dict(self, d: dict): self.ages = list(d["ages"])

class SeedPortfolioManager:
    def __init__(self, pool_size, base_seed, shuffle, pool_rng_seed, subset_k):
        self.scheduler = SeedAgeScheduler.from_fixed_pool(pool_size, base_seed, shuffle, pool_rng_seed)
        self.master_pool = self.scheduler.seed_pool
        self.active_subset = self.scheduler.select_subset(subset_k)
        print(f"🌱 Initial seed subset: {self.active_subset}")
    def get_active_subset(self) -> List[int]: return self.active_subset
    
    def update_and_refresh(self, results_matrix: np.ndarray, refresh_frac: float, refresh_direction: str, max_seed_age: int):
        # NOTE: results_matrix passed here should be RAW scores to evaluate true difficulty
        self.scheduler.update_after_generation(self.active_subset)
        indices_to_replace = set()
        
        if max_seed_age > 0:
            age_indices = {i for i, s in enumerate(self.active_subset) if self.scheduler.get_age_of_seed(s) > max_seed_age}
            indices_to_replace.update(age_indices)
            
        seed_total_scores = results_matrix.sum(axis=0)
        num_perf_replace = int(np.floor(len(self.active_subset) * refresh_frac))
        
        if num_perf_replace > 0 and refresh_direction != 'none':
            sorted_indices = np.argsort(seed_total_scores) 
            perf_indices = set()
            if refresh_direction == 'bottom': # Hardest (Low score)
                indices = sorted_indices[:num_perf_replace] 
                perf_indices.update(indices)
            elif refresh_direction == 'top': # Easiest
                indices = sorted_indices[-num_perf_replace:]
                perf_indices.update(indices)
            indices_to_replace.update(perf_indices)
            
        if not indices_to_replace: return
        
        num_to_replace = len(indices_to_replace)
        candidate_pool = [s for s in self.master_pool if s not in self.active_subset]
        if len(candidate_pool) < num_to_replace:
            num_to_replace = len(candidate_pool); indices_to_replace = list(indices_to_replace)[:num_to_replace]
        
        if num_to_replace > 0:
            new_seeds = random.sample(candidate_pool, num_to_replace)
            for i_rep, i_new in zip(indices_to_replace, range(num_to_replace)):
                self.active_subset[i_rep] = new_seeds[i_new]
            # print(f"  - Replaced {len(new_seeds)} seeds.")

    def state_dict(self): return self.scheduler.state_dict()
    def load_state_dict(self, d): self.scheduler.load_state_dict(d)

# ----------------------------
# 5. RPC Client
# ----------------------------
class RPCClient:
    def __init__(self, host, port, authkey="mc-rpc"):
        self.address = (host, port); self.authkey = authkey.encode("utf-8"); self.conn: Optional[Client] = None
    def __enter__(self):
        self.conn = Client(self.address, authkey=self.authkey); return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn: self.conn.close()
    def reset(self):
        self.conn.send(("reset", None)); _ = self.conn.recv()
    def infer(self, frame_bgr):
        self.conn.send(("infer", frame_bgr)); ok, z = self.conn.recv(); return z if ok else None

# ----------------------------
# 6. Evaluation Function (修正点：动作获取逻辑)
# ----------------------------
def evaluate_individual(args):
    """
    Returns: (pop_idx, seed_idx, SHAPED_REWARD, RAW_REWARD)
    """
    pop_idx, seed_idx, weights, seed, rpc_host, rpc_port, authkey, max_steps = args
    model = NNPolicy(); set_weights_vector(model, weights); total_reward = 0.0
    
    # History for Energy Calculation
    pos_history = []
    vel_history = []
    
    try:
        env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
        obs, _ = env.reset(seed=int(seed))
        
        pos_history.append(obs[0])
        vel_history.append(obs[1])
        
        last_valid_state = np.zeros(2, dtype=np.float32) 
        
        with RPCClient(rpc_host, rpc_port, authkey) as rpc:
            rpc.reset()
            for t in range(max_steps):
                frame = env.render();
                if frame is None: break
                frame_bgr = frame[..., ::-1].copy()
                
                state = rpc.infer(frame_bgr)
                if state is None: state = last_valid_state
                else: last_valid_state = state
                
                # Record for Energy
                pos_history.append(state[0])
                vel_history.append(state[1])
                
                # Natural Input (No manual scaling)
                s = torch.tensor(state, dtype=torch.float32)
                
                with torch.no_grad(): 
                    # 修正点：连续动作获取
                    # 1. model(s) 输出是一个包含单个值的 Tensor，范围 (-1, 1)
                    # 2. .item() 提取该浮点数
                    # 3. 放入列表 [] 中，因为 Gym 的 continuous env 期望 action 是数组形式
                    action_val = model(s).item()
                    act = [action_val]
                
                obs, reward, done, truncated, info = env.step(act)
                total_reward += reward
                if done or truncated: break
        env.close()
        
        # --- Calculate Rewards ---
        raw_reward = float(total_reward)
        shaped_reward = float(total_reward)
        
        # Apply Energy Shaping ONLY if failed (Raw <= -90.0 roughly means failed in Continuous)
        # In Continuous MC, reward is 100 for target - action^2 * 0.1.
        # If it fails, reward is usually negative around -30 to -50 depending on steps.
        # Let's be safe and use 0.0 as threshold for failure
        if raw_reward <= 0.0:
            max_energy = calculate_max_energy(pos_history, vel_history)
            # Shaping Formula
            shaped_reward = raw_reward + (max_energy * 10.0)
            
        return pop_idx, seed_idx, shaped_reward, raw_reward
            
    except Exception as e:
        # print(f"Eval Error: {e}") # Debug only
        return pop_idx, seed_idx, -500.0, -500.0

# ----------------------------
# 7. Fitness Logic
# ----------------------------
def calculate_competitive_fitness(results_matrix: np.ndarray) -> np.ndarray:
    # Use Proportional Sharing on the SHAPED matrix to guide evolution
    min_val = np.min(results_matrix)
    shifted_matrix = results_matrix - min_val + 0.1
    
    total_rewards_per_seed = shifted_matrix.sum(axis=0)
    total_rewards_per_seed[total_rewards_per_seed == 0] = 1e-9
    
    shared_fitness_matrix = shifted_matrix / total_rewards_per_seed
    fitness_scores = shared_fitness_matrix.sum(axis=1)
    return fitness_scores

# ----------------------------
# 8. Logging & Plotting
# ----------------------------
def save_metrics_csv(run_dir, gen, metrics):
    path = os.path.join(run_dir, "metrics.csv")
    header_needed = not os.path.exists(path)
    df = pd.DataFrame([metrics])
    df.to_csv(path, mode='a', header=header_needed, index=False)

def plot_separated_curves(run_dir, df_history):
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    gens = df_history['generation']
    
    # Plot 1: Optimization Metric (Fitness - Based on Shaped Reward)
    plt.figure(figsize=(10, 6))
    plt.plot(gens, df_history['best_fitness_score'], label='Best Fitness (Shaped)', color='purple')
    plt.xlabel("Generation"); plt.ylabel("Shared Fitness"); plt.grid(True, alpha=0.3)
    plt.title("Evolution Driver (Shaped)")
    plt.savefig(os.path.join(plots_dir, "plot1_fitness_score.png")); plt.close()

    # Plot 2: Real Performance (Raw Reward - No Shaping)
    plt.figure(figsize=(10, 6))
    plt.plot(gens, df_history['global_max_raw_reward'], label='Pop Max Raw', color='green')
    plt.plot(gens, df_history['global_avg_raw_reward'], label='Pop Avg Raw', color='gray', linestyle='--')
    plt.xlabel("Generation"); plt.ylabel("Raw Reward (Gym Standard)"); plt.grid(True, alpha=0.3)
    plt.title("Real Performance (Hidden Shaping)")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "plot2_max_raw_reward.png")); plt.close()

    # Plot 3: Shaping vs Reality
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Generation'); ax1.set_ylabel('Fitness (Shaped)', color='tab:purple')
    ax1.plot(gens, df_history['best_fitness_score'], color='tab:purple')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Selected Raw Reward', color='tab:blue')
    ax2.plot(gens, df_history['selected_individual_raw_reward'], color='tab:blue', linestyle='--')
    plt.title("Shaping Effect vs Real Result"); fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot3_fitness_vs_reward.png")); plt.close()

# [Full Final Evaluation]
def plot_final_summary_plots(final_results_matrix: np.ndarray, master_pool: List[int], run_dir: str):
    try:
        # Note: final_results_matrix contains RAW scores
        print("📊 Generating final evaluation plots (RAW scores)...")
        avg_scores_per_individual = final_results_matrix.mean(axis=1)
        avg_scores_per_seed = final_results_matrix.mean(axis=0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        sns.violinplot(y=avg_scores_per_individual, ax=ax1, inner='quartile', color='lightblue')
        ax1.set_title(f'Individual Performance (Raw)\n(N={len(avg_scores_per_individual)})')
        ax1.set_ylabel('Avg Raw Reward')
        ax1.grid(True, linestyle="--", alpha=0.5)

        sns.violinplot(y=avg_scores_per_seed, ax=ax2, inner='quartile', color='lightgreen')
        ax2.set_title(f'Seed Difficulty (Raw)\n(N={len(master_pool)})')
        ax2.set_ylabel('Avg Raw Reward')
        ax2.grid(True, linestyle="--", alpha=0.5)

        bins = [-np.inf, 90.0, np.inf] # Continuous MC solved is usually > 90
        labels = ["Fail", "Success"]
        categories = pd.cut(avg_scores_per_seed, bins=bins, labels=labels, right=False)
        proportions = categories.value_counts(normalize=True).sort_index() * 100
        
        prop_text = "Seed Difficulty:\n"
        for name, pct in proportions.items():
            prop_text += f" - {name}: {pct:.1f}%\n"
        
        fig.text(0.5, 0.01, prop_text, ha='center', fontsize=10, wrap=True)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        plt.savefig(os.path.join(run_dir, "final_evaluation_plots.png"), dpi=150)
        plt.close(fig)

    except Exception as e:
        print(f"[WARN] Failed to generate final plots: {e}")

def save_history_snapshot(run_dir, gen, population, results_matrix, subset_seeds, fitness_scores):
    history_dir = os.path.join(run_dir, "history")
    os.makedirs(history_dir, exist_ok=True)
    filename = os.path.join(history_dir, f"gen_{gen:04d}.npz")
    np.savez_compressed(filename, generation=gen, population_weights=np.array(population), 
                        results_matrix=results_matrix, subset_seeds=np.array(subset_seeds), 
                        fitness_scores=fitness_scores)

def save_full_checkpoint(run_dir, gen, population, portfolio, fitness_history, args):
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    state = {"generation": gen, "population": population, "portfolio_state": portfolio.state_dict(),
             "fitness_history": fitness_history, "args": vars(args)}
    torch.save(state, os.path.join(ckpt_dir, f"checkpoint_gen_{gen:04d}.pt"))

# ----------------------------
# 9. Main GA Loop
# ----------------------------
def run_ga(args):
    if args.global_seed: set_global_seed(args.global_seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.outdir, f"ga_mc_energy_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    
    portfolio = SeedPortfolioManager(args.pool_size, args.base_seed, args.shuffle_pool, 
                                     args.pool_rng_seed, args.subset_k)
    
    model = NNPolicy(); base_vec = get_weights_vector(model)
    pop = [mutate(base_vec, args.sigma) for _ in range(args.population)]
    
    fitness_history = {}; history_records = []; best_global_raw = -np.inf
    print(f"🚀 MountainCar GA (Energy Shaping) started. Logs: {run_dir}")

    for gen in range(1, args.generations + 1):
        subset_seeds = portfolio.get_active_subset()
        jobs = []
        for i in range(args.population):
            for j, seed in enumerate(subset_seeds):
                jobs.append((i, j, pop[i], seed, args.rpc_host, args.rpc_port, args.authkey, args.max_steps))
        
        # We need TWO matrices: one for evolution (shaped), one for stats (raw)
        shaped_results_matrix = np.full((args.population, args.subset_k), -np.inf)
        raw_results_matrix = np.full((args.population, args.subset_k), -np.inf)
        
        with mp.Pool(processes=args.processes) as pool:
            for pop_idx, seed_idx, shaped_rew, raw_rew in tqdm(pool.imap_unordered(evaluate_individual, jobs), 
                                                              total=len(jobs), desc=f"Gen {gen}"):
                shaped_results_matrix[pop_idx, seed_idx] = shaped_rew
                raw_results_matrix[pop_idx, seed_idx] = raw_rew

        # 1. Update & Refresh (Use RAW score to determine true difficulty)
        portfolio.update_and_refresh(raw_results_matrix, args.seed_refresh_frac, args.seed_refresh_direction, args.max_seed_age)
        
        # 2. Stats (Use RAW score for reporting)
        raw_avg_rewards_per_ind = raw_results_matrix.mean(axis=1)
        global_max_raw = np.max(raw_avg_rewards_per_ind) 
        global_avg_raw = np.mean(raw_avg_rewards_per_ind)
        
        # 3. Fitness (Use SHAPED score for evolution)
        comp_scores = calculate_competitive_fitness(shaped_results_matrix)
        
        # 4. Smoothing
        smoothed_scores = np.zeros(args.population)
        new_hist = {}
        for i in range(args.population):
            k = tuple(pop[i])
            hist = (fitness_history.get(k, []) + [comp_scores[i]])[-args.fitness_avg_generations:]
            smoothed_scores[i] = np.mean(hist)
            new_hist[k] = hist
        fitness_history = new_hist
        
        # 5. Selection (Based on Shaped Fitness)
        elite_num = max(2, int(args.elite_frac * args.population))
        sorted_indices = np.argsort(smoothed_scores)
        elite_indices = sorted_indices[-elite_num:]
        
        champion_idx = elite_indices[-1]
        best_fitness_val = smoothed_scores[champion_idx]
        
        selected_individual_raw_reward = raw_avg_rewards_per_ind[champion_idx]
        
        print(f"🏆 [GEN {gen}] PopMaxRaw={global_max_raw:.2f} | BestFit={best_fitness_val:.4f} | SelRaw={selected_individual_raw_reward:.2f}")
        
        metrics = {
            "generation": gen,
            "best_fitness_score": best_fitness_val,
            "global_max_raw_reward": global_max_raw,
            "global_avg_raw_reward": global_avg_raw,
            "selected_individual_raw_reward": selected_individual_raw_reward
        }
        history_records.append(metrics)
        save_metrics_csv(run_dir, gen, metrics)
        save_history_snapshot(run_dir, gen, pop, shaped_results_matrix, subset_seeds, smoothed_scores)
        
        if gen % args.checkpoint_freq == 0:
            save_full_checkpoint(run_dir, gen, pop, portfolio, fitness_history, args)
        if gen % 1 == 0:
            df_hist = pd.DataFrame(history_records)
            plot_separated_curves(run_dir, df_hist)

        if selected_individual_raw_reward > best_global_raw:
            best_global_raw = selected_individual_raw_reward
            np.savez(os.path.join(run_dir, "best_model.npz"), weights=pop[champion_idx])

        # 6. Evolution
        elites = [pop[i] for i in elite_indices]
        new_pop = elites.copy()
        while len(new_pop) < args.population:
            p1, p2 = random.sample(elites, 2); c1, c2 = uniform_crossover(p1, p2)
            new_pop.append(mutate(c1, args.sigma))
            if len(new_pop) < args.population: new_pop.append(mutate(c2, args.sigma))
        pop = new_pop

    print("\n✅ Training finished.")

    # [Final Eval using RAW scores]
    print(f"\n🏁 Starting final evaluation (RAW scores)...")
    master_pool = portfolio.master_pool
    final_jobs = [(i, j, pop[i], seed, args.rpc_host, args.rpc_port, args.authkey, args.max_steps) 
                  for i in range(args.population) for j, seed in enumerate(master_pool)]
    
    final_raw_matrix = np.full((args.population, len(master_pool)), -np.inf)
    
    try:
        with mp.Pool(processes=args.processes) as pool:
            results_iterator = tqdm(pool.imap_unordered(evaluate_individual, final_jobs), 
                                  total=len(final_jobs), desc="Final Evaluation")
            for pop_idx, seed_idx, shaped_rew, raw_rew in results_iterator:
                final_raw_matrix[pop_idx, seed_idx] = raw_rew # Store Raw
    except Exception as e:
        print(f"\n[FATAL] Final evaluation failed: {e}")

    final_avg_scores = final_raw_matrix.mean(axis=1)
    best_final_idx = np.argmax(final_avg_scores)
    best_final_score = final_avg_scores[best_final_idx]
    
    print(f"🏆 [Final Result] Best Model Avg Raw Score: {best_final_score:+.2f}")
    np.savez(os.path.join(run_dir, "best_model_full_eval.npz"), weights=pop[best_final_idx])
    
    plot_final_summary_plots(final_raw_matrix, master_pool, run_dir)
    print(f"💾 All results saved to {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-host", default="127.0.0.1")
    parser.add_argument("--rpc-port", type=int, default=6001)
    parser.add_argument("--authkey", default="mc-rpc")
    
    parser.add_argument("--population", type=int, default=50)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--subset-k", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=2025)
    
    parser.add_argument("--seed-refresh-frac", type=float, default=0.25)
    parser.add_argument("--seed-refresh-direction", type=str, default="bottom")
    parser.add_argument("--max-seed-age", type=int, default=10)
    
    parser.add_argument("--elite-frac", type=float, default=0.4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--fitness-avg-generations", type=int, default=5)
    
    parser.add_argument("--shuffle-pool", action="store_true", default=True)
    parser.add_argument("--pool-rng-seed", type=int, default=42)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--outdir", default="runs_mc")
    parser.add_argument("--checkpoint-freq", type=int, default=5)

    args = parser.parse_args()
    run_ga(args)