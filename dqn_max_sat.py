import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt


class MaxSatEnvironment:
    def __init__(self, weights, clauses):
        self.clauses = clauses
        self.num_vars = len(weights)
        self.num_clauses = len(clauses)

        self.solution = [0] * self.num_vars
        self.clause_sat_count = [0] * self.num_clauses
        self.var_occ = [[] for _ in range(self.num_vars)]

        # preprocess clause structure
        for c_idx, clause in enumerate(clauses):
            for lit in clause:
                v = abs(lit) - 1
                sign = 1 if lit > 0 else -1
                self.var_occ[v].append((c_idx, sign))

        self.reset()

    def reset(self):
        self.solution = [random.randint(0, 1) for _ in range(self.num_vars)]
        self.recompute()
        return self.full_state()

    def recompute(self):
        self.clause_sat_count = [0] * self.num_clauses
        for c_idx, clause in enumerate(self.clauses):
            sat = 0
            for lit in clause:
                v = abs(lit) - 1
                val = self.solution[v]
                if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                    sat += 1
            self.clause_sat_count[c_idx] = sat
        self.satisfied = sum(1 for x in self.clause_sat_count if x > 0)

    def variable_features(self, var):
        old_val = self.solution[var]
        new_val = 1 - old_val

        make = 0
        brk = 0
        weighted_sat_improv = 0

        for (c, sign) in self.var_occ[var]:
            curr = self.clause_sat_count[c]
            now_sat = (sign > 0 and old_val == 1) or (sign < 0 and old_val == 0)
            new_sat = (sign > 0 and new_val == 1) or (sign < 0 and new_val == 0)

            if not now_sat and new_sat and curr == 0:
                make += 1
            if now_sat and not new_sat and curr == 1:
                brk += 1
            if curr == 0 and new_sat:
                weighted_sat_improv += 1
            if curr == 1 and now_sat and not new_sat:
                weighted_sat_improv -= 1

        unsat_fraction = (self.num_clauses - self.satisfied) / self.num_clauses

        return np.array([
            make,
            brk,
            weighted_sat_improv,
            old_val,
            1 - old_val,
            len(self.var_occ[var]),
            make - brk,
            unsat_fraction
        ], dtype=np.float32)

    def flip(self, var):
        self.solution[var] ^= 1
        self.recompute()

    def full_state(self):
        return np.array([self.variable_features(v) for v in range(self.num_vars)])


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch):
        batch = random.sample(self.buf, batch)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.array(ns),
            np.array(d, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buf)


class QNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, num_vars):
        self.num_vars = num_vars
        self.state_dim = 8
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.995
        self.batch = 128
        self.update_target_freq = 500

        self.q = QNet(self.state_dim)
        self.target = QNet(self.state_dim)
        self.target.load_state_dict(self.q.state_dict())

        self.optim = optim.Adam(self.q.parameters(), lr=1e-4)
        self.rb = ReplayBuffer()
        self.steps_trained = 0

    def act(self, state_matrix):
        current_num_vars = state_matrix.shape[0]
        if random.random() < self.eps:
            return random.randint(0, current_num_vars - 1)

        with torch.no_grad():
            s = torch.tensor(state_matrix, dtype=torch.float32)
            qvals = self.q(s)
            best = torch.argmax(qvals.squeeze())
        return best.item()

    def train(self):
        if len(self.rb) < 500:
            return None

        s, a, r, ns, d = self.rb.sample(self.batch)
        s = torch.tensor(s, dtype=torch.float32)
        ns = torch.tensor(ns, dtype=torch.float32)
        r = torch.tensor(r, dtype=torch.float32)
        d = torch.tensor(d, dtype=torch.float32)

        qvals = self.q(s).squeeze()
        with torch.no_grad():
            target_q = self.target(ns).squeeze()
            y = r + self.gamma * (1 - d) * target_q

        loss = nn.functional.mse_loss(qvals, y)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.steps_trained += 1
        if self.steps_trained % self.update_target_freq == 0:
            self.target.load_state_dict(self.q.state_dict())
        return loss.item()


def solve_instance(agent, w, c, max_flips=5000):
    env = MaxSatEnvironment(w, c)
    s_matrix = env.full_state()
    best_sat = env.satisfied
    total_loss = 0
    loss_count = 0

    for flip in range(max_flips):
        action = agent.act(s_matrix)
        s = s_matrix[action]

        before = env.satisfied
        env.flip(action)
        after = env.satisfied

        reward = after - before
        if after == len(c):
            reward += 10
        if reward <= 0:
            reward -= 0.005

        next_matrix = env.full_state()
        ns = next_matrix[action]
        done = (after == len(c))

        agent.rb.push(s, action, reward, ns, done)
        loss = agent.train()
        if loss is not None:
            total_loss += loss
            loss_count += 1

        s_matrix = next_matrix
        best_sat = max(best_sat, after)
        if done:
            break

    agent.eps = max(agent.eps_min, agent.eps * agent.eps_decay)
    avg_loss = total_loss / loss_count if loss_count > 0 else 0
    return best_sat, flip + 1, avg_loss


def read_input(filename):
    weights = []
    clauses = []
    with open(filename, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("c") or line.startswith("p"):
            continue
        if line.startswith("w"):
            parts = list(map(int, line.split()[1:]))
            if parts[-1] == 0:
                parts = parts[:-1]
            weights = parts
        else:
            parts = list(map(int, line.split()))
            if parts[-1] == 0:
                parts = parts[:-1]
            if parts:
                clauses.append(parts)
    return weights, clauses



def plot_cumulative(data, title, xlabel, ylabel, output_file, epoch_starts=None):
    plt.figure(figsize=(10,5))
    plt.plot(range(1,len(data)+1), data, linestyle='-')
    if epoch_starts:
        for e_idx in epoch_starts:
            plt.axvline(x=e_idx, color='red', linestyle='--', alpha=0.6)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def plot_histogram(data, title, xlabel, ylabel, output_file, bin_width=1):
    min_val = min(data) - 3
    max_val = max(data) + 3
    bins = list(range(min_val, max_val+1, bin_width))
    plt.figure(figsize=(10,5))
    plt.hist(data, bins=bins, color='green', alpha=0.7, rwidth=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()



def run_folder(folder, num_vars, max_flips=5000, num_epochs=2, save_weights=True):
    agent = DQNAgent(num_vars=num_vars)
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".mwcnf")])
    output_folder = "plots_dqn_new_arch"
    os.makedirs(output_folder, exist_ok=True)

    cumulative_flips = []
    last_epoch_flips = []
    last_epoch_sat = []
    epoch_starts = []
    total_problems = 0

    for epoch in range(num_epochs):
        epoch_start_index = len(cumulative_flips) + 1
        epoch_starts.append(epoch_start_index)

        if epoch > 0:
            agent.eps = 0.5 / epoch

        for fp in files:
            w, c = read_input(fp)
            if len(w) != num_vars:
                continue

            sat, flips, loss = solve_instance(agent, w, c, max_flips=max_flips)
            cumulative_flips.append(flips)
            total_problems += 1
            if epoch == num_epochs - 1:
                last_epoch_sat.append(sat)
                last_epoch_flips.append(flips)

            print(f"[Epoch {epoch+1}] {os.path.basename(fp)} | SAT: {sat}/{len(c)} | Flips: {flips} | Loss: {loss:.4f}")

    plot_cumulative(
        cumulative_flips,
        f"Cumulative Flips - {folder}",
        "Problem Index",
        "Flips",
        os.path.join(output_folder, f"{folder}_cumulative_flips.png"),
        epoch_starts
    )
    plot_cumulative(
        last_epoch_sat,
        f"Cumulative Satisfied Clauses - {folder} (Last Epoch)",
        "Problem Index",
        "Clauses Satisfied",
        os.path.join(output_folder, f"{folder}_cumulative_sat_last_epoch.png")
    )
    plot_histogram(
        last_epoch_sat,
        f"Histogram of Clauses Satisfied - {folder} (Last Epoch)",
        "Satisfied Clauses",
        "Frequency",
        os.path.join(output_folder, f"{folder}_hist_sat_last_epoch.png")
    )

    avg_flips = np.mean(last_epoch_flips)
    avg_sat = np.mean(last_epoch_sat)
    with open(os.path.join(output_folder, f"{folder}_averages.txt"), "w") as f:
        f.write(f"Average flips per problem (last epoch): {avg_flips:.2f}\n")
        f.write(f"Average clauses satisfied (last epoch): {avg_sat:.2f}\n")
        f.write(f"Total problems solved: {total_problems}\n")

    if save_weights:
        model_save_path = os.path.join(output_folder, f"{folder}_dqn_weights.pth")
        torch.save(agent.q.state_dict(), model_save_path)
        print(f"Trained DQN weights saved to: {model_save_path}")


if __name__ == "__main__":
    run_configs = [
        {"folder":"wruf36-157R-Q","num_vars":36,"max_flips":5000,"num_epochs":2}
    ]
    for config in run_configs:
        run_folder(
            folder=config["folder"],
            num_vars=config["num_vars"],
            max_flips=config["max_flips"],
            num_epochs=config["num_epochs"]
        )
