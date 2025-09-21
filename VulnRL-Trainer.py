import sys
import os
import time
import random
import math
from dataclasses import dataclass
from collections import deque
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Environment (toy, safe)
class GridEnv:
    def __init__(self, size=6, n_keys=1, max_steps=80, seed=None):
        self.size = size
        self.n_keys = n_keys
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.steps = 0
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self.keys = set()
        while len(self.keys) < self.n_keys:
            x = self.rng.randrange(self.size)
            y = self.rng.randrange(self.size)
            if [x, y] not in [self.agent_pos, self.goal_pos]:
                self.keys.add((x, y))
        self.collected = set()
        return self._obs()

    def _obs(self):
        grid = np.zeros((3, self.size, self.size), dtype=np.float32)
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        grid[0, ax, ay] = 1.0
        grid[1, gx, gy] = 1.0
        for (kx, ky) in self.keys:
            if (kx, ky) not in self.collected:
                grid[2, kx, ky] = 1.0
        return grid.flatten()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        x, y = self.agent_pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and y < self.size-1:
            y += 1
        elif action == 2 and x < self.size-1:
            x += 1
        elif action == 3 and y > 0:
            y -= 1
        self.agent_pos = [x, y]

        if (x, y) in self.keys and (x, y) not in self.collected:
            self.collected.add((x, y))

        done = False
        reward = -0.01
        reward += 0.5 * len(self.collected)
        if self.agent_pos == self.goal_pos and len(self.collected) == len(self.keys):
            reward += 5.0
            done = True
        if self.steps >= self.max_steps:
            done = True
        return self._obs(), float(reward), done, {}

    @property
    def obs_size(self):
        return 3 * self.size * self.size

    @property
    def n_actions(self):
        return 4

    def render_to_matrix(self):
        mat = np.zeros((self.size, self.size), dtype=int)
        for (kx, ky) in self.keys:
            if (kx, ky) not in self.collected:
                mat[kx, ky] = 2
        ax, ay = self.agent_pos
        gx, gy = self.goal_pos
        mat[gx, gy] = 3
        mat[ax, ay] = 1
        return mat

# DQN Agent components
class SimpleNet(nn.Module):
    def __init__(self, obs_size, n_actions, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(32, hidden//2)),
            nn.ReLU(),
            nn.Linear(max(32, hidden//2), n_actions)
        )

    def forward(self, x):
        return self.net(x)

@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buf = deque(maxlen=capacity)

    def push(self, *args):
        self.buf.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s = np.stack([b.s for b in batch])
        a = np.array([b.a for b in batch], dtype=np.int64)
        r = np.array([b.r for b in batch], dtype=np.float32)
        s2 = np.stack([b.s2 for b in batch])
        d = np.array([b.done for b in batch], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buf)

# Training thread
class TrainerThread(QThread):
    update_signal = pyqtSignal(float, int, float)
    step_signal = pyqtSignal(object)
    finished_signal = pyqtSignal()

    def __init__(self, env: GridEnv, policy_net: nn.Module, target_net: nn.Module,
                 device='cpu', episodes=200, batch_size=64, gamma=0.99, lr=1e-3,
                 sync_every=20, max_steps_per_episode=80, parent=None):
        super().__init__(parent)
        self.env = env
        self.policy = policy_net
        self.target = target_net
        self.device = device
        self.episodes = episodes
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.sync_every = sync_every
        self.max_steps = max_steps_per_episode

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.replay = ReplayBuffer(20000)
        self.running = True

    def run(self):
        eps = 1.0
        eps_min = 0.05
        eps_decay = 0.995
        total_steps = 0

        for episode in range(1, self.episodes + 1):
            if not self.running:
                break
            s = self.env.reset()
            ep_reward = 0.0
            done = False
            step = 0
            while not done and step < self.max_steps:
                step += 1
                total_steps += 1
                if random.random() < eps:
                    a = random.randrange(self.env.n_actions)
                else:
                    with torch.no_grad():
                        input_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
                        logits = self.policy(input_t)
                        a = int(torch.argmax(logits, dim=1).item())
                s2, r, done, _ = self.env.step(a)
                self.replay.push(s, a, r, s2, done)
                ep_reward += r
                s = s2

                if len(self.replay) >= self.batch_size:
                    batch = self.replay.sample(self.batch_size)
                    self._learn_from_batch(batch)

                if step % 2 == 0:
                    mat = self.env.render_to_matrix()
                    self.step_signal.emit(mat)

            eps = max(eps_min, eps * eps_decay)
            self.update_signal.emit(ep_reward, episode, eps)

            if episode % self.sync_every == 0:
                self.target.load_state_dict(self.policy.state_dict())

        self.finished_signal.emit()

    def _learn_from_batch(self, batch):
        s, a, r, s2, d = batch
        s_t = torch.tensor(s, dtype=torch.float32).to(self.device)
        a_t = torch.tensor(a, dtype=torch.int64).to(self.device)
        r_t = torch.tensor(r, dtype=torch.float32).to(self.device)
        s2_t = torch.tensor(s2, dtype=torch.float32).to(self.device)
        d_t = torch.tensor(d, dtype=torch.float32).to(self.device)

        q_vals = self.policy(s_t)
        q_a = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target(s2_t).max(1)[0]
            q_target = r_t + (1.0 - d_t) * self.gamma * q_next
        loss = nn.functional.mse_loss(q_a, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def stop(self):
        self.running = False

# Consent dialog
class ConsentDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ethical Use Agreement")
        self.setWindowIcon(QIcon("AI8.ico"))
        self.setModal(True)
        self.setMinimumWidth(560)
        layout = QtWidgets.QVBoxLayout(self)

        txt = QtWidgets.QLabel(
            "This app is EDUCATIONAL and ALL activities must be legal and ethical.\n"
            "Author is not responsible for illegal use.\n\n"
            "Type EXACTLY: I AGREE  (caps) to proceed."
        )
        txt.setWordWrap(True)
        layout.addWidget(txt)

        self.input = QtWidgets.QLineEdit()
        self.input.setPlaceholderText("Type I AGREE here")
        layout.addWidget(self.input)

        btns = QtWidgets.QHBoxLayout()
        self.ok = QtWidgets.QPushButton("Proceed")
        self.ok.clicked.connect(self.accept_if_ok)
        self.ok.setEnabled(False)
        self.cancel = QtWidgets.QPushButton("Cancel")
        self.cancel.clicked.connect(self.reject)
        btns.addWidget(self.ok)
        btns.addWidget(self.cancel)
        layout.addLayout(btns)

        self.label_err = QtWidgets.QLabel("")
        self.label_err.setStyleSheet("color: red")
        layout.addWidget(self.label_err)

        self.input.textChanged.connect(self._on_text_changed)
        self.apply_qss()

    def _on_text_changed(self, text):
        if text.strip() == "I AGREE":
            self.ok.setEnabled(True)
            # glowing green style
            self.ok.setStyleSheet(
                "QPushButton { background: #3cb371; border-radius: 8px; padding: 8px; color: white; font-weight: bold; }"
            )
            self.label_err.setText("")
        else:
            self.ok.setEnabled(False)
            self.ok.setStyleSheet("")
            if text.strip() != "":
                self.label_err.setText("You must type EXACTLY: I AGREE")
            else:
                self.label_err.setText("")

    def accept_if_ok(self):
        if self.input.text().strip() == "I AGREE":
            self.accept()
        else:
            self.label_err.setText("You must type EXACTLY: I AGREE")

    def apply_qss(self):
        qss = """
        QDialog { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0f1626, stop:1 #1b2540); color: #eaeef6; }
        QLabel { color: #eaeef6; font-size: 13px; }
        QLineEdit { background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); padding: 6px; color: #eaeef6; border-radius: 6px; }
        QPushButton { background: #2a6f97; border-radius: 8px; padding: 8px; color: white; }
        QPushButton:disabled { background: #444a57; color: #888; }
        """
        self.setStyleSheet(qss)

# Matplotlib canvas - shows reward and epsilon
class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(fig)
        self.fig = fig
        self.ax = fig.add_subplot(111)
        self.ax.set_title("Episode reward and epsilon")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.eps_line, = self.ax.plot([], [], label="Epsilon (scaled)")
        self.reward_line, = self.ax.plot([], [], label="Reward")
        self.ax.legend()

    def update_plot(self, episodes, rewards, epsilons=None):
        self.ax.clear()
        self.ax.set_title("Episode reward and epsilon")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.plot(episodes, rewards, label="Reward")
        if epsilons is not None and len(epsilons) == len(episodes):
            # scale epsilon to reward range for combined view if needed
            scaled_eps = [e * (max(rewards) - min(rewards) + 1e-6) / 1.0 for e in epsilons]
            self.ax.plot(episodes, scaled_eps, label="Epsilon (scaled)")
        self.ax.legend()
        self.draw()

# Main window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VulnRL Trainer")
        self.setWindowIcon(QIcon("AI8.ico"))
        self.setMinimumSize(980, 640)
        self._setup_ui()
        self.apply_qss()

        self.env = None
        self.trainer_thread = None
        self.episodes_done = []
        self.rewards_done = []
        self.eps_history = []
        self.current_total_episodes = 0
        self.loaded_model_path = None

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 1)

        self.info_label = QtWidgets.QLabel("Welcome - Ethical RL trainer.")
        left.addWidget(self.info_label)

        form = QtWidgets.QFormLayout()
        self.spin_size = QtWidgets.QSpinBox()
        self.spin_size.setRange(4, 12)
        self.spin_size.setValue(6)
        self.spin_keys = QtWidgets.QSpinBox()
        self.spin_keys.setRange(0, 4)
        self.spin_keys.setValue(1)
        self.spin_episodes = QtWidgets.QSpinBox()
        self.spin_episodes.setRange(10, 2000)
        self.spin_episodes.setValue(200)

        self.spin_hidden = QtWidgets.QSpinBox()
        self.spin_hidden.setRange(32, 2048)
        self.spin_hidden.setValue(256)
        self.spin_lr = QtWidgets.QDoubleSpinBox()
        self.spin_lr.setRange(1e-5, 1e-1)
        self.spin_lr.setDecimals(5)
        self.spin_lr.setValue(1e-3)
        self.spin_sync = QtWidgets.QSpinBox()
        self.spin_sync.setRange(1, 200)
        self.spin_sync.setValue(10)

        form.addRow("Grid size:", self.spin_size)
        form.addRow("Number of keys:", self.spin_keys)
        form.addRow("Episodes:", self.spin_episodes)
        form.addRow("Hidden units:", self.spin_hidden)
        form.addRow("Learning rate:", self.spin_lr)
        form.addRow("Target sync freq:", self.spin_sync)
        left.addLayout(form)

        self.btn_init = QtWidgets.QPushButton("Init Environment")
        self.btn_init.clicked.connect(self.init_env)
        left.addWidget(self.btn_init)

        self.btn_start = QtWidgets.QPushButton("Start Training")
        self.btn_start.clicked.connect(self.start_training)
        self.btn_start.setEnabled(False)
        left.addWidget(self.btn_start)

        self.btn_stop = QtWidgets.QPushButton("Stop Training")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        left.addWidget(self.btn_stop)

        hsave = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save Model")
        self.btn_save.clicked.connect(self.save_model)
        self.btn_save.setEnabled(False)
        self.btn_load = QtWidgets.QPushButton("Load Model")
        self.btn_load.clicked.connect(self.load_model)
        hsave.addWidget(self.btn_save)
        hsave.addWidget(self.btn_load)
        left.addLayout(hsave)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        left.addWidget(self.progress)

        left.addStretch()

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 2)

        grid_group = QtWidgets.QGroupBox("Environment")
        grid_layout = QtWidgets.QVBoxLayout(grid_group)
        self.grid_view = QtWidgets.QLabel()
        self.grid_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grid_view.setTextFormat(Qt.TextFormat.RichText)
        grid_layout.addWidget(self.grid_view)
        right.addWidget(grid_group, 1)

        plot_group = QtWidgets.QGroupBox("Training")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        self.canvas = MatplotlibCanvas(self, width=6, height=4)
        plot_layout.addWidget(self.canvas)
        right.addWidget(plot_group, 1)

    def apply_qss(self):
        qss = """
        QMainWindow { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0f1626, stop:1 #1b2540); color: #eaeef6; }
        QLabel { color: #eaeef6; font-size: 13px; }
        QPushButton { background: #2a6f97; border-radius: 8px; padding: 8px; color: white; }
        QPushButton:disabled { background: #444a57; color: #888; }
        QGroupBox { border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 8px; margin-top: 8px; }
        QSpinBox, QDoubleSpinBox { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.04); padding: 4px; color: #eaeef6; }
        QProgressBar { border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; text-align: center; color:#eaeef6; background: rgba(255,255,255,0.02); }
        """
        self.setStyleSheet(qss)

    def init_env(self):
        size = int(self.spin_size.value())
        nkeys = int(self.spin_keys.value())
        self.env = GridEnv(size=size, n_keys=nkeys, max_steps=80, seed=42)
        self.info_label.setText(f"Initialized grid {size}x{size} with {nkeys} keys.")
        self.btn_start.setEnabled(True)
        self.btn_save.setEnabled(False)
        self._render_env(self.env.render_to_matrix())

    def _render_env(self, mat):
        if mat is None:
            return
        html_rows = []
        for row in mat:
            row_html = ""
            for v in row:
                if v == 0:
                    row_html += "<span style='color:#9aa6b2'>·</span> "
                elif v == 1:
                    row_html += "<span style='color:#2a6f97;font-weight:bold'>A</span> "
                elif v == 2:
                    row_html += "<span style='color:#f2c14e;font-weight:bold'>K</span> "
                elif v == 3:
                    row_html += "<span style='color:#f25c54;font-weight:bold'>G</span> "
            html_rows.append(row_html)
        text = "<br>".join(html_rows)
        font = QtGui.QFont("Courier", 12)
        self.grid_view.setFont(font)
        self.grid_view.setText(text)

    def start_training(self):
        if self.env is None:
            QtWidgets.QMessageBox.warning(self, "No env", "Initialize environment first.")
            return

        obs = self.env.obs_size
        n_actions = self.env.n_actions

        device = 'cpu'
        hidden_units = int(self.spin_hidden.value())
        lr = float(self.spin_lr.value())
        sync_every = int(self.spin_sync.value())
        episodes = int(self.spin_episodes.value())

        policy = SimpleNet(obs, n_actions, hidden=hidden_units).to(device)
        target = SimpleNet(obs, n_actions, hidden=hidden_units).to(device)
        target.load_state_dict(policy.state_dict())

        self.trainer_thread = TrainerThread(
            env=self.env,
            policy_net=policy,
            target_net=target,
            device=device,
            episodes=episodes,
            batch_size=64,
            gamma=0.99,
            lr=lr,
            sync_every=sync_every,
            max_steps_per_episode=80
        )

        self.trainer_thread.update_signal.connect(self._on_episode_update)
        self.trainer_thread.step_signal.connect(self._render_env)
        self.trainer_thread.finished_signal.connect(self._on_training_finished)

        self.episodes_done = []
        self.rewards_done = []
        self.eps_history = []
        self.current_total_episodes = episodes
        self.progress.setValue(0)

        self.btn_init.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_load.setEnabled(False)

        self.trainer_thread.start()
        self.info_label.setText("Training started - keep it ethical")

    def _on_episode_update(self, ep_reward, episode, eps):
        self.episodes_done.append(episode)
        self.rewards_done.append(ep_reward)
        self.eps_history.append(eps)
        self.canvas.update_plot(self.episodes_done, self.rewards_done, self.eps_history)
        percent = int(100.0 * episode / max(1, self.current_total_episodes))
        self.progress.setValue(percent)
        self.info_label.setText(f"Episode {episode} - reward {ep_reward:.3f} - eps {eps:.3f}")

    def stop_training(self):
        if self.trainer_thread:
            self.trainer_thread.stop()
            self.btn_stop.setEnabled(False)
            self.info_label.setText("Stopping...")

    def _on_training_finished(self):
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.btn_init.setEnabled(True)
        self.btn_save.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.info_label.setText("Training finished. Nice work - stay ethical!")

        # show stats
        if len(self.rewards_done) > 0:
            best = max(self.rewards_done)
            avg = sum(self.rewards_done[-50:]) / min(50, len(self.rewards_done))
            msg = QtWidgets.QMessageBox(self)
            msg.setWindowTitle("Training finished")
            msg.setText(f"Training finished.\nBest episode reward: {best:.3f}\nAverage reward (last 50): {avg:.3f}")
            msg.exec()

    def save_model(self):
        if not self.trainer_thread:
            QtWidgets.QMessageBox.information(self, "No model", "No model to save.")
            return
        # try to access policy network from thread
        policy = getattr(self.trainer_thread, "policy", None)
        if policy is None:
            QtWidgets.QMessageBox.information(self, "No model", "No model to save.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save model", os.getcwd(), "Torch model (*.pt)")
        if path:
            try:
                torch.save(policy.state_dict(), path)
                self.loaded_model_path = path
                QtWidgets.QMessageBox.information(self, "Saved", f"Model saved to {path}")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save model: {e}")

    def load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load model", os.getcwd(), "Torch model (*.pt)")
        if not path:
            return
        # create a model with current GUI hyperparams so it matches
        if self.env is None:
            QtWidgets.QMessageBox.warning(self, "No env", "Initialize environment first before loading a model.")
            return
        obs = self.env.obs_size
        n_actions = self.env.n_actions
        hidden_units = int(self.spin_hidden.value())
        model = SimpleNet(obs, n_actions, hidden=hidden_units)
        try:
            state = torch.load(path, map_location='cpu')
            model.load_state_dict(state)
            self.loaded_model_path = path
            QtWidgets.QMessageBox.information(self, "Loaded", f"Model loaded from {path}")
            # expose loaded model by setting trainer_thread placeholders if not training
            if not self.trainer_thread or not self.trainer_thread.isRunning():
                # create a short-lived TrainerThread-like container just to hold policy for save
                self.trainer_thread = TrainerThread(
                    env=self.env, policy_net=model, target_net=model, device='cpu',
                    episodes=1, batch_size=1, gamma=0.99, lr=1e-3, sync_every=1, max_steps_per_episode=1
                )
                # do not start thread
                self.btn_save.setEnabled(True)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load model: {e}")

def main():
    app = QtWidgets.QApplication(sys.argv)

    dlg = ConsentDialog()
    if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        print("User did not accept ethics agreement. Exiting.")
        sys.exit(0)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()