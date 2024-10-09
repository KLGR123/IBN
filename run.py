import os
import re
import sys
import time
import json
import numpy as np
import networkx as nx
import requests
from tqdm import tqdm
import mysql.connector
from openai import OpenAI, AzureOpenAI
from collections import deque

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, QMainWindow, QHBoxLayout, QWidget, QTextEdit, QPushButton
from PyQt5.QtCore import QTimer

from prompts import GOAL_LIST, GOAL_PROMPT, PROMPT, ALGO_DICT


class WebNode:
    def __init__(self, name, id, attributes):
        self.name = name
        self.id = id
        self.attributes = attributes


def interact(
    model: int = 0, 
    device: str = "cpu", 
    goal_id: int = 0, 
    dist_id: int = 0
): 

    res = requests.post(
        f'http://10.3.242.47:8074/sim/{goal_id}/{dist_id}', 
        json={"model": model, "divice": device}
    )
    res_dict = eval(res.json())
    state = {key: value for key, value in res_dict.items() if key != "gt"}
    gt = int(res_dict["gt"])
    return state, gt


class StartupDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.intent_label = QLabel("请输入初始意图:")
        self.intent_input = QLineEdit()
        layout.addWidget(self.intent_label)
        layout.addWidget(self.intent_input)

        self.dist_label = QLabel("请选择 SNR 测试趋势:")
        self.dist_combo = QComboBox()
        self.dist_combo.addItems(["平缓", "周期", "抖动"])
        layout.addWidget(self.dist_label)
        layout.addWidget(self.dist_combo)

        self.buttons_layout = QHBoxLayout()
        self.ok_button = QPushButton("确定")
        self.cancel_button = QPushButton("取消")
        self.buttons_layout.addWidget(self.ok_button)
        self.buttons_layout.addWidget(self.cancel_button)
        layout.addLayout(self.buttons_layout)

        self.setLayout(layout)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

    def get_inputs(self):
        return self.intent_input.text(), self.dist_combo.currentIndex()


class MainWindow(QMainWindow):
    def __init__(self, intent: str, dist_id: int):
        super().__init__()
        self.intent = intent
        self.dist_id = dist_id

        self.setWindowTitle("达意组网智能体 UAV 仿真交互环境")
        self.setGeometry(100, 100, 800, 800)

        self.queues = [deque([0]*10, maxlen=10) for _ in range(4)]
        self.ref_list = ["time", "ssim", "psnr", "ss"]
        self.ref_name_list = ["DELAY", "SSIM", "pSNR", "SS"]
        self.gt_list = deque([0]*10, maxlen=10)
        self.model_id_list = deque([0]*10, maxlen=10)
        self.baseline_values = None
        self.accuracy_list = []
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(main_layout)
        
        left_layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        left_layout.addWidget(self.text_edit)
        main_layout.addLayout(left_layout)

        right_layout = QVBoxLayout()
        self.figure_bar = Figure()
        self.canvas_bar = FigureCanvas(self.figure_bar)
        self.ax_bars = [self.figure_bar.add_subplot(4, 1, i + 1) for i in range(4)]
        for ax in self.ax_bars:
            ax.set_ylim(0, 1)
        right_layout.addWidget(self.canvas_bar)

        self.figure_topo = Figure()
        self.ax_topo = self.figure_topo.add_subplot(111)
        self.canvas_topo = FigureCanvas(self.figure_topo)
        right_layout.addWidget(self.canvas_topo)

        self.figure_acc = Figure()
        self.ax_acc = self.figure_acc.add_subplot(111)
        self.canvas_acc = FigureCanvas(self.figure_acc)
        right_layout.addWidget(self.canvas_acc)

        self.update_button = QPushButton("异步更新")
        self.update_button.clicked.connect(self.update_all)
        right_layout.addWidget(self.update_button)

        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

        # self.client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
        #     api_key=os.getenv("AZURE_OPENAI_KEY"),  
        #     api_version="2023-05-15"
        # )

        messages = [
            {"role": "system", "content": GOAL_PROMPT},
            {"role": "user", "content": self.intent}
        ]
        response = self.client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            temperature=0.1
        )
        response_text = response.choices[0].message.content
        assert type(eval(response_text)) == int

        goal_id_infer = eval(response_text)
        self.goal = GOAL_LIST[goal_id_infer]
        self.goal_id = goal_id_infer

        self.label = QLabel(self.goal.split("，")[0])
        right_layout.addWidget(self.label)

        right_layout.setStretch(0, 3)
        right_layout.setStretch(1, 1)
        right_layout.setStretch(2, 1)
        right_layout.setStretch(3, 0)
        main_layout.addLayout(right_layout)

        self.last_model_id, self.model_id = 1, 1
        self.last_device, self.device = 'gpu', 'gpu'
        self.history = []

        self.ctime = time.time()
        os.makedirs(f"logs/{self.ctime}", exist_ok=True)
        self.text_log_file = f"logs/{self.ctime}/text_log.txt"
        self.bar_log_file = f"logs/{self.ctime}/bar_log.txt"

    def update_all(self):
        state, gt = interact(
            model=self.model_id, 
            device=self.device,
            goal_id=self.goal_id,
            dist_id=self.dist_id
        )

        if self.baseline_values is None:
            self.baseline_values = {key: float(state[key]) for key in self.ref_list}

        self.text_edit.append(f"<span style='color: black;'>{state}</span>")

        prompt = PROMPT.format(
            goal=GOAL_LIST[self.goal_id], 
            history=self.history
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": str(state)}
        ]

        response = self.client.chat.completions.create(
            messages=messages,
            model="gpt-4o",
            temperature=0.1
        )
        response_text = response.choices[0].message.content

        pattern = r'```json(.*?)```'
        matches = re.findall(pattern, response_text, re.DOTALL)
        action = json.loads(matches[0]) if matches else None

        self.model_id = int(action["model"])
        self.device = str(action["device"])
        thoughts = str(action["thoughts"])
        prediction = str(action["prediction"])

        self.history.append(state)
        self.history.append(action)

        self.text_edit.append("<span style='color: green;'>\n智能体推理链：</span>")
        self.text_edit.append(f"<span style='color: black;'>{thoughts}\n</span>")

        self.text_edit.append("<span style='color: red;'>\n智能体态势预测：</span>")
        self.text_edit.append(f"<span style='color: black;'>{prediction}\n</span>")

        self.text_edit.append("<span style='color: green;'>智能体执行动作：</span>")

        if self.model_id == self.last_model_id:
            self.text_edit.append(f"<span style='color: black;'>维持在算法 {self.model_id} {ALGO_DICT[self.model_id]}</span>")
        else:
            self.text_edit.append(f"<span style='color: black;'>切换到算法 {self.model_id} {ALGO_DICT[self.model_id]}</span>")
            self.last_model_id = self.model_id

        if self.device == self.last_device:
            self.text_edit.append(f"<span style='color: black;'>维持在设备 {self.device}</span>")
        else:
            self.text_edit.append(f"<span style='color: black;'>切换到设备 {self.device}</span>")
            self.last_device = self.device

        self.text_edit.append("<span style='color: blue;'>\nPOST http://10.3.242.47:8074/sim STATE:200</span>")

        with open(self.text_log_file, 'w') as f:
            f.write(self.text_edit.toPlainText().replace('\n', ' ') + '\n')

        for i, queue in enumerate(self.queues):
            queue.append(float(state[self.ref_list[i]]))
            self.ax_bars[i].cla()
            self.ax_bars[i].bar(range(len(queue)), list(queue))
            self.ax_bars[i].set_title(self.ref_name_list[i])
            self.ax_bars[i].axhline(y=self.baseline_values[self.ref_list[i]], color='r', linestyle='--')
        self.canvas_bar.draw()

        accuracy = 1 if (gt - self.model_id) % 2 == 0 else 0
        self.accuracy_list.append(accuracy)
        cumulative_accuracy = np.cumsum(self.accuracy_list) / (np.arange(len(self.accuracy_list)) + 1)

        self.ax_acc.cla()
        self.ax_acc.plot(range(len(cumulative_accuracy)), cumulative_accuracy, label='Cumulative Accuracy', color='b')
        self.ax_acc.set_ylim(0, 1.1)
        self.ax_acc.set_title("Cumulative Accuracy")
        self.ax_acc.legend()
        self.canvas_acc.draw()

        with open(self.bar_log_file, 'a') as f:
            bar_list = [str(float(state[ref])) for ref in self.ref_list]
            bar_list.append(str(cumulative_accuracy[-1]))
            f.write(','.join(bar_list) + '\n')
        
        nodes = [
            WebNode("BASE", 0, {'SNR': float(state["snr"])}),
            WebNode("UAV", 1, {'SNR': float(state["snr"]), 'device': "gpu" if state["gpu"] else "cpu"})
        ]
        self.update_topology(nodes)

    def update_topology(self, nodes):
        self.ax_topo.cla()
        G = nx.Graph()
        labels = {}
        
        for node in nodes:
            G.add_node(node.id)
            labels[node.id] = f'{node.name} {node.id}\n' + '\n'.join([f'{k}: {v}' for k, v in node.attributes.items()])

        G.add_edge(nodes[0].id, nodes[1].id)

        pos = {0: np.zeros(2) + 0.5, 1: np.random.rand(2)} 
        nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=2500, alpha=0.7, ax=self.ax_topo)
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=self.ax_topo)

        self.ax_topo.set_xlim([-0.1, 1.1])
        self.ax_topo.set_ylim([-0.1, 1.1])
        self.canvas_topo.draw()


def main():
    app = QApplication(sys.argv)
    dialog = StartupDialog()

    if dialog.exec_() == QDialog.Accepted:
        intent, dist_id = dialog.get_inputs()
        window = MainWindow(intent, dist_id)
        window.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()