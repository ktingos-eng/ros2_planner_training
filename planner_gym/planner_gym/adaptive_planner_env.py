import rclpy
import threading
import numpy as np
import gymnasium as gym

from typing import Optional
from rclpy.executors import MultiThreadedExecutor
from planner_gym.sync_subscriber import SyncSubscriber
from planner_gym.process_manager import ProcessManager

class AdaptivePlannerEnv(gym.Env):

    def __init__(self):
        super().__init__()

        rclpy.init(args=None)

        self.ss_node = SyncSubscriber()
        self.pm_node = ProcessManager()

        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.ss_node)
        self.executor.add_node(self.pm_node)

        self.executor_thread = threading.Thread(
            target = self.executor.spin,
            daemon = True
        )

        self.executor_thread.start()

        print("ENV DONE INIT")