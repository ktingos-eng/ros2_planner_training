import rclpy
import threading
import math
import yaml
import io
import os
import time
import numpy as np
import gymnasium as gym

from typing import Optional
from rclpy.executors import MultiThreadedExecutor
from planner_gym.sync_subscriber import SyncSubscriber
from planner_gym.process_manager import ProcessManager

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from planner_interfaces.msg import Metrics

from ament_index_python.packages import get_package_share_directory

class AdaptivePlannerEnv(gym.Env):
    def __init__(self):
        super().__init__()

        ######### TBD PARAMS #########
        self.w_step = 0.1
        self.w_min = 0.1
        self.w_max = 10.0
        ##############################

        self.scan : LaserScan = None
        self.odom : Odometry = None
        self.metrics : Metrics = None


        yaml_file = os.path.join(
            get_package_share_directory('ros2_dwa_pso'),
            'config/params.yaml'
        )

        with io.open(yaml_file,'r') as stream:
            params = yaml.safe_load(stream)
            params = params['dwa_pso_planner']['ros__parameters']

        obs_limits = {
            'v_max': params['limits.max_vel.linear'],
            'v_min': 0.0,
            'w_max': params['limits.max_vel.angular'],
            'w_min': -params['limits.max_vel.angular']
        }
        
        weights = {
            'w_head': params['heading_weight'],
            'w_vel': params['velocity_weight'],
            'w_prog': params['progress_weight'],
            'w_clear': params['clearence_weight']
        }

        act_limits = self._calculate_action_limits(weights)

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

        self.observation_space = gym.spaces.Dict(
            {
                'velocity': gym.spaces.Box(
                    low=np.array(
                        [obs_limits['v_min'], obs_limits['w_min']],
                        dtype=np.float32
                    ), 
                    high=np.array(
                        [obs_limits['v_max'], obs_limits['w_max']],
                        dtype=np.float32
                    ),
                    dtype=np.float32 
                ),
                'heading': gym.spaces.Box(
                    low=-math.pi,
                    high=math.pi,
                    shape=(1,),
                    dtype=np.float32
                ),
                'distance': gym.spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32
                ),
                'scan': gym.spaces.Box(
                    low=0.0,
                    high=10.0,
                    shape=(128,),
                    dtype=np.float32
                ),
            }
        )

        self.action_space = gym.spaces.Dict(
            {
                'n_head': gym.spaces.Discrete(
                    n=act_limits['w_head']['max']-act_limits['w_head']['min'],
                    start=act_limits['w_head']['min'],
                    dtype=np.int32
                ),
                'n_vel': gym.spaces.Discrete(
                    n=act_limits['w_vel']['max']-act_limits['w_vel']['min'],
                    start=act_limits['w_vel']['min'],
                    dtype=np.int32
                ),
                'n_prog': gym.spaces.Discrete(
                    n=act_limits['w_prog']['max']-act_limits['w_prog']['min'],
                    start=act_limits['w_prog']['min'],
                    dtype=np.int32
                ),
                'n_clear': gym.spaces.Discrete(
                    n=act_limits['w_clear']['max']-act_limits['w_clear']['min'],
                    start=act_limits['w_clear']['min'],
                    dtype=np.int32
                )
            }
        )

    def _calculate_action_limits(self, w: dict) -> dict:
        # Lowest possible weights: action step s
        # Reduction limit for weight w: -(w/s-1)
        wl = dict()
        for key, value in w.items():
            wl[key] = {
                'min': -int((value - self.w_min) / self.w_step),
                'max': int((self.w_max - value) / self.w_step)
            }
        return wl

    def _get_obs(self):
        self.scan = self.ss_node.scan
        self.odom = self.ss_node.odom
        self.metrics = self.ss_node.metrics

        return {
            'velocity': np.array(
                [self.odom.twist.twist.linear.x, self.odom.twist.twist.angular.z],
                dtype=np.float32
            ),
            'heading': np.array(
                [self.metrics.theta_goal],
                dtype=np.float32
            ),
            'distance': np.array(
                [self.metrics.dis_goal],
                dtype=np.float32
            ),
            'scan': np.array(
                self.scan.ranges,
                dtype=np.float32
            )
        }

    def _get_info(self):

        if (self.odom is None) or (self.metrics is None):
            return None

        return {
            'v': self.odom.twist.twist.linear.x,
            'w': self.odom.twist.twist.angular.z,
            'theta': self.metrics.theta_goal,
            'occupancy': self.metrics.robot_clearence,
            'goal_dis': self.metrics.dis_goal,
            'range_min': min(self.scan.ranges),
        }


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        max_tries = 20
        for attempt in range(max_tries):
            if self.pm_node.kill_planner():
                break

            if attempt < max_tries - 1:
                time.sleep(1)
        else:
            raise RuntimeError(f'Failed to stop planner after {max_tries} attempts.')

        time.sleep(1)
        
        for attempt in range(max_tries):
            if self.pm_node.kill_bringup():
                break

            if attempt < max_tries - 1:
                time.sleep(1)
        else:
            raise RuntimeError(f'Failed to stop bringup after {max_tries} attempts.')

        time.sleep(2)

        for attempt in range(max_tries):
            if self.pm_node.run_bringup():
                break

            if attempt < max_tries - 1:
                time.sleep(1)
        else:
            raise RuntimeError(f'Failed to start bringup after {max_tries} attempts.')

        obs = self._get_obs()
        info = self._get_info()

        return obs, info