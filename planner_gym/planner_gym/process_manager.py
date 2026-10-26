import subprocess as sp
import os
import signal
import rclpy
import time

from rclpy.node import Node

class ProcessManager(Node):
    def __init__(self):
        super().__init__("process_manager")
        self._bringup = None

        self._expected_nodes = {
            "/Ros2Supervisor/Ros2Supervisor",
            "/SimulationSupervisor",
            "/Turtlebot3Waffle",
            "/costmap/costmap",
            "/joint_state_publisher",
            "/lifecycle_manager",
            "/robot_state_publisher",
            "/rviz2",
        }

    def run_bringup(self) -> bool:
        if(self._bringup is None or self._bringup.poll() is not None):
            try:
                self._bringup = sp.Popen(
                    ["ros2", "launch", "tbot3_webots", "bringup.launch.py"],
                    start_new_session=True
                )
            except OSError:
                self._bringup = None
                return False
        else:
            return True

        timeout = 30.0
        start = time.monotonic()

        try:
            while time.monotonic() - start < timeout:
                node_list = self._get_node_list(
                    self.get_node_names_and_namespaces()
                )

                if self._expected_nodes.issubset(node_list):
                    return True
                                
                time.sleep(0.1)

            print("Run timed-out, returning False")
            return False
        except:
            self._bringup = None
            return False
            

    def kill_bringup(self) -> bool:
        if self._bringup is None: 
            return True
        elif self._bringup.poll() is None:
            try:
                self._bringup.send_signal(signal.SIGINT)
                self._bringup.wait(timeout=10.0)
            except OSError:
                self._bringup = None
                return False

        timeout = 30.0
        start = time.monotonic()

        try:
            while time.monotonic() - start < timeout:
                node_list = self._get_node_list(
                    self.get_node_names_and_namespaces()
                )

                if len(self._expected_nodes.intersection(node_list)) == 0:
                    return True

                time.sleep(0.1)

            print("Kill timed-out, returning False")
            return False      
        except:
            self._bringup = None
            return False

    def _get_node_list(self, nodes_ns: list[tuple[str, str]]) -> set:
        node_list = set()
        for name, ns in nodes_ns:
            node_list.add(f"{ns.rstrip('/')}/{name}")

        return node_list