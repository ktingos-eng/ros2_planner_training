import rclpy

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from planner_interfaces.msg import Metrics
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer

class PlannerGymNode(Node):
    def __init__(self):
        super().__init__('planner_gym_node')

        self.scan = None
        self.odom = None
        self.metrics = None
        
        self.scan_sub = Subscriber(self, LaserScan, '/scan/normalized')
        self.odom_sub = Subscriber(self, Odometry, '/odom')
        self.metrics_sub = Subscriber(self, Metrics, '/planner_metrics')

        ts = TimeSynchronizer([self.scan_sub, self.odom_sub, self.metrics_sub], queue_size=10)
        ts.registerCallback(self.sync_callback)

    def sync_callback(self, scan_msg, odom_msg, metrics_msg):
        self.scan = scan_msg
        self.odom = odom_msg
        self.metrics = metrics_msg
        