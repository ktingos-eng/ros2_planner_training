import rclpy

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from planner_interfaces.msg import Metrics
from message_filters import Subscriber, ApproximateTimeSynchronizer

class SyncSubscriber(Node):
    def __init__(self):
        super().__init__('sync_subscriber')

        self.declare_parameter('sync_metrics', True)

        self.sync_metrics = self.get_parameter(
            'sync_metrics'
        ).get_parameter_value().bool_value

        self.scan = None
        self.odom = None
        self.metrics = None

        self._scan_sub = Subscriber(self, LaserScan, '/scan/normalized')
        self._odom_sub = Subscriber(self, Odometry, '/odom')
        self._metrics_sub = Subscriber(self, Metrics, '/planner_metrics')

        if self.sync_metrics:
            self._ats = ApproximateTimeSynchronizer(
                [self._scan_sub, self._odom_sub, self._metrics_sub],
                queue_size=10,
                slop=0.1
            )
            self._ats.registerCallback(self.sync_global_callback)
        else:
            self._ats = ApproximateTimeSynchronizer(
                [self._scan_sub, self._odom_sub],
                queue_size=10,
                slop=0.1
            )
            self._ats.registerCallback(self.sync_scan_odom_callback)

        self.get_logger().info("Synchronized Callback registered.")

    def sync_global_callback(self, scan_msg, odom_msg, metrics_msg):
        self.scan = scan_msg
        self.odom = odom_msg
        self.metrics = metrics_msg

    def sync_scan_odom_callback(self, scan_msg, odom_msg):
        self.scan = scan_msg
        self.odom = odom_msg