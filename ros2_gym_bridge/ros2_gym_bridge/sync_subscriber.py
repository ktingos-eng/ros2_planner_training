import rclpy
import time

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from planner_interfaces.msg import Metrics
from message_filters import Subscriber, ApproximateTimeSynchronizer

from typing import Optional

class SyncSubscriber(Node):
    def __init__(self):
        super().__init__('sync_subscriber')

        self.declare_parameter('sync_metrics', False)

        self.sync_metrics = self.get_parameter(
            'sync_metrics'
        ).get_parameter_value().bool_value

        self.last_sub_time = None
        self.timeout = 5.0

        self.scan: Optional[LaserScan] = None
        self.odom: Optional[Odometry] = None
        self.metrics: Optional[Metrics] = None

        self._scan_sub = Subscriber(self, LaserScan, '/scan/normalized')
        self._odom_sub = Subscriber(self, Odometry, '/odom')

        if self.sync_metrics:
            self._metrics_sub = Subscriber(self, Metrics, '/planner_metrics')
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

            self._metrics_sub = self.create_subscription(
                Metrics,
                '/planner_metrics',
                self.metrics_callback,
                10
            )

        self.get_logger().info("Synchronized Callback registered.")

    def sync_global_callback(self, scan_msg, odom_msg, metrics_msg):
        self.scan = scan_msg
        self.odom = odom_msg
        self.metrics = metrics_msg

        self.last_sub_time = time.monotonic()

    def sync_scan_odom_callback(self, scan_msg, odom_msg):
        self.scan = scan_msg
        self.odom = odom_msg

        self.last_sub_time = time.monotonic()

    def metrics_callback(self, msg):
        self.metrics = msg

    @property
    def data_available(self) -> bool:
        if self.last_sub_time is None:
            return False

        return time.monotonic() - self.last_sub_time < self.timeout
        