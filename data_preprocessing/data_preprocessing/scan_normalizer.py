import rclpy
import math
import copy
from rclpy.node import Node
from rclpy import qos

from sensor_msgs.msg import LaserScan

class ScanNormalizer(Node):

    def __init__(self):
        super().__init__('scan_normalizer')

        self.max_range = float(10.0)
        self.scan = LaserScan()
        self.scan_norm = LaserScan()

        self.scan = None

        scan_qos = qos.QoSProfile(
            history = qos.HistoryPolicy.KEEP_LAST,
            depth = 5,
            reliability = qos.ReliabilityPolicy.RELIABLE,
            durability = qos.DurabilityPolicy.VOLATILE
        )

        self.pub = self.create_publisher(
            LaserScan,
            '/scan/normalized',
            scan_qos
        )

        self.sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            scan_qos
        )

        self.wall_timer = self.create_timer(
            0.5,
            self.timer_callback
        )

    def scan_callback(self, msg):
        self.scan = msg

    def timer_callback(self):
        if self.scan == None:
            return
        
        self.scan_norm = copy.deepcopy(self.scan)
        self.scan_norm.ranges = self.normalize_range(self.scan_norm.ranges)

        self.pub.publish(self.scan_norm)

        

    def normalize_range(self, ranges: list[float]) -> list[float]:
        for i in range(0,len(ranges)):
            if math.isinf(ranges[i]):
                ranges[i] = 1.0
            else:
                ranges[i] = ranges[i] / self.max_range
        return ranges


def main(args=None):
    rclpy.init(args=args)

    node = ScanNormalizer()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__=='__main__':
    main()