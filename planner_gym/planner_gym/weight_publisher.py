import rclpy

from rclpy.node import Node
from typing import Optional
from planner_interfaces.msg import DynamicWeights


class WeightPublisher(Node):
    def __init__(self):
        super().__init__('weight_publisher')

        self.weights: Optional[DynamicWeights] = None

        self.publisher = self.create_publisher(
            DynamicWeights,
            '/dynamic_weights',
            10
        )

        self.timer = self.create_timer(
            0.5,
            self.timer_callback
        )

    # Init publish to remove timer delay
    def set_weights(self, weights: DynamicWeights):
        self.weights = weights
        self.publisher.publish(self.weights)

    def timer_callback(self):
        if self.weights is not None:
            self.publisher.publish(self.weights)