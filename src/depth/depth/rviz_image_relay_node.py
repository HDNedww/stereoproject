#!/usr/bin/env python3
from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image


class RvizImageRelayNode(Node):
    def __init__(self):
        super().__init__("rviz_image_relay_node")

        self.declare_parameter("in_intensity_topic", "/camera/intensity_rgb")
        self.declare_parameter("out_intensity_topic", "/rviz/camera/intensity_rgb")
        self.declare_parameter("in_intensity_mono_topic", "/camera/intensity")
        self.declare_parameter("out_intensity_mono_topic", "/rviz/camera/intensity")
        self.declare_parameter("in_disparity_topic", "/camera/disparity")
        self.declare_parameter("out_disparity_topic", "/rviz/camera/disparity")
        self.declare_parameter("in_colormap_topic", "/depth/colormap")
        self.declare_parameter("out_colormap_topic", "/rviz/depth/colormap")
        self.declare_parameter("in_det_viz_topic", "/det/viz")
        self.declare_parameter("out_det_viz_topic", "/rviz/det/viz")
        self.declare_parameter("in_depth_topic", "/depth/image_m")
        self.declare_parameter("out_depth_topic", "/rviz/depth/image_m")
        self.declare_parameter("relay_depth_image", False)
        self.declare_parameter("qos_depth", 10)

        in_intensity_topic = str(self.get_parameter("in_intensity_topic").value)
        out_intensity_topic = str(self.get_parameter("out_intensity_topic").value)
        in_intensity_mono_topic = str(self.get_parameter("in_intensity_mono_topic").value)
        out_intensity_mono_topic = str(self.get_parameter("out_intensity_mono_topic").value)
        in_disparity_topic = str(self.get_parameter("in_disparity_topic").value)
        out_disparity_topic = str(self.get_parameter("out_disparity_topic").value)
        in_colormap_topic = str(self.get_parameter("in_colormap_topic").value)
        out_colormap_topic = str(self.get_parameter("out_colormap_topic").value)
        in_det_viz_topic = str(self.get_parameter("in_det_viz_topic").value)
        out_det_viz_topic = str(self.get_parameter("out_det_viz_topic").value)
        in_depth_topic = str(self.get_parameter("in_depth_topic").value)
        out_depth_topic = str(self.get_parameter("out_depth_topic").value)
        relay_depth_image = bool(self.get_parameter("relay_depth_image").value)
        qos_depth = int(self.get_parameter("qos_depth").value)

        self.qos_sub = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, qos_depth),
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.qos_pub = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, qos_depth),
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.pub_intensity = self.create_publisher(Image, out_intensity_topic, self.qos_pub)
        self.pub_intensity_mono = self.create_publisher(Image, out_intensity_mono_topic, self.qos_pub)
        self.pub_disparity = self.create_publisher(Image, out_disparity_topic, self.qos_pub)
        self.pub_colormap = self.create_publisher(Image, out_colormap_topic, self.qos_pub)
        self.pub_det_viz = self.create_publisher(Image, out_det_viz_topic, self.qos_pub)
        self.sub_intensity = self.create_subscription(
            Image, in_intensity_topic, partial(self._relay_cb, self.pub_intensity), self.qos_sub
        )
        self.sub_intensity_mono = self.create_subscription(
            Image, in_intensity_mono_topic, partial(self._relay_cb, self.pub_intensity_mono), self.qos_sub
        )
        self.sub_disparity = self.create_subscription(
            Image, in_disparity_topic, partial(self._relay_cb, self.pub_disparity), self.qos_sub
        )
        self.sub_colormap = self.create_subscription(
            Image, in_colormap_topic, partial(self._relay_cb, self.pub_colormap), self.qos_sub
        )
        self.sub_det_viz = self.create_subscription(
            Image, in_det_viz_topic, partial(self._relay_cb, self.pub_det_viz), self.qos_sub
        )

        if relay_depth_image:
            self.pub_depth = self.create_publisher(Image, out_depth_topic, self.qos_pub)
            self.sub_depth = self.create_subscription(
                Image, in_depth_topic, partial(self._relay_cb, self.pub_depth), self.qos_sub
            )
        else:
            self.pub_depth = None
            self.sub_depth = None

        self.get_logger().info(
            f"RViz relay started: {in_intensity_topic} -> {out_intensity_topic}, "
            f"{in_intensity_mono_topic} -> {out_intensity_mono_topic}, "
            f"{in_disparity_topic} -> {out_disparity_topic}, "
            f"{in_colormap_topic} -> {out_colormap_topic}, "
            f"{in_det_viz_topic} -> {out_det_viz_topic}, "
            f"depth relay={'on' if relay_depth_image else 'off'}"
        )

    def _relay_cb(self, pub, msg: Image):
        pub.publish(msg)


def main():
    rclpy.init()
    node = RvizImageRelayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
