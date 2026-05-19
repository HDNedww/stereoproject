# camera/camera/rc_visard_intensity_node.py
import time
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import gi
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis


class RcVisardIntensityNode(Node):
    def __init__(self):
        super().__init__("rc_visard_intensity_node")

        self.declare_parameter("topic", "/camera/intensity")
        self.declare_parameter("n_buffers", 16)
        self.declare_parameter("timeout_us", 200_000)

        self.topic = self.get_parameter("topic").value
        self.n_buffers = int(self.get_parameter("n_buffers").value)
        self.timeout_us = int(self.get_parameter("timeout_us").value)

        self.pub = self.create_publisher(Image, self.topic, 10)
        self.bridge = CvBridge()

        self.cam, self.dev, self.stream = self._init_aravis()
        self.timer = self.create_timer(0.0, self._loop)  # "as fast as possible"

        self.get_logger().info(f"Publishing rc_visard intensity -> {self.topic}")

    def _find_device(self):
        Aravis.enable_interface("GigEVision")
        Aravis.update_device_list()
        for i in range(Aravis.get_n_devices()):
            dev_id = Aravis.get_device_id(i)
            if "rc_visard" in dev_id.lower():
                return dev_id
        return None

    def _init_aravis(self):
        dev_id = self._find_device()
        if dev_id is None:
            raise RuntimeError("rc_visard not found via Aravis")

        self.get_logger().info(f"Using device: {dev_id}")
        cam = Aravis.Camera.new(dev_id)
        dev = cam.get_device()

        # įjungiam Intensity komponentą
        dev.set_string_feature_value("ComponentSelector", "Intensity")
        dev.set_boolean_feature_value("ComponentEnable", True)

        payload = int(dev.get_integer_feature_value("PayloadSize"))
        stream = cam.create_stream(None, None)
        for _ in range(self.n_buffers):
            stream.push_buffer(Aravis.Buffer.new_allocate(payload))

        cam.start_acquisition()
        time.sleep(0.2)
        return cam, dev, stream

    def _loop(self):
        buf = self.stream.timeout_pop_buffer(self.timeout_us)
        if buf is None or buf.get_status() != Aravis.BufferStatus.SUCCESS:
            if buf is not None:
                self.stream.push_buffer(buf)
            return

        w = buf.get_image_width()
        h = buf.get_image_height()
        data = bytes(buf.get_data())
        self.stream.push_buffer(buf)

        expected = w * h
        if len(data) < expected:
            return

        img8 = np.frombuffer(data[:expected], dtype=np.uint8).reshape((h, w))
        msg = self.bridge.cv2_to_imgmsg(img8, encoding="mono8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "rc_visard_left"
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            self.cam.stop_acquisition()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = RcVisardIntensityNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
