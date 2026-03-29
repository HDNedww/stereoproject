#!/usr/bin/env python3
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import gi
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis


class RcVisardCameraNode(Node):
    def __init__(self):
        super().__init__("rc_visard_camera_node")

        self.declare_parameter("intensity_topic", "/camera/intensity")
        self.declare_parameter("disparity_topic", "/camera/disparity")
        self.declare_parameter("n_buffers", 16)
        self.declare_parameter("timeout_us", 200_000)
        self.declare_parameter("drain_max", 6)

        self.intensity_topic = str(self.get_parameter("intensity_topic").value)
        self.disparity_topic = str(self.get_parameter("disparity_topic").value)
        self.n_buffers = int(self.get_parameter("n_buffers").value)
        self.timeout_us = int(self.get_parameter("timeout_us").value)
        self.drain_max = int(self.get_parameter("drain_max").value)

        self.pub_i = self.create_publisher(Image, self.intensity_topic, qos_profile_sensor_data)
        self.pub_d = self.create_publisher(Image, self.disparity_topic, qos_profile_sensor_data)
        self.bridge = CvBridge()

        self.cam, self.dev, self.stream = self._init_aravis()
        self.timer = self.create_timer(0.0, self._loop)

        self.get_logger().info(f"Publishing intensity -> {self.intensity_topic}")
        self.get_logger().info(f"Publishing disparity -> {self.disparity_topic}")

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

        dev.set_string_feature_value("ComponentSelector", "Intensity")
        dev.set_boolean_feature_value("ComponentEnable", True)
        dev.set_string_feature_value("ComponentSelector", "Disparity")
        dev.set_boolean_feature_value("ComponentEnable", True)
    
        payload = int(dev.get_integer_feature_value("PayloadSize"))
        stream = cam.create_stream(None, None)
        for _ in range(self.n_buffers):
            stream.push_buffer(Aravis.Buffer.new_allocate(payload))

        cam.start_acquisition()
        time.sleep(0.2)
        return cam, dev, stream

    def _pop_latest(self):
        buf = self.stream.timeout_pop_buffer(self.timeout_us)
        if buf is None or buf.get_status() != Aravis.BufferStatus.SUCCESS:
            if buf is not None:
                self.stream.push_buffer(buf)
            return None

        latest = buf
        for _ in range(self.drain_max):
            b2 = self.stream.try_pop_buffer()
            if b2 is None:
                break
            if b2.get_status() == Aravis.BufferStatus.SUCCESS:
                self.stream.push_buffer(latest)
                latest = b2
            else:
                self.stream.push_buffer(b2)
        return latest

    def _loop(self):
        buf = self._pop_latest()
        if buf is None:
            return

        w = buf.get_image_width()
        h = buf.get_image_height()

        if not hasattr(self, "_printed_resolution"):
            self.get_logger().info(f"Incoming stream resolution: {w}x{h}")
            self._printed_resolution = True

        data = bytes(buf.get_data())
        self.stream.push_buffer(buf)

        n = len(data)
        mono_expected = w * h
        disp_expected = w * h * 2
        stamp = self.get_clock().now().to_msg()

        # disparity u16
        if n >= disp_expected and (n - disp_expected) < 65536:
            raw = data[:disp_expected]
            disp_u16 = np.frombuffer(raw, dtype=np.uint16).reshape((h, w))
            msg = self.bridge.cv2_to_imgmsg(disp_u16, encoding="mono16")
            msg.header.stamp = stamp
            msg.header.frame_id = "rc_visard_left"
            self.pub_d.publish(msg)
            return

        # intensity mono8
        if n >= mono_expected and (n - mono_expected) < 65536:
            raw = data[:mono_expected]
            img8 = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))
            msg = self.bridge.cv2_to_imgmsg(img8, encoding="mono8")
            msg.header.stamp = stamp
            msg.header.frame_id = "rc_visard_left"
            self.pub_i.publish(msg)
            return
    def destroy_node(self):
        try:
            self.cam.stop_acquisition()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = RcVisardCameraNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
