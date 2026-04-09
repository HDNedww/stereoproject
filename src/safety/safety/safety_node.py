#!/usr/bin/env python3
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32, String, UInt8, Bool


class SafetyNode(Node):
    """
    States:
      0 SAFE
      1 WARNING
      2 DANGEROUS
    """

    def __init__(self):
        super().__init__("safety_node")

        self.declare_parameter("z_topic", "/depth/person_z")
        self.declare_parameter("detected_topic", "/depth/person_detected")
        self.declare_parameter("depth_valid_topic", "/depth/depth_valid")
        self.declare_parameter("in_zone_topic", "/depth/in_zone")

        self.declare_parameter("out_state_topic", "/safety/state")
        self.declare_parameter("out_level_topic", "/safety/level")

        self.declare_parameter("danger_m", 0.8)
        self.declare_parameter("warning_m", 1.5)

        self.declare_parameter("invalid_zone_frames_for_danger", 3)
        self.declare_parameter("min_publish_hz", 10.0)
        self.declare_parameter("qos_reliability", "best_effort")  # reliable | best_effort
        self.declare_parameter("qos_depth", 10)

        self.z_topic = str(self.get_parameter("z_topic").value)
        self.detected_topic = str(self.get_parameter("detected_topic").value)
        self.depth_valid_topic = str(self.get_parameter("depth_valid_topic").value)
        self.in_zone_topic = str(self.get_parameter("in_zone_topic").value)

        self.out_state_topic = str(self.get_parameter("out_state_topic").value)
        self.out_level_topic = str(self.get_parameter("out_level_topic").value)

        self.danger_m = float(self.get_parameter("danger_m").value)
        self.warning_m = float(self.get_parameter("warning_m").value)
        self.invalid_zone_frames_for_danger = int(self.get_parameter("invalid_zone_frames_for_danger").value)
        self.min_publish_hz = float(self.get_parameter("min_publish_hz").value)
        self.qos_reliability = str(self.get_parameter("qos_reliability").value).strip().lower()
        self.qos_depth = int(self.get_parameter("qos_depth").value)
        self.qos = self._build_qos()

        self.pub_state = self.create_publisher(String, self.out_state_topic, self.qos)
        self.pub_level = self.create_publisher(UInt8, self.out_level_topic, self.qos)

        self.sub_z = self.create_subscription(Float32, self.z_topic, self.on_z, self.qos)
        self.sub_detected = self.create_subscription(Bool, self.detected_topic, self.on_detected, self.qos)
        self.sub_depth_valid = self.create_subscription(Bool, self.depth_valid_topic, self.on_depth_valid, self.qos)
        self.sub_in_zone = self.create_subscription(Bool, self.in_zone_topic, self.on_in_zone, self.qos)

        self.person_detected = False
        self.depth_valid = False
        self.in_zone = False
        self.last_z = -1.0

        self.invalid_in_zone_count = 0

        self.last_state = None
        self.last_level = None
        self.last_pub_t = self.get_clock().now()

        self.input_count = 0
        self.output_count = 0
        self.stats_start_t = time.time()
        self.last_debug_print_t = time.time()

        self.get_logger().info(
            f"Safety node started | z={self.z_topic} detected={self.detected_topic} "
            f"depth_valid={self.depth_valid_topic} in_zone={self.in_zone_topic} "
            f"| qos={self.qos_reliability}"
        )

    def _build_qos(self):
        reliability = ReliabilityPolicy.RELIABLE
        if self.qos_reliability == "best_effort":
            reliability = ReliabilityPolicy.BEST_EFFORT
        elif self.qos_reliability != "reliable":
            self.get_logger().warn(f"Unknown qos_reliability='{self.qos_reliability}', using 'reliable'")
            self.qos_reliability = "reliable"

        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, self.qos_depth),
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
        )

    def on_detected(self, msg: Bool):
        self.person_detected = bool(msg.data)

    def on_depth_valid(self, msg: Bool):
        self.depth_valid = bool(msg.data)

    def on_in_zone(self, msg: Bool):
        self.in_zone = bool(msg.data)

    def evaluate_state(self, z: float):
        if not self.person_detected:
            self.invalid_in_zone_count = 0
            return "SAFE", 0

        if self.depth_valid and math.isfinite(z) and z > 0.0:
            self.invalid_in_zone_count = 0

            if z < self.danger_m:
                return "DANGEROUS", 2
            if z < self.warning_m:
                return "WARNING", 1
            return "SAFE", 0

        # person detected but depth invalid
        if not self.in_zone:
            self.invalid_in_zone_count = 0
            return "SAFE", 0

        self.invalid_in_zone_count += 1
        if self.invalid_in_zone_count >= self.invalid_zone_frames_for_danger:
            return "DANGEROUS", 2
        return "WARNING", 1

    def publish(self, state: str, level: int, z: float):
        msg_s = String()
        if self.depth_valid and math.isfinite(z) and z > 0.0:
            msg_s.data = f"{state} (z={z:.2f}m)"
        else:
            msg_s.data = f"{state} (z=invalid)"
        self.pub_state.publish(msg_s)

        msg_l = UInt8()
        msg_l.data = int(level)
        self.pub_level.publish(msg_l)

        self.output_count += 1

    def print_debug_status(self, state: str):
        now = time.time()
        elapsed = now - self.stats_start_t
        if elapsed <= 0.0:
            return

        if now - self.last_debug_print_t >= 1.0:
            input_fps = self.input_count / elapsed
            output_fps = self.output_count / elapsed

            z_text = f"{self.last_z:.2f} m" if self.depth_valid and self.last_z > 0.0 else "invalid"

            self.get_logger().info(
                f"detected={'YES' if self.person_detected else 'NO'} | "
                f"in_zone={'YES' if self.in_zone else 'NO'} | "
                f"depth_valid={'YES' if self.depth_valid else 'NO'} | "
                f"distance={z_text} | state={state} | "
                f"invalid_zone_frames={self.invalid_in_zone_count} | "
                f"input_fps={input_fps:.2f} | output_fps={output_fps:.2f}"
            )

            self.input_count = 0
            self.output_count = 0
            self.stats_start_t = now
            self.last_debug_print_t = now

    def on_z(self, msg: Float32):
        self.input_count += 1

        z = float(msg.data)
        self.last_z = z

        state, level = self.evaluate_state(z)

        now = self.get_clock().now()
        dt = (now - self.last_pub_t).nanoseconds / 1e9

        changed = (state != self.last_state) or (level != self.last_level)
        rate_due = dt >= (1.0 / max(1e-6, self.min_publish_hz))

        if changed or rate_due:
            self.publish(state, level, z)
            self.last_state = state
            self.last_level = level
            self.last_pub_t = now

        self.print_debug_status(state)


def main():
    rclpy.init()
    node = SafetyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
