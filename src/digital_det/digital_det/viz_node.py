import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge


class VizNode(Node):
    def __init__(self):
        super().__init__("viz_node")
        self.bridge = CvBridge()
        self.latest_dets = []
        self.declare_parameter("input_topic", "/camera/intensity")
        self.declare_parameter("det_topic", "/det/persons")
        self.declare_parameter("output_topic", "/det/viz")
        self.declare_parameter("qos_reliability", "best_effort")  # reliable | best_effort
        self.declare_parameter("qos_depth", 10)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.det_topic = str(self.get_parameter("det_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.qos_reliability = str(self.get_parameter("qos_reliability").value).strip().lower()
        self.qos_depth = int(self.get_parameter("qos_depth").value)
        self.qos = self._build_qos()

        self.create_subscription(Image, self.input_topic, self.on_image, self.qos)
        self.create_subscription(Detection2DArray, self.det_topic, self.on_dets, self.qos)
        self.pub = self.create_publisher(Image, self.output_topic, self.qos)
        self.get_logger().info(f"viz_node ready -> {self.output_topic} | qos={self.qos_reliability}")

    def _build_qos(self):
        reliability = ReliabilityPolicy.RELIABLE
        if self.qos_reliability == "best_effort":
            reliability = ReliabilityPolicy.BEST_EFFORT
        elif self.qos_reliability != "reliable":
            self.get_logger().warn(f"Unknown qos_reliability='{self.qos_reliability}', using 'best_effort'")
            reliability = ReliabilityPolicy.BEST_EFFORT
            self.qos_reliability = "best_effort"

        return QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=max(1, self.qos_depth),
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
        )

    def on_dets(self, msg: Detection2DArray):
        self.latest_dets = msg.detections

    def on_image(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"decode failed: {e}")
            return

        if img is None:
            return
        if len(img.shape) == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            if msg.encoding.lower() == "rgb8":
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                bgr = img
        else:
            self.get_logger().warn(f"Unsupported image shape={img.shape}, encoding={msg.encoding}")
            return

        for det in self.latest_dets:
            cx = int(det.bbox.center.position.x)
            cy = int(det.bbox.center.position.y)
            w  = int(det.bbox.size_x)
            h  = int(det.bbox.size_y)
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2
            score = det.results[0].hypothesis.score if det.results else 0.0
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(bgr, f"person {score:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        out = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        out.header = msg.header
        self.pub.publish(out)


def main():
    rclpy.init()
    node = VizNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
