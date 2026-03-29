import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge


class VizNode(Node):
    def __init__(self):
        super().__init__("viz_node")
        self.bridge = CvBridge()
        self.latest_dets = []

        self.create_subscription(Image, "/camera/intensity", self.on_image, 10)
        self.create_subscription(Detection2DArray, "/det/persons", self.on_dets, 10)
        self.pub = self.create_publisher(Image, "/det/viz", 10)
        self.get_logger().info("viz_node ready -> /det/viz")

    def on_dets(self, msg: Detection2DArray):
        self.latest_dets = msg.detections

    def on_image(self, msg: Image):
        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            self.get_logger().error(f"decode failed: {e}")
            return

        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

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