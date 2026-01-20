import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "./runs/pose/train24/weights/best.pt"
VIDEO_PATH = "real44.mp4"
OUT_VIDEO_PATH = "mobilenet.mp4"

# 相机内参（录视频的相机的内参）
K = np.array([
    [1784.4272351926288, 0.0, 734.66324795640344],
    [0.0, 1785.7686229320641, 567.64643472768648],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST = np.array([-0.04244, -0.20965, 0.00270, -0.00066, 0.91215], dtype=np.float64)

OBJECT_POINTS = np.array([
    [0.0,   -125.0, 0.0],
    [125.0,    0.0, 0.0],
    [0.0,    125.0, 0.0],
    [-125.0,   0.0, 0.0],
    [0.0,    194.5, 0.0],
    [0.0,    524.5, 0.0],
], dtype=np.float32)


AXIS_LEN = 100.0


def draw_axis(img, rvec, tvec, K, dist):
    axis_3d = np.float32([
        [0, 0, 0],
        [AXIS_LEN, 0, 0],
        [0, AXIS_LEN, 0],
        [0, 0, -AXIS_LEN],
    ])

    imgpts, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, dist)
    p0, px, py, pz = imgpts.reshape(-1, 2).astype(int)

    cv2.line(img, tuple(p0), tuple(px), (0, 0, 255), 2)     # X 红
    cv2.line(img, tuple(p0), tuple(py), (0, 255, 0), 2)     # Y 绿
    cv2.line(img, tuple(p0), tuple(pz), (255, 0, 0), 2)     # Z 蓝


def check(image_points, w, h):
    if image_points is None:
        return False
    if image_points.shape != (6, 2):
        return False

    xs = image_points[:, 0]
    ys = image_points[:, 1]

    if np.any(xs <= 0) or np.any(ys <= 0):
        return False
    if np.any(xs >= w) or np.any(ys >= h):
        return False

    return True


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("视频打不开")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps and fps > 0 else 30.0,
        (W, H)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        frame_ok = False

        if len(results) > 0 and results[0].keypoints is not None:
            kpts = results[0].keypoints.xy.cpu().numpy()  # [N,K,2]

            for kp in kpts:
                if kp.shape[0] < 6:
                    continue
                image_points = kp[:6].astype(np.float32)

                if not check(image_points, W, H):
                    continue

                ok, rvec, tvec = cv2.solvePnP(
                    OBJECT_POINTS,
                    image_points,
                    K,
                    DIST,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    continue

                frame_ok = True

                proj_pts, _ = cv2.projectPoints(OBJECT_POINTS, rvec, tvec, K, DIST)
                proj_pts = proj_pts.reshape(-1, 2)

                # 画检测点（绿）
                for p in image_points:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

                # 画重投影点（红）
                for p in proj_pts:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)

                # 画坐标轴
                draw_axis(frame, rvec, tvec, K, DIST)

        cv2.imshow("pose_check", frame)
        if frame_ok:
            writer.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
