import cv2
import numpy as np

# Lukas Kanade params
lk_params = dict(winSize=(200, 200),
                 maxLevel=9,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 10, 0.03))

def interpolate_points(p1, p2, num_points):
    points = []
    for i in range(num_points):
        alpha = i / (num_points - 1)
        x = (1 - alpha) * p1[0] + alpha * p2[0]
        y = (1 - alpha) * p1[1] + alpha * p2[1]
        points.append((x, y))
    return points

def select_points(event, x, y, flags, params):
    global points, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            if len(points) == 2:
                point_selected = True
                line1_points = interpolate_points(points[0], points[1], num_points_per_line)
                #line2_points = interpolate_points(points[2], points[3], num_points_per_line)
                all_points = line1_points #+ line2_points
                old_points = np.array(all_points, dtype=np.float32).reshape(-1, 1, 2)

num_points_per_line = 200
points = []
point_selected = False
old_points = np.array([[]], dtype=np.float32).reshape(-1, 1, 2)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_points)

video_file = "X:/Life/TFG/Coding/Testing/Videos/BikeBi/VID_20220427_143651.mp4"
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_file}")

old_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if point_selected:
        if old_gray is None:
            old_gray = gray_frame.copy()
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()

        if new_points is not None and status is not None:
            new_points = new_points.reshape(-1, 2)
            status = status.reshape(-1)

            good_new = new_points[status == 1]
            good_old = old_points.reshape(-1, 2)[status == 1]

            if len(good_new) == len(good_old):
                old_points = good_new.reshape(-1, 1, 2)
                line1_points = good_new[:num_points_per_line]
                line2_points = good_new[num_points_per_line:]

                for i in range(len(line1_points) - 1):
                    x1, y1 = line1_points[i]
                    x2, y2 = line1_points[i + 1]
                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # for i in range(len(line2_points) - 1):
                #     x1, y1 = line2_points[i]
                #     x2, y2 = line2_points[i + 1]
                #     cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(30)
    if key == ord("q"):
        break
    elif key == ord("d"):
        points = []
        while True:
            for point in points:
                x_t, y_t = point
                cv2.circle(frame, (int(x_t), int(y_t)), 5, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(120)
            if key == ord("c"):
                break

cap.release()
cv2.destroyAllWindows()
