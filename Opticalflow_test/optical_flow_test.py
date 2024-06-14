import cv2
import numpy as np

# Lukas Kanade params
lk_params = dict(winSize = (100, 100),
                 maxLevel = 9,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 10, 0.03))

# Mouse selection
def select_point(event, x, y, flags, params):
    global points, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y)) 
        point_selected = True
        old_points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)

point_selected = False
points = []
old_points = np.array([[]], dtype=np.float32).reshape(-1, 1, 2)

video_file = "X:\Life\TFG\Coding\Testing\Videos\BikeBi\VID_20220511_132940.mp4"

cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    print(f"Error: Could not open video file: {video_file}")
    #return None

old_gray = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if point_selected is True:
        for point in points:
            x_t, y_t = point
            cv2.circle(frame, (int(x_t), int(y_t)), 5, (0, 0, 255), 2)
        if old_gray is None:
            old_gray = gray_frame.copy()
        new_points,status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()

        # Update points and draw the tracking
        if new_points is not None and status is not None:
            new_points = new_points.reshape(-1, 2)
            status = status.reshape(-1)

            good_new = new_points[status == 1]
            good_old = old_points.reshape(-1, 2)[status == 1]

            old_points = good_new.reshape(-1, 1, 2)
            points = [tuple(pt) for pt in good_new]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                x, y = new.ravel()
                #print(new)
                cv2.circle(frame, (int(x),int(y)), 5, (0,255,0), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(120)

    if key == ord("q"):
        break
    elif key == ord("d"):
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
