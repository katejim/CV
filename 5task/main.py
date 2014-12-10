import numpy as np
import cv2

VIDEO_PATH = "video.mpg"
HARRIS_OUTPUT = "harrisRes.mpg"
FAST_OUTPUT = "fastRes.mpg"
POINTS_COUNT = 150


def harris(frame):
    return cv2.goodFeaturesToTrack(image=frame, maxCorners=POINTS_COUNT, qualityLevel=0.15,
                                   minDistance=7, blockSize=2, useHarrisDetector=True)


def fast(frame):
    fast = cv2.FastFeatureDetector().detect(frame, None)
    fast = list(sorted(fast, key=lambda x: x.response, reverse=True))[:POINTS_COUNT]
    return np.array([[kp.pt] for kp in fast], np.float32)


def open(input_video, output_video):
    video = cv2.VideoCapture(input_video)
    width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.cv.CV_CAP_PROP_FPS))
    return video, cv2.VideoWriter(output_video, cv2.cv.CV_FOURCC('m', 'p', '4', 'v'), fps, (width, height))


def processVideo(output_filename, detector):
    color = np.random.randint(0, 255, (POINTS_COUNT, 3))
    lk_params = {'winSize': (10, 10), 'maxLevel': 3,
                 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)}

    cap, videoWriter = open(VIDEO_PATH, output_filename)
    ret, frame = cap.read()

    oldGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    oldFeatures = detector(oldGray)

    videoWriter.write(frame)
    flow = np.zeros_like(frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        newFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        newFeatures, status, err = cv2.calcOpticalFlowPyrLK(oldGray, newFrame, oldFeatures, None, **lk_params)

        goodNew = newFeatures[status == 1]
        goodOld = oldFeatures[status == 1]

        for i, (new, old) in enumerate(zip(goodNew, goodOld)):
            a, b = new.ravel()
            cv2.circle(frame, (a, b), 4, color[i].tolist(), -1)

        videoWriter.write(cv2.add(frame, flow))
        oldGray = newFrame.copy()
        oldFeatures = goodNew.reshape(-1, 1, 2)

    videoWriter.release()
    cap.release()


processVideo(HARRIS_OUTPUT, harris)
print("Harris done")
processVideo(FAST_OUTPUT, fast)
print("FAST done")
