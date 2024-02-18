import cv2 as cv
import numpy as np
import math
import argparse

# PARAMETERS
# todo: export to cli arguments
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 200
HOUGH_RHO = 1
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH = 100
HOUGH_MAX_LINE_GAP = 20

def deskew(img, demo):

    canny = cv.Canny(img, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    contours, _ = cv.findContours(
        canny, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    height, width, _ = img.shape
    drawing = np.copy(img)
    largest = np.copy(img)

    # enclosing circle
    maxAreaCircle = 0
    maxCenter = (0, 0)
    maxRadius = 0
    # bounding rectangle
    maxX, maxY, maxW, maxH = 0, 0, 0, 0

    for i in contours:
        c, r = cv.minEnclosingCircle(i)
        x, y, w, h = cv.boundingRect(i)

        center = (int(c[0]), int(c[1]))
        radius = int(r)
        if np.pi * radius ** 2 > maxAreaCircle:
            maxAreaCircle = np.pi * radius ** 2
            maxCenter = center
            maxRadius = radius
            maxX = x
            maxY = y
            maxW = w
            maxH = h

        cv.circle(drawing, center, radius, (0, 0, 255), 2)
        cv.rectangle(drawing, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv.circle(largest, maxCenter, maxRadius, (255, 0, 0), 2)
    cv.rectangle(largest, (maxX, maxY),
                 (maxX + maxW, maxY + maxH), (255, 0, 0), 2)
    if demo:
        cv.imshow("original", img)
        cv.waitKey()
        cv.imshow("largest enclosing circle and bounding rectangle", largest)
        cv.waitKey()

    centerX, centerY = maxCenter[0], maxCenter[1]
    points = np.array(((maxX, maxY), (maxX + maxW, maxY),
                      (maxX, maxY + maxH), (maxX + maxW, maxY + maxH)), dtype=np.float32)

    warpedPoints = np.array(((centerX - maxRadius, centerY - maxRadius), (centerX + maxRadius,
                            centerY-maxRadius), (centerX - maxRadius, centerY+maxRadius), (centerX + maxRadius, centerY + maxRadius)), dtype=np.float32)
    M = cv.getPerspectiveTransform(points, warpedPoints)
    M2 = cv.findHomography(points, warpedPoints,
                           method=cv.RANSAC, ransacReprojThreshold=3.0)

    maxHeight = max(np.amax(warpedPoints, axis=0)[0].astype(
        int), np.amax(points, axis=0)[0].astype(int), height)
    maxWidth = max(np.amax(warpedPoints, axis=0)[1].astype(
        int), np.amax(points, axis=0)[1].astype(int), width)
    aligned_img = cv.warpPerspective(
        img, M, (maxHeight, maxWidth))
    # aligned_img = cv.warpPerspective(img, M2[0], (maxHeight, maxWidth))

    if demo:
        cv.imshow("aligned", aligned_img)
        cv.waitKey()
    return aligned_img


def get_hands(img, demo):
    canny = cv.Canny(img, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)
    lines = cv.HoughLinesP(canny, HOUGH_RHO, np.pi / 180, HOUGH_THRESHOLD,
                           None, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP)
    drawing = np.copy(img)
    print(lines.shape)

    sorted_lines = []

    # ignored_lines_img = np.copy(img)
    # used_lines_img = np.copy(img)
    # height, width, _ = np.shape(img)
    # centerX, centerY = height // 2, width // 2
    # radius = height // 10

    for line in lines:
        print(line)
        x1, y1, x2, y2 = line[0]
        angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
        sorted_lines.append(([x1, y1, x2, y2], angle))

        # todo: verify line passes through estimated center. Maybe not needed if good Canny/Hough thresholds?
        # m = (y2 - y1) // (x2 - x1)
        # c = (y2 - m * y1)
        # dist = ((abs(m * x1 - y1 + c)) /
        #     math.sqrt(m * m + 1))
        # if dist <= radius:
        #     sorted_lines.append(([x1, y1, x2, y2], angle))
        #     cv.line(used_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # else:
        #     cv.line(ignored_lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    sorted_lines.sort(key=lambda x: x[1])

    if demo:
        for line in sorted_lines:
            x1, y1, x2, y2 = line[0]
            cv.line(drawing, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.imshow("lines", drawing)
        cv.waitKey()

    sorted_similar = []
    angle_threshold = 5
    i = 0

    for line in sorted_lines:
        similar_lines = [line[0]]
        x1, y1, x2, y2 = line[0]
        angle1 = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))

        similarity = np.copy(img)

        cv.line(similarity, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for other_line in sorted_lines:
            if line != other_line:

                x3, y3, x4, y4 = other_line[0]
                angle2 = np.rad2deg(np.arctan2(y4 - y3, x4 - x3))
                if (abs(angle1 - angle2) < angle_threshold):
                    similar_lines.append(other_line[0])
                    cv.line(similarity, (x3, y3), (x4, y4), (0, 255, 0), 2)
                else:
                    cv.line(similarity, (x3, y3), (x4, y4), (0, 0, 255), 2)
        if demo:
            cv.imshow("similarity", similarity)
            cv.waitKey()
        sorted_similar.append(similar_lines)

    print("sorted similar:", sorted_similar)

    # get average of similar lines
    final_lines = []
    for i in range(len(sorted_similar)):
        x1, y1, x2, y2 = 0, 0, 0, 0
        num_lines = len(sorted_similar[i])
        for line in sorted_similar[i]:
            x3, y3, x4, y4 = line
            x1 += x3
            y1 += y3
            x2 += x4
            y2 += y4
        x1 = x1 // num_lines
        y1 = y1 // num_lines
        x2 = x2 // num_lines
        y2 = y2 // num_lines
        final_lines.append([x1, y1, x2, y2])

    print("final lines:", final_lines)
    unique_final_lines = [list(x) for x in set(tuple(x) for x in final_lines)]
    print("final lines set:", unique_final_lines)
    
    # find center by taking intersection of lines
    # todo: gauge/single hand center will have to be calculated in align()

    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    if (len(unique_final_lines) < 2):
        print("Only detected 1 hand. Exiting...")
        return
    x1, y1, x2, y2 = unique_final_lines[0]
    x3, y3, x4, y4 = unique_final_lines[1]
    centerX, centerY = line_intersection(
        ((x1, y1), (x2, y2)), ((x3, y3), (x4, y4)))
    print("centerX: ", centerX)
    print("centerY: ", centerY)

    final_line_img = np.copy(img)
    clock_hands = []
    for line in unique_final_lines:
        x1, y1, x2, y2 = line

        x_diff = float(x2-centerY)
        y_diff = float(y2-centerY)

        if(x_diff*y_diff > 0):
            if(x_diff >= 0 and y_diff > 0):
                angle = np.rad2deg(np.pi-np.arctan(x_diff/y_diff))
            elif(x_diff <= 0 and y_diff < 0):
                angle = np.rad2deg(2*np.pi-np.arctan(x_diff/y_diff))
        elif(x_diff*y_diff < 0):
            if(y_diff >= 0 and x_diff < 0):
                angle = np.rad2deg((3*np.pi)/4+np.arctan(x_diff/y_diff))
            elif(y_diff <= 0 and x_diff > 0):
                angle = np.rad2deg(-np.arctan(x_diff/y_diff))

        length = math.sqrt((x2 - x1) ** 2 + (y2-y1) ** 2)
        clock_hands.append((line, angle, length))
        cv.line(final_line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    print("clock hands:", clock_hands)

    clock_hands.sort(key=lambda x: x[2])
    print("sorted clock hands:", clock_hands)

    cv.circle(final_line_img, (centerX.astype(int),
              centerY.astype(int)), 3, (0, 0, 255), 5)

    cv.imshow("final lines", final_line_img)
    cv.waitKey()
    return clock_hands, final_line_img

def get_time(hands):
    use_sec = False
    hour_hand = hands[0][1]
    min_hand = hands[1][1]
    if len(hands) == 3:
        use_sec = True
        sec_hand = hands[2][1]

    hour = hour_hand//30
    hour = 12 if hour == 0 else int(hour)
    minute = int(min_hand/(np.rad2deg(2*np.pi))*60)

    if use_sec:
        second = int(sec_hand/(np.rad2deg(2*np.pi))*60)

    if use_sec:
        return (str(hour).zfill(2) + ":" + str(minute).zfill(2) + ":" + str(second).zfill(2))
    else:
        return (str(hour).zfill(2) + ":" + str(minute).zfill(2))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Detect time from clock image. ')
    parser.add_argument('path', metavar='path_to_image', type=str,
                        help='path to clock image')
    parser.add_argument(
        '--demo', action=argparse.BooleanOptionalAction, help='show intermediary steps')

    args = parser.parse_args()

    img = cv.imread(args.path)
    img = deskew(img, args.demo)

    hands, final_line_img = get_hands(img, args.demo)
    time = get_time(hands)
    print("time: ", time)
    cv.imshow("detected hands", final_line_img)
    cv.waitKey()
