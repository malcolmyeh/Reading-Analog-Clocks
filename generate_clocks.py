import numpy as np
import cv2
import math
import random
import pandas as pd
from matplotlib import pyplot as plt
import argparse


def random_int(min, max):
    return random.choice(range(min, max+1))


def random_colour(min=0, max=255):
    colour = random.choice(range(min, max))
    return (colour, colour, colour)


def line(img, src, dst, colour, thickness):
    img = cv2.line(img, src, dst, colour, thickness)
    # todo: shadows
    return img


def generate_ticks(img, center, radius, numerals, border_thickness):
    center_x, center_y = center
    angle = np.arange(60)
    angle *= 6
    cos_angle = np.cos(angle * math.pi/180)
    sin_angle = np.sin(angle * math.pi/180)

    tick_offset = random_int(0, 15)  # distance from border to tick
    hour_tick_length = random_int(1, 15)
    minute_tick_length = random_int(1, hour_tick_length)
    hour_tick_thickness = random_int(1, 10)
    minute_tick_thickness = random_int(1, hour_tick_thickness)
    tick_colour = random_colour()

    hour_x1 = np.rint(center_x + cos_angle * (radius -
                      border_thickness - tick_offset)).astype(int)
    hour_y1 = np.rint(center_y + sin_angle * (radius -
                      border_thickness - tick_offset)).astype(int)
    hour_x2 = np.rint(center_x + cos_angle * (radius -
                      border_thickness - tick_offset - hour_tick_length)).astype(int)
    hour_y2 = np.rint(center_y + sin_angle * (radius -
                      border_thickness - tick_offset - hour_tick_length)).astype(int)

    minute_x1 = np.rint(center_x + cos_angle * (radius -
                        border_thickness - tick_offset)).astype(int)
    minute_y1 = np.rint(center_y + sin_angle * (radius -
                        border_thickness - tick_offset)).astype(int)
    minute_x2 = np.rint(center_x + cos_angle * (radius -
                        border_thickness - tick_offset - minute_tick_length)).astype(int)
    minute_y2 = np.rint(center_y + sin_angle * (radius -
                        border_thickness - tick_offset - minute_tick_length)).astype(int)

    for i in range(len(angle)):
        img = cv2.line(img, (minute_x1[i], minute_y1[i]),
                       (minute_x2[i], minute_y2[i]), tick_colour, minute_tick_thickness)

    for i in range(len(angle[::5]), 5):
        img = cv2.line(img, (hour_x1[i], hour_y1[i]),
                       (hour_x2[i], hour_y2[i]), tick_colour, hour_tick_thickness)
    if numerals:
        font_length = np.random.uniform(0.5, 2)
        number_offset = random_int(10, 40)
        font_thickness = random_int(1, 4)
        font_colour = random_colour(0, 100)
        angle = np.arange(12)
        angle *= 30
        cos_angle = np.cos(angle * math.pi/180)
        sin_angle = np.sin(angle * math.pi/180)
        text_x = np.rint(center_x + cos_angle * (radius - border_thickness -
                         hour_tick_length - number_offset)).astype(int)
        text_y = np.rint(center_y + sin_angle * (radius - border_thickness -
                         hour_tick_length - number_offset)).astype(int)
        for i in range(len(angle)):
            # shift number to corerspond to angles
            shifted_i = i + 2
            shifted_i %= 12
            # center text
            text_x[i] -= cv2.getTextSize(
                str(numerals[shifted_i]), 0, font_length, font_thickness)[0][0] // 2
            text_y[i] += cv2.getTextSize(
                str(numerals[shifted_i]), 0, font_length, font_thickness)[0][1] // 2
            cv2.putText(img, str(numerals[shifted_i]), (text_x[i],
                        text_y[i]), 0, font_length, font_colour, font_thickness)
    return img


def generate_hands(img, center, radius, hour, minute, second, use_second):
    # shadow = random.choice([True, False])
    second_colour = random_colour(0, 150)
    minute_colour = random_colour(0, 150)
    hour_colour = random_colour(0, 150)
    second_thickness = random_int(1, 3)
    minute_thickness = random_int(5, 10)
    hour_thickness = random_int(minute_thickness, 12)

    second_length = random.uniform(0.8, 0.9)
    minute_length = random.uniform(0.6, second_length - 0.1)
    hour_length = random.uniform(0.3, minute_length - 0.1)
    second_back_length = random.uniform(-0.3, 0)
    minute_back_length = random.uniform(-0.3, 0)
    hour_back_length = random.uniform(-0.15, 0)

    center_x, center_y = center

    def get_hand_coordinates(cx, cy, radius, length, back_length, angle):
        x1 = (cx + length * np.cos(angle * math.pi/180)*radius).astype(int)
        y1 = (cy + length * np.sin(angle * math.pi/180)*radius).astype(int)
        x2 = (cx + back_length * np.cos(angle * math.pi/180)*radius).astype(int)
        y2 = (cy + back_length * np.sin(angle * math.pi/180)*radius).astype(int)
        return (x1, y1), (x2, y2)

    if use_second:
        second_angle = second * 6
        src, dest = get_hand_coordinates(
            center_x, center_y, radius, second_length, second_back_length, second_angle)
        img = line(img, src, dest, second_colour, second_thickness)

    adjusted_minute = minute + (second / 60)
    minute_angle = adjusted_minute * 6 - 90

    src, dest = get_hand_coordinates(
        center_x, center_y, radius, minute_length, minute_back_length, minute_angle)
    img = line(img, src, dest, minute_colour, minute_thickness)

    adjusted_hour = hour + minute / 60
    hour_angle = adjusted_hour * 30 - 90
    src, dest = get_hand_coordinates(
        center_x, center_y, radius, hour_length, hour_back_length, hour_angle)
    img = line(img, src, dest, hour_colour, hour_thickness)

    return img


def generate_clock(warp=False):
    height = 450
    width = 450
    background_colour = random_colour(200, 255)
    border_colour = random_colour(0, 150)
    center = (height//2, width//2)
    border_thickness = random_int(0, 25)
    radius = (min(height, width) // 2 - border_thickness // 2 - 1)

    minute = random_int(0, 59)
    hour = random_int(1, 12)
    second = random_int(0, 59)
    use_second = random.choice([True, False])

    arabic_numerals = random.choice([True, False])
    roman_numerals = False if arabic_numerals else random.choice([True, False])
    numeral_text = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] if arabic_numerals else [
        'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'] if roman_numerals else []

    img = np.zeros((height, width, 3), np.uint8)
    img[:] = background_colour
    clock_face_colour = (255, 255, 255)  # white
    img = cv2.circle(img, center, radius, clock_face_colour, cv2.FILLED)
    img = cv2.circle(img, center, radius, border_colour, border_thickness)

    # generate ticks
    img = generate_ticks(img, center, radius, numeral_text, border_thickness)

    # generate hands
    img = generate_hands(img, center, radius, hour, minute, second, use_second)

    # warp
    h = int(height * random.uniform(0.85, 0.9))
    w = int(width * random.uniform(0.85, 0.9))
    y = (height-h)//2
    x = (width-w)//2

    points = np.array(((x, y), (x+w, y), (x, y+h),
                      (x+w, y+h)), dtype=np.float32)

    WARP_FACTOR = 1

    warped_points = points + \
        np.random.randint(-WARP_FACTOR*x, WARP_FACTOR*x,
                          np.shape(points)).astype(np.float32)
    M = cv2.getPerspectiveTransform(points, warped_points)

    # calculate max of warp
    max_height = max(np.amax(warped_points, axis=0)[0].astype(
        int), np.amax(points, axis=0)[0].astype(int), height)
    max_width = max(np.amax(warped_points, axis=0)[1].astype(
        int), np.amax(points, axis=0)[1].astype(int), width)
    if warp:
        img = cv2.warpPerspective(img, M, (max_height, max_width),
                                  borderValue=background_colour)
    cv2.imwrite("clock.jpg", img)
    return img, hour, minute


def generate_data(num_clocks=10, dir='images/', warp=False):
    hours = []
    minutes = []
    for i in range(num_clocks):
        img, hour, minute = generate_clock(warp)
        print('\r', int(i / num_clocks * 100), end="% ", flush=True)
        plt.imsave(dir+str(i)+'.jpg', img)
        hours.append(hour)
        minutes.append(minute)
    label = {'hour': hours, 'minute': minutes}
    label = pd.DataFrame(label)
    label.to_csv('label.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate clock dataset.')
    parser.add_argument('num_clocks', metavar='N', type=int,
                        help='number of clocks to generate')
    parser.add_argument(
        '--skew', action=argparse.BooleanOptionalAction, help='skew generated clocks')

    args = parser.parse_args()
    generate_data(num_clocks=args.num_clocks,
                  warp=True if args.skew else False)
