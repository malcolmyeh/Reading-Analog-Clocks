import numpy as np
import pandas as pd
import cv2 as cv

# preprocessing subtracts one from hour and divides minute by 60
def get_time(hour, minute):
    return (hour + 1, int(minute * 60))
    # return (hour + 1, minute + 1)

def preprocess(img, img_size):
    img = cv.resize(img, (img_size, img_size), interpolation=cv.INTER_AREA)
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY).reshape((img_size, img_size, 1))
    normalized = cv.normalize(
        img, None, alpha=0, beta=200, norm_type=cv.NORM_MINMAX)
    return normalized


def load_batch(ids=[0, 1, 2, 3, 4], batch_size=5, img_size=150):
    data = pd.read_csv('label.csv')
    path = 'images/'

    images = []
    hours = []
    minutes = []
    batch_ids = np.random.choice(ids, batch_size)

    for i in range(len(batch_ids)):
        img = cv.imread(path + str(batch_ids[i]) + '.jpg')

        images.append(preprocess(img, img_size))
        hours.append((data['hour'][data.index == batch_ids[i]])-1) # offset label by 1 for classification (keras expects label to start at 0)
        minutes.append((data['minute'][data.index == batch_ids[i]])/60) # normalize range to [0,1] for regression
        # minutes.append((data['minute'][data.index == batch_ids[i]])) # classification minute


    return (np.array(images), np.array(hours).astype(int), np.array(minutes).astype(float))
    # return (np.array(images), np.array(hours).astype(int), np.array(minutes).astype(int)) # classification minute
