import base64
import pickle
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image

from src.configs import DriverVariables, PathVariables, GameImage


def screen_shot_game(_driver):
    image_b64 = _driver.execute_script(DriverVariables.GETBASE64SCRIPT)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = image_preprocessing(screen)
    return image


def image_preprocessing(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
    image = image[:300, :500]  # Crop Region of Interest(ROI)
    image = cv2.resize(image, (GameImage.HEIGHT, GameImage.WIDTH))
    image[image > 0] = 255
    image = np.reshape(image, (GameImage.HEIGHT, GameImage.WIDTH, 1))
    return image


def image_to_tensor(image):
    image = np.transpose(image, (2, 0, 1))
    image_tensor = image.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    return image_tensor


def show_img():
    """
    Show images in new window
    """
    while True:
        screen = (yield)
        window_title = "Dinosaur"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.imshow(window_title, screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def show_conv_outputs():
    """
    Show images in new window
    """
    while True:
        screen = (yield)
        window_title = "Layer Outputs (C1-C2-C3)"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        temp = np.zeros((46, 40, 3), dtype=np.uint8)
        sc1, sc2, sc3 = screen[0], screen[1], screen[2]
        sc1 = np.array((((sc1 - sc1.min()) * 255) / (sc1.max() - sc1.min())), dtype=np.uint8)[:, :, :, 0]
        sc2 = np.array((((sc2 - sc2.min()) * 255) / (sc2.max() - sc2.min())), dtype=np.uint8)[:, :, :, 0]
        sc3 = np.array((((sc3 - sc3.min()) * 255) / (sc3.max() - sc3.min())), dtype=np.uint8)[:, :, :, 0]

        numpy_concat1 = np.concatenate((sc1[:, :, 0:3], sc1[:, :, 29:32]), axis=1)

        numpy_concat2 = np.concatenate((sc2[:, :, 0:3], sc2[:, :, 29:32]), axis=1)
        numpy_concat2 = np.concatenate((numpy_concat2, sc2[:, :, 61:64]), axis=1)

        numpy_concat3 = np.concatenate((sc3[:, :, 0:3], sc3[:, :, 13:16]), axis=1)
        numpy_concat3 = np.concatenate((numpy_concat3, sc3[:, :, 29:32]), axis=1)
        numpy_concat3 = np.concatenate((numpy_concat3, sc3[:, :, 61:64]), axis=1)

        temp[0:20, 0:40] = numpy_concat1
        temp[25:34, 0:27] = numpy_concat2
        temp[39:46, 0:28] = numpy_concat3

        cv2.imshow(window_title, temp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def save_obj(obj, name):
    with open(PathVariables.RECORD_SAVE_PATH + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
