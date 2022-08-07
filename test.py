# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
import time
import State
import os
import cv2
import numpy as np
import chainer
from torch import uint8, tensor
from MyFCN import MyFcn
from common import unsharp, box_blur, laplacian, edge_enhance_more, sharpen, draw_action_map, write_image
from mini_batch_loader import MiniBatchLoader
from pixelwise_a3c import PixelWiseA3C_InnerState_ConvR

# _/_/_/ paths _/_/_/
TRAINING_DATA_PATH = "./training.txt"
TESTING_DATA_PATH = "./testing.txt"
IMAGE_DIR_PATH = "./"
SAVE_PATH = "./"
OUTPUT_PATH = "./BSD68_example/"

# _/_/_/ training parameters _/_/_/
LEARNING_RATE = 0.001
TEST_BATCH_SIZE = 1  # must be 1
EPISODE_LEN = 5
GAMMA = 0.95  # discount factor

N_ACTIONS = 9

# number of actions that inc/decrement the pixel values.
# e.g., MOVE_RANGE=3, there are three actions: pixel_value +=1, +=0, -=1.
MOVE_RANGE = 3

CROP_SIZE = 70

GPU_ID = 0

visualization = True


def test(loader, agnt):
    sum_reward = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE)
    raw_x = None
    input_im = None

    COLOUR_TABLE = {
        0: tensor([0, 0, 0], dtype=uint8),  # black (pixel val -1)
        1: tensor([255, 255, 255], dtype=uint8),  # white (do nothing)
        2: tensor([255, 0, 0], dtype=uint8),  # red (pixel val +1)
        3: tensor([0, 255, 0], dtype=uint8),  # lime (unsharp)
        4: tensor([0, 0, 255], dtype=uint8),  # blue (laplace)
        5: tensor([255, 255, 0], dtype=uint8),  # yellow (gaussian)
        6: tensor([0, 255, 255], dtype=uint8),  # cyan / aqua (box blur)
        7: tensor([255, 0, 255], dtype=uint8),  # magenta / fuchsia (edge_enhance_more)
        8: tensor([128, 128, 128], dtype=uint8)  # gray (sharpen)
    }

    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        print("\nTESTING IMAGE", i, '/', test_data_size)
        raw_x, processed_x, desired_index = loader.load_testing_data(np.array(range(i, i + TEST_BATCH_SIZE)))
        # print(raw_x[0][0][32])
        # print("RAW X MIN", min(raw_x[0][0][32]), "RAW X MAX", max(raw_x[0][0][32]))

        # print(desired_index[0][0][-2])
        input_im = processed_x

        current_state.reset(input_im/2, input_im/2)
        reward = np.zeros(raw_x.shape, raw_x.dtype) * 255

        for t in range(0, EPISODE_LEN):
            print("\tPROCESSING EPISODE", t + 1, '/', EPISODE_LEN)
            previous_image = current_state.image.copy()
            action, inner_state = agnt.act(current_state.tensor)
            current_state.step(action, inner_state)
            reward = np.square(raw_x - previous_image) * 255 - np.square(raw_x - current_state.image) * 255
            sum_reward += np.mean(reward) * np.power(GAMMA, t)

            desire_path = OUTPUT_PATH + desired_index[0][0][-2] + '/' + desired_index[0][0][-1]
            # print("CURRENT STATE", current_image)

            if visualization:
                if t == 0:
                    # print("INPUT", input_im[0][0] * 255)
                    # print("OUTPUT", current_state.image[0][0] * 255)
                    os.makedirs(desire_path)

                    cv2.imwrite(desire_path + f"/raw_image.png",
                                raw_x[0][0] * 255)

                    cv2.imwrite(desire_path + f"/input_image.png",
                                input_im[0][0] * 255)

                    cv2.imwrite(desire_path + f"/step_{t}_output_image.png",
                                current_state.image[0][0] * 255)

                    cv2.imwrite(desire_path + f"/unsharp.png",
                                unsharp(input_im[0][0]) * 255)

                    cv2.imwrite(desire_path + f"/gauss.png",
                                cv2.GaussianBlur(input_im[0][0], ksize=(5, 5), sigmaX=0.5) * 255)

                    cv2.imwrite(desire_path + f"/box.png",
                                box_blur(input_im[0][0], (5, 5)) * 255)

                    cv2.imwrite(desire_path + f"/laplace.png",
                                laplacian(input_im[0][0]) * 255)

                    cv2.imwrite(desire_path + f"/edge.png",
                                edge_enhance_more(input_im[0][0]) * 255)

                    cv2.imwrite(desire_path + f"/sharpen.png",
                                sharpen(input_im[0][0]) * 255)

                    # print("ACTIONS", action)
                else:
                    cv2.imwrite(desire_path + f"/step_{t}_output_image.png",
                                np.clip(current_state.image[0][0] * 255, 0, 255))

                # print("CREATE ACTION MAP")
                action_map = draw_action_map(action, COLOUR_TABLE)

                # print("SAVE ACTION MAP")
                write_image(os.path.join(desire_path, f"step_{t}.png"), action_map)
            elif t == 4:
                # print("WRITING OUTPUT")
                os.makedirs(desire_path)
                cv2.imwrite(desire_path + f"/step_{t}_output_image.png",
                            np.clip(current_state.image[0][0] * 255, 0, 255))

        agnt.stop_episode()

    sys.stdout.flush()


def main():
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    chainer.cuda.get_device_from_id(GPU_ID).use()

    # load myfcn model
    model_path = './model/medical-pre-trained/model.npz'  # Sup res opt medical train fixed actions

    # print("LOADING MODEL")
    model = MyFcn(N_ACTIONS)
    # print("LOADED MODEL", type(model))

    # _/_/_/ setup _/_/_/
    # print("LOADING OPTIMIZER")
    optimizer_path = './model/medical-pre-trained/optimizer.npz'  # Sup res opt medical train fixed actions

    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agnt = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    print("LOADING MODEL")
    chainer.serializers.load_npz(model_path, agnt.model)
    print("FINISHED LOADING MODEL")

    print("LOADING OPTIMIZER")
    chainer.serializers.load_npz(optimizer_path, agnt.optimizer)
    print("FINISHED LOADING OPTIMIZER")

    agnt.act_deterministically = True
    agnt.model.to_device('@cupy:0')
    # print("LOADED OPTIMIZER")

    # _/_/_/ testing _/_/_/
    test(mini_batch_loader, agnt)


if __name__ == '__main__':
    try:
        fout = open('test_log_BSD68_example.txt', "w")
        start = time.time()
        main()
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start) / 60))
        print("{s}[h]".format(s=(end - start) / 60 / 60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start) / 60))
        fout.write("{s}[h]\n".format(s=(end - start) / 60 / 60))
        fout.close()
    except Exception as error:
        print(str(error))
