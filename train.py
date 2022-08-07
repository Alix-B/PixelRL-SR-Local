# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
import time
import State
import os
import torch
import cv2
import chainer
import numpy as np
from mini_batch_loader import MiniBatchLoader
from MyFCN import MyFcn
from pixelwise_a3c import PixelWiseA3C_InnerState_ConvR
from common import draw_action_map, write_image
from skimage.metrics import peak_signal_noise_ratio as psnr

# _/_/_/ paths _/_/_/
TRAINING_DATA_PATH = "./training.txt"
VALIDATION_DATA_PATH = "./validation.txt"
IMAGE_DIR_PATH = "./"
SAVE_PATH = "./model/BSD68_example/"

# _/_/_/ training parameters _/_/_/
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 64  # train in batches of x
TEST_BATCH_SIZE = 1  # must be 1
N_EPISODES = 4096  # total episodes to train for
EPISODE_LEN = 5  # Number of iterations (actions to take) per episode
TEST_EPISODES = 512  # how often to validate and save model
GAMMA = 0.95  # discount factor

N_ACTIONS = 9

# number of actions that move the pixel values. e.g., MOVE_RANGE=3, there are three actions: pixel_value+=1, +=0, -=1.
MOVE_RANGE = 3

CROP_SIZE = 64

GPU_ID = 0


def main(file_out):
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        VALIDATION_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    chainer.cuda.get_device_from_id(GPU_ID).use()

    current_state = State.State((TRAIN_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE)

    # load myfcn model
    model = MyFcn(N_ACTIONS)

    # _/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agnt = PixelWiseA3C_InnerState_ConvR(model, optimizer, EPISODE_LEN, GAMMA)
    agnt.model.to_device('@cupy:0')

    # _/_/_/ training _/_/_/

    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    best_reward = 0

    for episode in range(1, N_EPISODES + 1):
        # display current state
        print(f"\nepisode {episode}",
              f"- Saving in {TEST_EPISODES - (episode % TEST_EPISODES)}",
              f"- Training complete in {N_EPISODES - episode}")

        file_out.write("episode %d\n" % episode)
        sys.stdout.flush()

        # load images
        r = indices[i:i + TRAIN_BATCH_SIZE]
        raw_x, processed_x, desired_index = mini_batch_loader.load_training_data(r)

        # generate input
        input_im = processed_x

        # print("SANITY CHECK", (blur/2 + blur/2) == blur)
        # print(raw_x.shape, b.shape, comb.shape)
        # print("SANITY CHECK", blur == (blur/2 + blur/2))

        # initialize the current state and reward

        current_state.reset(input_im / 2, input_im / 2)
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0
        # cv2.imwrite(f"../INPUT_IMAGE_{episode}.png", current_state.image.copy()[0][0] * 255)
        # cv2.imwrite(f"../RAW_IMAGE_{episode}.png", raw_x[0][0] * 255)

        # print("STARTING TRAINING")
        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agnt.act_and_train(current_state.tensor, reward)
            current_state.step(action, inner_state)
            reward = np.square(raw_x - previous_image) * 255 - np.square(raw_x - current_state.image) * 255
            sum_reward += np.mean(reward) * np.power(GAMMA, t)

        agnt.stop_episode_and_train(current_state.tensor, reward, True)

        if episode == 1:
            best_reward = sum_reward * 255
        elif sum_reward * 255 > best_reward:
            best_reward = sum_reward * 255

        print("train total reward:", round(sum_reward * 255, 2))
        print("best reward:", round(best_reward, 2))
        file_out.write("train total reward {a}\n".format(a=sum_reward * 255))
        sys.stdout.flush()

        if episode % TEST_EPISODES == 0:
            # _/_/_/ testing _/_/_/
            test(mini_batch_loader, agnt, file_out, episode)

        if i + TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:
            i += TRAIN_BATCH_SIZE

        if i + 2 * TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        optimizer.alpha = LEARNING_RATE * ((1 - episode / N_EPISODES) ** 0.9)


def test(loader, agnt, file_out, episode):

    sum_psnr = 0
    sum_reward = 0
    test_data_size = MiniBatchLoader.count_paths(VALIDATION_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE), MOVE_RANGE)
    raw_x = None
    input_im = None
    path = SAVE_PATH + str(episode)
    # os.makedirs(path)
    desired_im_count = 1

    COLOUR_TABLE = {
        0: torch.tensor([0, 0, 0], dtype=torch.uint8),  # black (pixel val -1)
        1: torch.tensor([255, 255, 255], dtype=torch.uint8),  # white (do nothing)
        2: torch.tensor([255, 0, 0], dtype=torch.uint8),  # red (pixel val +1)
        3: torch.tensor([0, 255, 0], dtype=torch.uint8),  # lime (unsharp)
        4: torch.tensor([0, 0, 255], dtype=torch.uint8),  # blue (laplace)
        5: torch.tensor([255, 255, 0], dtype=torch.uint8),  # yellow (gaussian)
        6: torch.tensor([0, 255, 255], dtype=torch.uint8),  # cyan / aqua (box blur)
        7: torch.tensor([255, 0, 255], dtype=torch.uint8),  # magenta / fuchsia (edge_enhance_more)
        8: torch.tensor([128, 128, 128], dtype=torch.uint8)  # gray (sharpen)
    }

    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        # print("TESTING", round(count / test_data_size, 4) * 100, '%')
        raw_x, processed_x, desired_index = loader.load_testing_data(np.array(range(i, i + TEST_BATCH_SIZE)))

        input_im = processed_x

        current_state.reset(input_im / 2, input_im / 2)
        reward = np.zeros(raw_x.shape, raw_x.dtype) * 255

        for t in range(0, EPISODE_LEN):
            previous_image = current_state.image.copy()
            action, inner_state = agnt.act(current_state.tensor)
            current_state.step(action, inner_state)
            reward = np.square(raw_x - previous_image) * 255 - np.square(raw_x - current_state.image) * 255
            sum_reward += np.mean(reward) * np.power(GAMMA, t)

            for index in desired_index:
                if i == index[1]:
                    print("SAVING DESIRED IMAGE", desired_im_count, '/', 2 * 5)
                    desired_im_count += 1

                    desire_path = path + '/' + desired_index[0][0][-1]

                    if t == 0:
                        # print("INPUT", input_im[0][0] * 255)
                        # print("OUTPUT", current_state.image[0][0] * 255)
                        os.makedirs(desire_path)

                        cv2.imwrite(desire_path + f"/raw_image.png",
                                    raw_x[0][0] * 255)

                        cv2.imwrite(desire_path + f"/input_image.png",
                                    input_im[0][0] * 255)

                        cv2.imwrite(desire_path + f"/episode_{episode}_step_{t}_output_image.png",
                                    current_state.image[0][0] * 255)

                        # print("ACTIONS", action)
                    else:
                        cv2.imwrite(desire_path + f"/episode_{episode}_step_{t}_output_image.png",
                                    np.clip(current_state.image[0][0] * 255, 0, 255))

                    # print("CREATE ACTION MAP")
                    action_map = draw_action_map(action, COLOUR_TABLE)

                    # print("ACTION MAP", action_map)

                    # print("SAVE ACTION MAP")
                    write_image(os.path.join(desire_path, f"{episode}_{index[0]}_step_{t}.png"), action_map)

        agnt.stop_episode()

        I = np.maximum(0, raw_x)
        I = np.minimum(1, I)
        I = (I * 255 + 0.5).astype(np.uint8)

        p = np.maximum(0, current_state.image)
        p = np.minimum(1, p)
        p = (p * 255 + 0.5).astype(np.uint8)

        inp = np.maximum(0, input_im)
        inp = np.minimum(1, inp)
        inp = (inp * 255 + 0.5).astype(np.uint8)
        # print(type(inp), type(I))

        # print(p, "\n \n \n \n", blur_in)

        sum_psnr += psnr(p[0][0], I[0][0])

        # print(f"PSNR for image {i}: {psnr(p[0][0], I[0][0])}")
        # print(f"Input PSNR for image {i}: {psnr(inp[0][0], I[0][0])}")

        file_out.write(f"PSNR for image {i}: {psnr(p[0][0], I[0][0])}\n")
        file_out.write(f"Input PSNR for image {i}: {psnr(inp[0][0], I[0][0])}\n")

    agnt.save(desire_path)

    print("test total reward {a}, PSNR {b}".format(a=sum_reward * 255 / test_data_size, b=sum_psnr / test_data_size))
    file_out.write(
        "test total reward {a}, PSNR {b}\n".format(a=sum_reward * 255 / test_data_size, b=sum_psnr / test_data_size))
    sys.stdout.flush()


if __name__ == '__main__':
    try:
        fout = open('train-log-07-08-22-bsd68-example.txt', "w")
        start = time.time()
        main(fout)
        end = time.time()
        print("{s}[s]".format(s=end - start))
        print("{s}[m]".format(s=(end - start) / 60))
        print("{s}[h]".format(s=(end - start) / 60 / 60))
        fout.write("{s}[s]\n".format(s=end - start))
        fout.write("{s}[m]\n".format(s=(end - start) / 60))
        fout.write("{s}[h]\n".format(s=(end - start) / 60 / 60))
        fout.close()
    except Exception as error:
        print(error)
