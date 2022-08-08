# SPDX-License-Identifier: Unlicense

import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np
import cv2
from common import low_res_bilinear, low_res_nearest, mae, box_blur, edge_enhance_more, sharpen, laplacian
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from sewar.full_ref import vifp

load_training_data = False
load_testing_data = False
squareDataset = False
writeText = False
getStats = False
generateJPG = False
generateAltLowRes = False

avg_train_reward = []
avg_valid_PSNR = []

in_test_MAE = []
out_test_MAE = []
unsharp_test_MAE = []
bilinear_test_MAE = []
nearest_test_MAE = []

in_test_NRMSE = []
out_test_NRMSE = []
unsharp_test_NRMSE = []
bilinear_test_NRMSE = []
nearest_test_NRMSE = []

in_test_PSNR = []
out_test_PSNR = []
unsharp_test_PSNR = []
bilinear_test_PSNR = []
nearest_test_PSNR = []

in_test_SSIM = []
out_test_SSIM = []
unsharp_test_SSIM = []
bilinear_test_SSIM = []
nearest_test_SSIM = []

in_test_VIF = []
out_test_VIF = []
unsharp_test_VIF = []
bilinear_test_VIF = []
nearest_test_VIF = []

im_count = 0
seen_folders = []

# ------------------------------ LOAD AND RENDER TRAINING / TESTING METRICS ------------------------------
if load_training_data or load_testing_data:
    with open("../ACDC-never-stats.txt", "r") as file:
        # Loading and formatting data for visualization

        if load_training_data:
            for line in file:
                line = line.split('reward ')
                # print(line)
                if len(line) == 2:
                    line = line[1][0:-12]
                    line = line.split(', PSNR ')

                    if len(line) == 2:
                        avg_train_reward.append(float(line[0]))
                        avg_valid_PSNR.append(float(line[1]))
                    else:
                        # print("TRAINING", float(line[0]))
                        avg_train_reward.append(float(line[0]))
        else:
            for line in file:
                # print(line)
                line = line.strip().split(': ')
                if "Input PSNR" in line[0]:
                    in_test_PSNR.append(float(line[1]))
                elif "Output PSNR" in line[0]:
                    out_test_PSNR.append(float(line[1]))
                elif "Unsharp PSNR" in line[0]:
                    unsharp_test_PSNR.append(float(line[1]))
                elif "Bilinear PSNR" in line[0]:
                    bilinear_test_PSNR.append(float(line[1]))
                elif "Nearest PSNR" in line[0]:
                    nearest_test_PSNR.append(float(line[1]))
                elif "Input MAE" in line[0]:
                    in_test_MAE.append(float(line[1]))
                elif "Output MAE" in line[0]:
                    out_test_MAE.append(float(line[1]))
                elif "Unsharp MAE" in line[0]:
                    unsharp_test_MAE.append(float(line[1]))
                elif "Bilinear MAE" in line[0]:
                    bilinear_test_MAE.append(float(line[1]))
                elif "Nearest MAE" in line[0]:
                    nearest_test_MAE.append(float(line[1]))
                elif "Input SSIM" in line[0]:
                    in_test_SSIM.append(float(line[1]))
                elif "Output SSIM" in line[0]:
                    out_test_SSIM.append(float(line[1]))
                elif "Unsharp SSIM" in line[0]:
                    unsharp_test_SSIM.append(float(line[1]))
                elif "Bilinear SSIM" in line[0]:
                    bilinear_test_SSIM.append(float(line[1]))
                elif "Nearest SSIM" in line[0]:
                    nearest_test_SSIM.append(float(line[1]))
                elif "Input VIF" in line[0]:
                    in_test_VIF.append(float(line[1]))
                elif "Output VIF" in line[0]:
                    out_test_VIF.append(float(line[1]))
                elif "Unsharp VIF" in line[0]:
                    unsharp_test_VIF.append(float(line[1]))
                elif "Bilinear VIF" in line[0]:
                    bilinear_test_VIF.append(float(line[1]))
                elif "Nearest VIF" in line[0]:
                    nearest_test_VIF.append(float(line[1]))
                elif "Input NRMSE" in line[0]:
                    in_test_NRMSE.append(float(line[1]))
                elif "Output NRMSE" in line[0]:
                    out_test_NRMSE.append(float(line[1]))
                elif "Unsharp NRMSE" in line[0]:
                    unsharp_test_NRMSE.append(float(line[1]))
                elif "Bilinear NRMSE" in line[0]:
                    bilinear_test_NRMSE.append(float(line[1]))
                elif "Nearest NRMSE" in line[0]:
                    nearest_test_NRMSE.append(float(line[1]))

        file.close()

    if load_training_data:
        xs = np.arange(0, 16416, 1)
        print(len(xs) - len(avg_train_reward))

        m, b = np.polynomial.polynomial.polyfit(xs, avg_train_reward, 1)

        plt.plot(xs, avg_train_reward)
        plt.plot(xs, m * xs + b)
        plt.title("AVERAGE TRAINING - Medical")
        plt.show()

        xs = np.arange(0, 16416, 512)[0:32]
        plt.plot(xs, avg_valid_PSNR)
        plt.title("AVERAGE VALID PSNR")
        plt.show()
    else:
        xs = np.arange(0, len(out_test_PSNR), 1)

        # ----------------------- PSNR -----------------------
        ax = plt.subplot(1, 1, 1)
        ax.bar(xs, out_test_PSNR[0:36], color=['g'])
        ax.bar(xs, in_test_PSNR[0:36], color=['b'])
        ax.bar(xs, unsharp_test_PSNR[0:36], color=['darkred'])
        ax.plot(xs, np.mean(out_test_PSNR) * np.ones_like(xs), "--", color='lime')
        ax.plot(xs, np.mean(unsharp_test_PSNR) * np.ones_like(xs), "--", color='red')
        ax.plot(xs, np.mean(in_test_PSNR) * np.ones_like(xs), "--", color='deepskyblue')
        ax.legend(["Average Output PSNR", "Average Unsharp Masking PSNR", "Average Input PSNR",
                   "Output PSNR", "Input PSNR", "Unsharp Masking PSNR"],
                  loc='upper center', ncol=2, fancybox=True, shadow=True)

        plt.ylabel("PSNR")
        plt.xlabel("Image number")
        plt.ylim([20, 35])
        out_sd = format(round(np.std(out_test_PSNR), 2), ".2f")
        out_avg = format(round(np.mean(out_test_PSNR), 2), ".2f")
        in_sd = format(round(np.std(in_test_PSNR), 2), ".2f")
        in_avg = format(round(np.mean(in_test_PSNR), 2), ".2f")
        lin_sd = format(round(np.std(bilinear_test_PSNR), 2), ".2f")
        lin_avg = format(round(np.mean(bilinear_test_PSNR), 2), ".2f")
        near_sd = format(round(np.std(nearest_test_PSNR), 2), ".2f")
        near_avg = format(round(np.mean(nearest_test_PSNR), 2), ".2f")
        unsharp_avg = format(round(np.mean(unsharp_test_PSNR), 2), ".2f")
        unsharp_sd = format(round(np.std(unsharp_test_PSNR), 2), ".2f")

        print("Output PSNR", out_avg, out_sd)
        print("Input (bicubic) PSNR", in_avg, in_sd)
        print("Bilinear PSNR", lin_avg, lin_sd)
        print("Nearest PSNR", near_avg, near_sd)
        print("Unsharp Masking PSNR", unsharp_avg, unsharp_sd)
        plt.title(f"ACDC - Peak Signal to Noise Ratio")
        ax.set_facecolor('dimgrey')
        plt.show()

        # ----------------------- MAE -----------------------
        ax = plt.subplot(1, 1, 1)
        ax.bar(xs, unsharp_test_MAE[0:36], color=['darkred'])
        ax.bar(xs, in_test_MAE[0:36], color=['b'])
        ax.bar(xs, out_test_MAE[0:36], color=['g'])
        ax.plot(xs, np.mean(unsharp_test_MAE) * np.ones_like(xs), "--", color='red')
        ax.plot(xs, np.mean(in_test_MAE) * np.ones_like(xs), "--", color='deepskyblue')
        ax.plot(xs, np.mean(out_test_MAE) * np.ones_like(xs), "--", color='lime')
        ax.legend(["Average Unsharp Masking MAE", "Average Input MAE", "Average Output MAE",
                   "Unsharp Masking MAE", "Input MAE", "Output MAE"],
                  loc='upper center', ncol=2, fancybox=True, shadow=True)

        plt.ylabel("MAE")
        plt.xlabel("Image number")
        plt.ylim([0, 15.0])
        out_sd = format(round(np.std(out_test_MAE), 2), ".2f")
        out_avg = format(round(np.mean(out_test_MAE), 2), ".2f")
        in_sd = format(round(np.std(in_test_MAE), 2), ".2f")
        in_avg = format(round(np.mean(in_test_MAE), 2), ".2f")
        lin_sd = format(round(np.std(bilinear_test_MAE), 2), ".2f")
        lin_avg = format(round(np.mean(bilinear_test_MAE), 2), ".2f")
        near_sd = format(round(np.std(nearest_test_MAE), 2), ".2f")
        near_avg = format(round(np.mean(nearest_test_MAE), 2), ".2f")
        unsharp_avg = format(round(np.mean(unsharp_test_MAE), 2), ".2f")
        unsharp_sd = format(round(np.std(unsharp_test_MAE), 2), ".2f")

        print("Output MAE", out_avg, out_sd)
        print("Input (bicubic) MAE", in_avg, in_sd)
        print("Bilinear MAE", lin_avg, lin_sd)
        print("Nearest MAE", near_avg, near_sd)
        print("Unsharp Masking MAE", unsharp_avg, unsharp_sd)
        plt.title(f"ACDC - Mean Average Error")
        ax.set_facecolor('dimgrey')
        plt.show()

        # ----------------------- NRMSE -----------------------
        ax = plt.subplot(1, 1, 1)
        ax.bar(xs, unsharp_test_NRMSE[0:36], color=['darkred'])
        ax.bar(xs, in_test_NRMSE[0:36], color=['b'])
        ax.bar(xs, out_test_NRMSE[0:36], color=['g'])
        ax.plot(xs, np.mean(unsharp_test_NRMSE) * np.ones_like(xs), "--", color='red')
        ax.plot(xs, np.mean(in_test_NRMSE) * np.ones_like(xs), "--", color='deepskyblue')
        ax.plot(xs, np.mean(out_test_NRMSE) * np.ones_like(xs), "--", color='lime')
        ax.legend(["Average Unsharp Masking NRMSE", "Average Input NRMSE", "Average Output NRMSE",
                   "Unsharp Masking NRMSE", "Input NRMSE", "Output NRMSE"],
                  loc='upper center', ncol=2, fancybox=True, shadow=True)

        plt.ylabel("NRMSE")
        plt.xlabel("Image number")
        plt.ylim([0, 0.35])
        out_sd = format(round(np.std(out_test_NRMSE), 2), ".2f")
        out_avg = format(round(np.mean(out_test_NRMSE), 2), ".2f")
        in_sd = format(round(np.std(in_test_NRMSE), 2), ".2f")
        in_avg = format(round(np.mean(in_test_NRMSE), 2), ".2f")
        lin_sd = format(round(np.std(bilinear_test_NRMSE), 2), ".2f")
        lin_avg = format(round(np.mean(bilinear_test_NRMSE), 2), ".2f")
        near_sd = format(round(np.std(nearest_test_NRMSE), 2), ".2f")
        near_avg = format(round(np.mean(nearest_test_NRMSE), 2), ".2f")
        unsharp_avg = format(round(np.mean(unsharp_test_NRMSE), 2), ".2f")
        unsharp_sd = format(round(np.std(unsharp_test_NRMSE), 2), ".2f")

        print("Output NRMSE", out_avg, out_sd)
        print("Input (bicubic) NRMSE", in_avg, in_sd)
        print("Bilinear NRMSE", lin_avg, lin_sd)
        print("Nearest NRMSE", near_avg, near_sd)
        print("Unsharp Masking NRMSE", unsharp_avg, unsharp_sd)
        plt.title(f"ACDC - Normalized Root Mean Squared Error")
        ax.set_facecolor('dimgrey')
        plt.show()

        # ----------------------- VIF -----------------------
        ax = plt.subplot(1, 1, 1)
        ax.bar(xs, out_test_VIF[0:36], color=['g'])
        ax.bar(xs, in_test_VIF[0:36], color=['b'])
        ax.bar(xs, unsharp_test_VIF[0:36], color=['darkred'])
        ax.plot(xs, np.mean(unsharp_test_VIF) * np.ones_like(xs), "--", color='red')
        ax.plot(xs, np.mean(out_test_VIF) * np.ones_like(xs), "--", color='lime')
        ax.plot(xs, np.mean(in_test_VIF) * np.ones_like(xs), "--", color='deepskyblue')
        ax.legend(["Average Unsharp Masking VIF", "Average Output VIF", "Average Input VIF",
                   "Output VIF", "Input VIF", "Unsharp Masking VIF"],
                  loc='upper center', ncol=2, fancybox=True, shadow=True)

        out_sd = format(round(np.std(out_test_VIF), 2), ".2f")
        out_avg = format(round(np.mean(out_test_VIF), 2), ".2f")
        in_sd = format(round(np.std(in_test_VIF), 2), ".2f")
        in_avg = format(round(np.mean(in_test_VIF), 2), ".2f")
        lin_sd = format(round(np.std(bilinear_test_VIF), 2), ".2f")
        lin_avg = format(round(np.mean(bilinear_test_VIF), 2), ".2f")
        near_sd = format(round(np.std(nearest_test_VIF), 2), ".2f")
        near_avg = format(round(np.mean(nearest_test_VIF), 2), ".2f")
        unsharp_avg = format(round(np.mean(unsharp_test_VIF), 2), ".2f")
        unsharp_sd = format(round(np.std(unsharp_test_VIF), 2), ".2f")

        print("Output VIF", out_avg, out_sd)
        print("Input (bicubic) VIF", in_avg, in_sd)
        print("Bilinear VIF", lin_avg, lin_sd)
        print("Nearest VIF", near_avg, near_sd)
        print("Unsharp Masking VIF", unsharp_avg, unsharp_sd)
        plt.title(f"ACDC - Visual Information Fidelity")
        ax.set_facecolor('dimgrey')
        plt.show()

        # ----------------------- SSIM -----------------------
        ax = plt.subplot(1, 1, 1)
        ax.bar(xs, out_test_SSIM[0:36], color=['g'])
        ax.bar(xs, in_test_SSIM[0:36], color=['b'])
        ax.bar(xs, unsharp_test_SSIM[0:36], color=['darkred'])
        ax.plot(xs, np.mean(unsharp_test_SSIM) * np.ones_like(xs), "--", color='red')
        ax.plot(xs, np.mean(in_test_SSIM) * np.ones_like(xs), "--", color='deepskyblue')
        ax.plot(xs, np.mean(out_test_SSIM) * np.ones_like(xs), "--", color='lime')
        ax.legend(["Average Unsharp Masking SSIM", "Average Input SSIM", "Average Output SSIM",
                   "Output SSIM", "Input SSIM", "Unsharp Masking SSIM"],
                  loc='upper center', ncol=2, fancybox=True, shadow=True)

        plt.ylabel("SSIM")
        plt.xlabel("Image number")
        plt.ylim([0.5, 1.0])
        out_sd = format(round(np.std(out_test_SSIM), 2), ".2f")
        out_avg = format(round(np.mean(out_test_SSIM), 2), ".2f")
        in_sd = format(round(np.std(in_test_SSIM), 2), ".2f")
        in_avg = format(round(np.mean(in_test_SSIM), 2), ".2f")
        lin_sd = format(round(np.std(bilinear_test_SSIM), 2), ".2f")
        lin_avg = format(round(np.mean(bilinear_test_SSIM), 2), ".2f")
        near_sd = format(round(np.std(nearest_test_SSIM), 2), ".2f")
        unsharp_avg = format(round(np.mean(unsharp_test_SSIM), 2), ".2f")
        unsharp_sd = format(round(np.std(unsharp_test_SSIM), 2), ".2f")
        print("Output SSIM", out_avg, out_sd)
        print("Input (bicubic) SSIM", in_avg, in_sd)
        print("Bilinear SSIM", lin_avg, lin_sd)
        print("Nearest SSIM", near_avg, near_sd)
        print("Unsharp Masking SSIM", unsharp_avg, unsharp_sd)
        plt.title(f"ACDC - Structural Similarity Index Measure")
        ax.set_facecolor('dimgrey')
        plt.show()

for root, dirs, files in os.walk("./BSD68_square/valid", topdown=True):
    for name in files:
        file_name = os.path.splitext(os.path.join(root, name))[0].replace('\\', '/') + \
                    os.path.splitext(os.path.join(root, name))[1]

        sub_folder_len = len(file_name.split('/'))

        if sub_folder_len >= 3:
            folder_name = file_name.split('/')[2]

        if sub_folder_len >= 4:
            sub_folder_name = file_name.split('/')[3]

        if sub_folder_len >= 5:
            sub_sub_folder_name = file_name.split('/')[4]

        # ------------------------------ GENERATE ALTERNATE LOW RES ------------------------------
        if generateAltLowRes:
            if (sub_folder_name + sub_sub_folder_name) not in seen_folders:
                seen_folders.append(sub_folder_name + sub_sub_folder_name)

                print("Generating bilinear and nearest for", sub_folder_name + '/' + sub_sub_folder_name,
                      '|', len(seen_folders), "out of around 3,933 files processed")

                ground_truth = cv2.imread("../deblur/ACDC-never-seen-fixed" + '/' + sub_folder_name + '/'
                                          + sub_sub_folder_name + '/' + "raw_image.png")[:, :, 0]

                bil_save = folder_name + '/' + sub_folder_name + '/' + sub_sub_folder_name + '/' + "bilinear.png"
                cv2.imwrite(bil_save, low_res_bilinear(ground_truth))

                near_save = folder_name + '/' + sub_folder_name + '/' + sub_sub_folder_name + '/' + "nearest.png"
                cv2.imwrite(near_save, low_res_nearest(ground_truth))

        # ------------------------------ GENERATE METRICS ------------------------------
        if getStats:
            with open("../ACDC-never-stats.txt", "a+") as file_out:
                if (sub_folder_name + sub_sub_folder_name) not in seen_folders:
                    seen_folders.append(sub_folder_name + sub_sub_folder_name)
                    print("Getting stats for", sub_folder_name + '/' + sub_sub_folder_name,
                          '|', len(seen_folders), "out of around 36 files processed")
                    # print(sub_folder_name + sub_sub_folder_name)

                    inp = cv2.imread("../deblur/ACDC-never-seen-fixed" + '/' + sub_folder_name + '/'
                                     + sub_sub_folder_name + '/' + "input_image.png")[:, :, 0]

                    output = cv2.imread("../deblur/ACDC-never-seen-fixed" + '/' + sub_folder_name + '/'
                                        + sub_sub_folder_name + '/' + "step_4_output_image.png")[:, :, 0]

                    ground_truth = cv2.imread("../deblur/ACDC-never-seen-fixed" + '/' + sub_folder_name + '/'
                                              + sub_sub_folder_name + '/' + "raw_image.png")[:, :, 0]

                    unsharp = cv2.imread("../deblur/ACDC-never-seen-fixed" + '/' + sub_folder_name + '/'
                                         + sub_sub_folder_name + '/' + "unsharp.png")[:, :, 0]

                    bilinear = cv2.imread("../deblur/ACDC-never-seen-fixed" + '/' + sub_folder_name + '/'
                                          + sub_sub_folder_name + '/' + "bilinear.png")[:, :, 0]

                    nearest = cv2.imread("../deblur/ACDC-never-seen-fixed" + '/' + sub_folder_name + '/'
                                         + sub_sub_folder_name + '/' + "nearest.png")[:, :, 0]

                    file_out.write(
                        f"\nOutput MAE for image {im_count}: " + format(round(mae(output, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nInput MAE for image {im_count}: " + format(round(mae(inp, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nUnsharp MAE for image {im_count}: " + format(round(mae(unsharp, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nBilinear MAE for image {im_count}: " + format(round(mae(bilinear, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nNearest MAE for image {im_count}: " + format(round(mae(nearest, ground_truth), 4), ".4f"))

                    file_out.write(
                        f"\nOutput NRMSE for image {im_count}: " + format(round(nrmse(output, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nInput NRMSE for image {im_count}: " + format(round(nrmse(inp, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nUnsharp NRMSE for image {im_count}: " + format(round(nrmse(unsharp, ground_truth), 4),
                                                                           ".4f"))
                    file_out.write(
                        f"\nBilinear NRMSE for image {im_count}: " + format(round(nrmse(bilinear, ground_truth), 4),
                                                                            ".4f"))
                    file_out.write(
                        f"\nNearest NRMSE for image {im_count}: " + format(round(nrmse(nearest, ground_truth), 4),
                                                                           ".4f"))

                    file_out.write(
                        f"\nOutput PSNR for image {im_count}: " + format(round(psnr(output, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nInput PSNR for image {im_count}: " + format(round(psnr(inp, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nUnsharp PSNR for image {im_count}: " + format(round(psnr(unsharp, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nBilinear PSNR for image {im_count}: " + format(round(psnr(bilinear, ground_truth), 4),
                                                                           ".4f"))
                    file_out.write(
                        f"\nNearest PSNR for image {im_count}: " + format(round(psnr(nearest, ground_truth), 4), ".4f"))

                    file_out.write(
                        f"\nOutput SSIM for image {im_count}: " + format(round(ssim(output, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nInput SSIM for image {im_count}: " + format(round(ssim(inp, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nUnsharp SSIM for image {im_count}: " + format(round(ssim(unsharp, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nBilinear SSIM for image {im_count}: " + format(round(ssim(bilinear, ground_truth), 4),
                                                                           ".4f"))
                    file_out.write(
                        f"\nNearest SSIM for image {im_count}: " + format(round(ssim(nearest, ground_truth), 4), ".4f"))

                    file_out.write(
                        f"\nOutput VIF for image {im_count}: " + format(round(vifp(output, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nInput VIF for image {im_count}: " + format(round(vifp(inp, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nUnsharp VIF for image {im_count}: " + format(round(vifp(unsharp, ground_truth), 4), ".4f"))
                    file_out.write(
                        f"\nBilinear VIF for image {im_count}: " + format(round(vifp(bilinear, ground_truth), 4),
                                                                          ".4f"))
                    file_out.write(
                        f"\nNearest VIF for image {im_count}: " + format(round(vifp(nearest, ground_truth), 4), ".4f"))

                    im_count += 1

                file_out.close()

        # ------------------------------ GENERATE TRAINING / TESTING FILE ------------------------------
        if writeText:
            with open("../validation.txt", "a+") as file:
                temp_name = file_name.replace('\\', '/').split('/')[-1]

                print("WRITING", file_name[2:].replace('\\', '/'), "to file")

                file.write("\n" + file_name[2:].replace('\\', '/'))

        # ------------------------------ SQUARE DATASET ------------------------------
        if squareDataset:
            im = cv2.imread(file_name, 0)
            width, height = im.shape

            outfile = "../BSD68_square/" + folder_name + '/' + sub_folder_name[:-4] + "_square.jpg"

            os.makedirs("../BSD68_square/" + folder_name + '/', exist_ok=True)

            if width <= 256 and height <= 256:

                change_width = (256 - width) // 2
                change_height = (256 - height) // 2
                # print("RESCALING TO:", width + 2 * change_width, height + 2 * change_height)

                sqr = cv2.copyMakeBorder(im, change_width, change_width, change_height, change_height,
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])

                # print("OUTPUTTING TO", outfile)
                square_im = Image.fromarray(np.clip(sqr, 0.0, 255.0).astype(np.uint8))
                square_im.thumbnail(square_im.size)
                square_im.save(outfile, "JPEG", quality=100)
            elif height > 256 and width < 256:
                print("CURRENT DIM:", width, height)
                img = im[:, 0:256].copy()
                new_width, new_height = img.shape
                change_width = (256 - width) // 2
                print("LARGE! RESCALING TO:", new_width + 2 * change_width, new_height)

                sqr = cv2.copyMakeBorder(img, change_width, change_width, 0, 0,
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])

                print("OUTPUTTING TO", outfile)
                square_im = Image.fromarray(np.clip(sqr, 0.0, 255.0).astype(np.uint8))
                square_im.thumbnail(square_im.size)
                square_im.save(outfile, "JPEG", quality=100)
            elif width > 256 and height < 256:
                print("CURRENT DIM:", width, height)
                img = im[0:256, :].copy()
                new_width, new_height = img.shape
                change_height = (256 - height) // 2
                print("LARGE! RESCALING TO:", new_width, new_height + 2 * change_height)

                sqr = cv2.copyMakeBorder(img, 0, 0, change_height, change_height,
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])

                print("OUTPUTTING TO", outfile)
                square_im = Image.fromarray(np.clip(sqr, 0.0, 255.0).astype(np.uint8))
                square_im.thumbnail(square_im.size)
                square_im.save(outfile, "JPEG", quality=100)
            elif width > 256 and height > 256:
                print("CURRENT DIM:", width, height)
                img = im[0:256, 0:256].copy()

                print("LARGE! OUTPUTTING TO", outfile)
                square_im = Image.fromarray(np.clip(img, 0.0, 255.0).astype(np.uint8))
                square_im.thumbnail(square_im.size)
                square_im.save(outfile, "JPEG", quality=100)

        # ------------------------------ DATASET TO GREYSCALE JPG ------------------------------
        if generateJPG:
            if file_name[-4:] in [".gif", ".tif", ".png", ".dcm"]:

                # Check if jpeg version of file exists
                if os.path.isfile(os.path.splitext(os.path.join("./ADNI_testing/0", name))[0] + ".jpg"):
                    print("A jpeg file already exists for", name)
                    # os.remove(os.path.splitext(os.path.join("../waterloo_gray", name))[0] + ".jpg")

                # Convert bmp to jpeg to allow cv2 to properly load images
                else:
                    print("Generating jpeg for", name, "in", file_name[:-4])
                    outfile = file_name[:-4] + "_gray.jpg"
                    # print(root)
                    # print(name)
                    # print(os.path.join(root, name).replace('\\', '/'))
                    im = ImageOps.grayscale(Image.open(os.path.join(root, name).replace('\\', '/')))
                    im.thumbnail(im.size)
                    im.save(outfile, "JPEG", quality=100)
