# Multi-Step Reinforcement Learning for Medical Single Image Super-Resolution
This is the official implementation of the add-link-to-paper.
We provide the sample codes for training and testing, as well as the pretrained model found in the paper.

## Requirements
- Chainer >= 5.0.0
- ChainerRL >=0.5.0
- Cupy >= 5.0.0
- Future ~= 0.18.2
- Matplotlib ~=3.5.2
- Numpy >= 1.15.4
- OpenCV >= 3.4.3.18
- Pillow ~= 9.1.1
- Scikit-image ~=0.19.3
- Scipy ~= 1.7.3
- Sewar ~= 0.4.5
- Torch ~=1.12.0
- Torchvision ~=0.13.0

You can install the required libraries by the command `pip install -r requirements.txt`.

## Usage

### Training
If you want to train the model on the BSD68 dataset please run `train.py`.

### Test with pretrained models
If you want to test the pretrained model please run `test.py`.

### Custom dataset
To use a custom dataset the images must first be in the JPEG format, greyscale, and have dimension 256x256 pixels.
The validation, training, and testing text files must then be generated.
All relevant code for processing datasets for implementation in addition to post-testing metric calculations can be found in 'QOL_code.py'.

## Note
Although we used the ACDC dataset for training in our paper, this sample code only contains BSD68 training set as an example, as the datasets used in our paper were closed-source.

## Copyright notice

```
    Copyright (C) 2022 Alix Bouffard

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
```

The following license applies solely to 'common.py' and 'QOL_code.py'.

```
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>
```


## References
Fully Convolutional Network with Multi-Step Reinforcement Learning for Image Processing - https://arxiv.org/pdf/1811.04323v2.pdf

Multi-Step Reinforcement Learning for Single Image Super-Resolution - https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Vassilo_Multi-Step_Reinforcement_Learning_for_Single_Image_Super-Resolution_CVPRW_2020_paper.pdf

BSD68 dataset - https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html

Automated Cardiac Diagnosis Challenge - https://acdc.creatis.insa-lyon.fr/description/databases.html

Multimodal Brain Tumor Segmentation Challenge 2018 - https://www.med.upenn.edu/sbia/brats2018/data.html

Alzheimer's Disease NeuroImaging Initiative - https://adni.loni.usc.edu/
