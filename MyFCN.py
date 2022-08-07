# SPDX-License-Identifier: AGPL-3.0-or-later

from abc import ABC

import chainer
import chainer.links as links
import chainer.functions as functions

import numpy as np

import chainerrl
from chainerrl.agents import a3c


class MyFcnTrained(chainer.Chain, a3c.A3CModel, ABC):

    def __init__(self, n_actions):
        super(MyFcnTrained, self).__init__(
            conv1=links.Convolution2D(1, 64, 3, stride=1, pad=1, nobias=False, initialW=None, initial_bias=None),
            diconv2=DilatedConvBlock(2, None, None),
            diconv3=DilatedConvBlock(3, None, None),
            diconv4=DilatedConvBlock(4, None, None),
            diconv5_pi=DilatedConvBlock(3, None, None),
            diconv6_pi=DilatedConvBlock(2, None, None),
            conv7_Wz=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv7_Uz=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv7_Wr=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv7_Ur=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv7_W=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv7_U=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv8_pi=chainerrl.policies.SoftmaxPolicy(
                links.Convolution2D(64, n_actions, 3, stride=1, pad=1, nobias=False, initialW=None)),
            diconv5_V=DilatedConvBlock(3, None, None),
            diconv6_V=DilatedConvBlock(2, None, None),
            conv7_V=links.Convolution2D(64, 1, 3, stride=1, pad=1, nobias=False, initialW=None, initial_bias=None),
        )


class DilatedConvBlock(chainer.Chain):

    def __init__(self, d_factor, weight, bias):
        super(DilatedConvBlock, self).__init__(
            diconv=links.DilatedConvolution2D(in_channels=64, out_channels=64, ksize=3, stride=1, pad=d_factor,
                                              dilate=d_factor, nobias=False, initialW=weight, initial_bias=bias),
            # bn=links.BatchNormalization(64)
        )

        self.train = True

    def __call__(self, x):
        h = functions.relu(self.diconv(x))
        # h = functions.relu(self.bn(self.diconv(x)))
        return h


class MyFcn(chainer.Chain, a3c.A3CModel):

    def __init__(self, n_actions):
        chainer.backends.cuda.set_max_workspace_size(7340032)
        # print("MAX WORKSPACE", chainer.backends.cuda.get_max_workspace_size())
        # chainer.print_runtime_info()
        # w = chainer.initializers.HeNormal()
        w_i = np.zeros((1, 1, 33, 33))
        w_i[:, :, 16, 16] = 1
        net = MyFcnTrained(n_actions)
        # chainer.serializers.load_npz('../deblur/model/pretrained_15.npz', net)
        super(MyFcn, self).__init__(
            conv1=links.Convolution2D(1, 64, 3, stride=1, pad=1, nobias=False, initialW=net.conv1.W.data,
                                      initial_bias=net.conv1.b.data),

            diconv2=DilatedConvBlock(2, net.diconv2.diconv.W.data, net.diconv2.diconv.b.data),
            diconv3=DilatedConvBlock(3, net.diconv3.diconv.W.data, net.diconv3.diconv.b.data),
            diconv4=DilatedConvBlock(4, net.diconv4.diconv.W.data, net.diconv4.diconv.b.data),
            diconv5_pi=DilatedConvBlock(3, net.diconv5_pi.diconv.W.data, net.diconv5_pi.diconv.b.data),
            diconv6_pi=DilatedConvBlock(2, net.diconv6_pi.diconv.W.data, net.diconv6_pi.diconv.b.data),
            conv7_Wz=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_Wz.W.data),
            conv7_Uz=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_Uz.W.data),
            conv7_Wr=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_Wr.W.data),
            conv7_Ur=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_Ur.W.data),
            conv7_W=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_W.W.data),
            conv7_U=links.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_U.W.data),

            conv8_pi=chainerrl.policies.SoftmaxPolicy(links.Convolution2D(64, n_actions, 3, stride=1, pad=1,
                                                                          nobias=False,
                                                                          initialW=net.conv8_pi.model.W.data,
                                                                          initial_bias=net.conv8_pi.model.b.data)),

            diconv5_V=DilatedConvBlock(3, net.diconv5_V.diconv.W.data, net.diconv5_V.diconv.b.data),
            diconv6_V=DilatedConvBlock(2, net.diconv6_V.diconv.W.data, net.diconv6_V.diconv.b.data),
            conv7_V=links.Convolution2D(64, 1, 3, stride=1, pad=1, nobias=False, initialW=net.conv7_V.W.data,
                                        initial_bias=net.conv7_V.b.data),
            conv_R=links.Convolution2D(1, 1, 33, stride=1, pad=16, nobias=True, initialW=w_i),
        )
        self.train = True

    def pi_and_v(self, x):
        h = functions.relu(self.conv1(x[:, 0:1, :, :]))
        h = self.diconv2(h)
        h = self.diconv3(h)
        h = self.diconv4(h)
        h_pi = self.diconv5_pi(h)
        x_t = self.diconv6_pi(h_pi)
        h_t1 = x[:, -64:, :, :]
        z_t = functions.sigmoid(self.conv7_Wz(x_t) + self.conv7_Uz(h_t1))
        r_t = functions.sigmoid(self.conv7_Wr(x_t) + self.conv7_Ur(h_t1))
        h_tilde_t = functions.tanh(self.conv7_W(x_t) + self.conv7_U(r_t * h_t1))
        h_t = (1 - z_t) * h_t1 + z_t * h_tilde_t
        pout = self.conv8_pi(h_t)

        h_v = self.diconv5_V(h)
        h_v = self.diconv6_V(h_v)
        vout = self.conv7_V(h_v)

        return pout, vout, h_t

    def conv_smooth(self, x):
        x = self.conv_R(x)

        return x
