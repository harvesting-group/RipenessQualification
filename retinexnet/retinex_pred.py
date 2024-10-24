import os
import time
import random

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max= torch.max(input_im, dim=1, keepdim=True)[0]
        input_img= torch.cat((input_max, input_im), dim=1)
        feats0   = self.net1_conv0(input_img)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)
        R        = torch.sigmoid(outs[:, 0:3, :, :])
        L        = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L

class RelightNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(RelightNet, self).__init__()

        self.relu         = nn.ReLU()
        self.net2_conv0_1 = nn.Conv2d(4, channel, kernel_size,
                                      padding=1, padding_mode='replicate')

        self.net2_conv1_1 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_2 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')
        self.net2_conv1_3 = nn.Conv2d(channel, channel, kernel_size, stride=2,
                                      padding=1, padding_mode='replicate')

        self.net2_deconv1_1= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_2= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')
        self.net2_deconv1_3= nn.Conv2d(channel*2, channel, kernel_size,
                                       padding=1, padding_mode='replicate')

        self.net2_fusion = nn.Conv2d(channel*3, channel, kernel_size=1,
                                     padding=1, padding_mode='replicate')
        self.net2_output = nn.Conv2d(channel, 1, kernel_size=3, padding=0)

    def forward(self, input_L, input_R):
        input_img = torch.cat((input_R, input_L), dim=1)
        out0      = self.net2_conv0_1(input_img)
        out1      = self.relu(self.net2_conv1_1(out0))
        out2      = self.relu(self.net2_conv1_2(out1))
        out3      = self.relu(self.net2_conv1_3(out2))

        out3_up   = F.interpolate(out3, size=(out2.size()[2], out2.size()[3]))
        deconv1   = self.relu(self.net2_deconv1_1(torch.cat((out3_up, out2), dim=1)))
        deconv1_up= F.interpolate(deconv1, size=(out1.size()[2], out1.size()[3]))
        deconv2   = self.relu(self.net2_deconv1_2(torch.cat((deconv1_up, out1), dim=1)))
        deconv2_up= F.interpolate(deconv2, size=(out0.size()[2], out0.size()[3]))
        deconv3   = self.relu(self.net2_deconv1_3(torch.cat((deconv2_up, out0), dim=1)))

        deconv1_rs= F.interpolate(deconv1, size=(input_R.size()[2], input_R.size()[3]))
        deconv2_rs= F.interpolate(deconv2, size=(input_R.size()[2], input_R.size()[3]))
        feats_all = torch.cat((deconv1_rs, deconv2_rs, deconv3), dim=1)
        feats_fus = self.net2_fusion(feats_all)
        output    = self.net2_output(feats_fus)
        return output


class RetinexNet(nn.Module):
    def __init__(self):
        super(RetinexNet, self).__init__()

        self.DecomNet  = DecomNet()
        self.RelightNet= RelightNet()

    def forward(self, input_low):
        # Forward DecompNet
        input_low = Variable(torch.FloatTensor(torch.from_numpy(input_low))).cuda()
        
        R_low, I_low   = self.DecomNet(input_low)
        # R_high, I_high = self.DecomNet(input_high)

        # Forward RelightNet
        I_delta = self.RelightNet(I_low, R_low)

        # Other variables
        I_low_3  = torch.cat((I_low, I_low, I_low), dim=1)
        # I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        I_delta_3= torch.cat((I_delta, I_delta, I_delta), dim=1)

        # Compute losses
        # self.recon_loss_low  = F.l1_loss(R_low * I_low_3,  input_low)
        # self.recon_loss_high = F.l1_loss(R_high * I_high_3, input_high)
        # self.recon_loss_mutal_low = F.l1_loss(R_high * I_low_3, input_low)
        # self.recon_loss_mutal_high = F.l1_loss(R_low * I_high_3, input_high)
        # self.equal_R_loss = F.l1_loss(R_low,  R_high.detach())
        # self.relight_loss = F.l1_loss(R_low * I_delta_3, input_high)

        # self.Ismooth_loss_low   = self.smooth(I_low, R_low)
        # self.Ismooth_loss_high  = self.smooth(I_high, R_high)
        # self.Ismooth_loss_delta = self.smooth(I_delta, R_low)

        # self.loss_Decom = self.recon_loss_low + \
        #                   self.recon_loss_high + \
        #                   0.001 * self.recon_loss_mutal_low + \
        #                   0.001 * self.recon_loss_mutal_high + \
        #                   0.1 * self.Ismooth_loss_low + \
        #                   0.1 * self.Ismooth_loss_high + \
        #                   0.01 * self.equal_R_loss
        # self.loss_Relight = self.relight_loss + \
        #                     3 * self.Ismooth_loss_delta

        self.output_R_low   = R_low.detach().cpu()
        self.output_I_low   = I_low_3.detach().cpu()
        self.output_I_delta = I_delta_3.detach().cpu()
        self.output_S       = R_low.detach().cpu() * I_delta_3.detach().cpu()

    # def gradient(self, input_tensor, direction):
    #     self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
    #     self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

    #     if direction == "x":
    #         kernel = self.smooth_kernel_x
    #     elif direction == "y":
    #         kernel = self.smooth_kernel_y
    #     grad_out = torch.abs(F.conv2d(input_tensor, kernel,
    #                                   stride=1, padding=1))
    #     return grad_out

    # def ave_gradient(self, input_tensor, direction):
    #     return F.avg_pool2d(self.gradient(input_tensor, direction),
    #                         kernel_size=3, stride=1, padding=1)

    # def smooth(self, input_I, input_R):
    #     input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
    #     input_R = torch.unsqueeze(input_R, dim=1)
    #     return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
    #                       self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

    def evaluate(self, epoch_num, eval_low_data_names, vis_dir, train_phase):
        print("Evaluating for phase %s / epoch %d..." % (train_phase, epoch_num))

        for idx in range(len(eval_low_data_names)):
            eval_low_img   = Image.open(eval_low_data_names[idx])
            eval_low_img   = np.array(eval_low_img, dtype="float32")/255.0
            eval_low_img   = np.transpose(eval_low_img, (2, 0, 1))
            input_low_eval = np.expand_dims(eval_low_img, axis=0)

            if train_phase == "Decom":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                input    = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                cat_image= np.concatenate([input, result_1, result_2], axis=2) #ori, R, L 
            if train_phase == "Relight":
                self.forward(input_low_eval, input_low_eval)
                result_1 = self.output_R_low
                result_2 = self.output_I_low
                result_3 = self.output_I_delta
                result_4 = self.output_S
                input = np.squeeze(input_low_eval)
                result_1 = np.squeeze(result_1)
                result_2 = np.squeeze(result_2)
                result_3 = np.squeeze(result_3)
                result_4 = np.squeeze(result_4)
                cat_image= np.concatenate([input, result_1, result_2, result_3, result_4], axis=2) #ori R_Low I_low I_adjust new image

            cat_image = np.transpose(cat_image, (1, 2, 0))
            # print(cat_image.shape)
            im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
            # filepath = os.path.join(vis_dir, 'eval_%s_%d_%d.png' %
            #            (train_phase, idx + 1, epoch_num))
            # im.save(filepath[:-4] + '.jpg')


    def save(self, iter_num, ckpt_dir):
        save_dir = ckpt_dir + '/' + self.train_phase + '/'
        save_name= save_dir + '/' + str(iter_num) + '.tar'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if self.train_phase == 'Decom':
            torch.save(self.DecomNet.state_dict(), save_name)
        elif self.train_phase == 'Relight':
            torch.save(self.RelightNet.state_dict(),save_name)

    def load(self, ckpt_dir):
        load_dir   = ckpt_dir + '/' + self.train_phase + '/'
        if os.path.exists(load_dir):
            load_ckpts = os.listdir(load_dir)
            load_ckpts.sort()
            load_ckpts = sorted(load_ckpts, key=len)
            if len(load_ckpts)>0:
                load_ckpt  = load_ckpts[-1]
                global_step= int(load_ckpt[:-4])
                ckpt_dict  = torch.load(load_dir + load_ckpt)
                if self.train_phase == 'Decom':
                    self.DecomNet.load_state_dict(ckpt_dict)
                elif self.train_phase == 'Relight':
                    self.RelightNet.load_state_dict(ckpt_dict)
                return True, global_step
            else:
                return False, 0
        else:
            return False, 0

    def predict(self,
                test_low_data_names,
                res_dir,
                ckpt_dir):

        self.train_phase= 'Decom'
        load_model_status, _ = self.load(ckpt_dir)

        self.train_phase= 'Relight'
        load_model_status, _ = self.load(ckpt_dir)

        test_img_path  = test_low_data_names

        test_low_img   = Image.open(test_img_path)
        test_low_img   = np.array(test_low_img, dtype="float32")/255.0
        test_low_img   = np.transpose(test_low_img, (2, 0, 1))
        input_low_test = np.expand_dims(test_low_img, axis=0)

        self.forward(input_low_test)

        result_4 = self.output_S

        result_4 = np.squeeze(result_4)

        
        cat_image= np.concatenate([result_4], axis=2)

        cat_image = np.transpose(cat_image, (1, 2, 0))

        im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))        
        return im # Predict for the test images
