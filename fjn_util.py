'''
Common code for image processing
Author  : Terminator(FJN)
Date    : 2022-03-19
'''
from collections import OrderedDict
import torch.nn as nn
import torch, torch.optim
import os
import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import matplotlib.pyplot as plt
import math


# -------------------------- img index --------------------------

def check_img_data_range(img: np.uint8) -> float:
    return 255 if (img.dtype == np.uint8) else 1.0


def psnr(img1, img2) -> float:
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    return peak_signal_noise_ratio(img1, img2, data_range=check_img_data_range(img1))


def ssim(img1, img2):
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    assert len(img1.shape) in [2, 3], 'shape doesnt match'
    return structural_similarity(img1, img2, multichannel=(len(img1.shape) == 3), data_range=check_img_data_range(img1))


def mse(img1, img2):
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    assert len(img1.shape) in [2, 3], 'shape doesnt match'
    if check_img_data_range(img1) == 1.0:
        img1, img2 = (img1 * 255.).astype(np.uint8), (img2 * 255.).astype(np.uint8)
    return mean_squared_error(img1, img2)


def mae(img1, img2):
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    assert len(img1.shape) in [2, 3], 'shape doesnt match'
    if check_img_data_range(img1) == 1.0:
        img1, img2 = (img1 * 255.).astype(np.uint8), (img2 * 255.).astype(np.uint8)
    return np.mean(abs(img1 - img2))


def pearsonr_corr(img1, img2):
    # 1. check img's dtype and shape
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    assert len(img1.shape) in [2, 3], 'shape doesnt match'

    # 2. def pearsonr for 2D shape list
    def cal_pear(img1, img2):
        result = 0
        for i in range(img1.shape[0]):
            pear = pearsonr(img1[i], img2[i])[0]
            if np.isnan(pear):
                result += 0
            else:
                result += pear
        return result

    # 3. select single channel or multi-channel
    if len(img1.shape) == 2:
        return cal_pear(img1, img2) / img1.shape[0]
    else:
        result = 0
        img1s = np.array_split(img1, 3, axis=2)
        img2s = np.array_split(img2, 3, axis=2)
        for i in range(len(img1s)):
            result += cal_pear(np.squeeze(img1s[i], 2), np.squeeze(img2s[i], 2))
        return result / img1s[0].shape[0] / 3


def cal_all_index(img1, img2) -> list:
    assert img1.dtype == img2.dtype, 'dtype doesnt match'
    assert img1.shape == img2.shape, 'shape doesnt match'
    assert len(img1.shape) in [2, 3], 'shape doesnt match'
    return [psnr(img1, img2), ssim(img1, img2), mse(img1, img2), mae(img1, img2), pearsonr_corr(img1, img2)]


class AverageCounter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


class AccuracyCounter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.accu = 0
        self.sum = 0
        self.correct = 0

    def update(self, val):
        if val:
            self.correct += 1
        self.sum += 1
        self.accu = self.correct / self.sum


def make_folder(*args):
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)


# # 返回 list  [ 总参数量，  输入大小MB，  输出大小MB，  总参数量大小MB ]
# result = model_summary(model.to(torch.device('cuda:0')), input_size=(3, 224, 224), batch_size=1, show_detail=True)
def model_summary(model, input_size, batch_size=-1, device="cuda", show_detail=False):
    '''
    :param model:
    :param input_size:
    :param batch_size:
    :param device:
    :param show_detail:
    :return: [
        total params of model
        input size (related with batchsize)                     MB
        output size (related with batchsize and input size)     MB
        params size of model                                    MB
     ]
    '''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size
            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"
    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    if isinstance(input_size, tuple):
        input_size = [input_size]
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    summary = OrderedDict()
    hooks = []
    model.apply(register_hook)
    model(*x)
    for h in hooks:
        h.remove()
    if show_detail:
        print("----------------------------------------------------------------")
        print("{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #"))
        print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]), )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        if show_detail:
            print(line_new)
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size
    params_list = [total_params.numpy().tolist(), total_input_size, total_output_size, total_params_size]
    if show_detail:
        print("================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params - trainable_params))
        print("----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print("----------------------------------------------------------------")
    return params_list


class Model_Statistics():

    def __init__(self,
                 statistics_path: str,
                 param_list: list):
        assert len(param_list) > 1, 'MyAssert: param_list must contain two more values'

        self.txt_path = statistics_path + '/txt/'
        self.graph_path = statistics_path + '/graph/'

        make_folder(self.txt_path, self.graph_path)

        self.list_count, self.average_count = {}, {}
        self.param_list = param_list

        for item in self.param_list:
            self.list_count[item] = []
            self.average_count[item] = AverageCounter()

        self.read_from_txt()
        self.reset_all_counter()

    # append new value to average_count
    def update_counter(self, name, value):
        assert self.average_count.__contains__(name), 'MyAssert: not found ' + name + ' in average_count'
        self.average_count[name].update(value)
        # print(value)
        # print('sum: ', self.average_count[name].sum)
        # print('count: ', self.average_count[name].count)
        # print('avg: ', self.average_count[name].avg)
        pass

    # append new value to list_count
    def update_list(self):
        for name in self.param_list:
            assert self.list_count.__contains__(name), 'MyAssert: not found ' + name + ' in list_count'
            self.list_count[name].append(self.average_count[name].avg)
        pass

    # reset all Averagecounter
    def reset_all_counter(self):
        for averagecounter in self.average_count:
            self.average_count[averagecounter].reset()
        pass

    # draw one item in count_list
    def draw(self, name_list, saved=False, color_start=0, show=False):
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.keys())  # 颜色变化
        title_name = ''
        for i, name in enumerate(name_list):
            plt.plot(list(range(len(self.list_count[name]))), self.list_count[name], linestyle="-", marker="",
                     linewidth=2, color=mcolors.TABLEAU_COLORS[colors[i + color_start]], label=name)
            title_name += (name + ' ')
        plt.legend(loc='upper right')
        plt.title(title_name)
        plt.xlabel('epoch')
        plt.ylabel('value')
        if saved:
            plt.savefig(self.graph_path + title_name + '.png')
        if show:
            plt.show()
        pass

    # draw all items in count_list
    def draw_all(self):
        i = 0
        for name in self.param_list:
            self.draw(name, True, color_start=i)
            i+=1

    # write all records to the txt
    def write_all_to_txt(self):
        for name in self.param_list:
            file = open(os.path.join(self.txt_path, name + '.txt'), 'w')
            for item in self.list_count[name]:
                file.write(str(item) + '\n')
            # print('write ' + name +' to ' + os.path.join(self.self.statistics_path, name+'.txt') + ' successfully!')
        pass

    # read records from statistics_path
    def read_from_txt(self):
        for name in self.param_list:
            if (os.path.exists(os.path.join(self.txt_path, name + '.txt'))):
                for line in open(os.path.join(self.txt_path, name + '.txt'), "r"):  # 设置文件对象并读取每一行文件
                    self.list_count[name].append(float(line[:-1]))


# combine_graphs_from_txt(src_list=['./txt/txt1/loss.txt', './txt/txt2/loss.txt'], col=3, name_list=['StepLR(1,0.9)', 'StepLR(2,0.9)'], save_path='./graph/', save_name='combine')
def combine_graphs_from_txt(src_list, name_list, save_path, save_name, col=2):
    row = math.ceil(len(src_list) / col)
    fig, ax = plt.subplots(nrows=row, ncols=col, constrained_layout=True)
    fig.set_size_inches(col * 3, row * 3)
    axes = ax.flatten()
    for step, txt_path in enumerate(src_list):
        y_list = []
        if (os.path.exists(txt_path)):
            for line in open(txt_path, "r"):  # 设置文件对象并读取每一行文件
                y_list.append(float(line[:-1]))
            x_list = list(range(1, len(y_list) + 1))
            axes[step].plot(x_list, y_list, linestyle="-", marker="", linewidth=1)
            axes[step].set_title(str(name_list[step]))
    plt.savefig(save_path + save_name + '.png')
    plt.show()
    pass
