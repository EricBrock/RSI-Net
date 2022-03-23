import argparse
import torch
import random
import cv2
import warnings
import torch.nn as nn
import numpy as np
import pytorch_msssim as torchssim
from tqdm import tqdm
from torch.backends import cudnn

import fjn_util
from DataLoader import Color_BGR_Data_Loader, SR_BGR_Data_Loader
from model import net

warnings.filterwarnings('ignore')

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='colorx2', choices=['color', 'colorx2', 'srx2', 'srx4', 'srx8'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--img_test_path', type=str, default='./data/test/label/')
    parser.add_argument('--romate', type=str2bool, default=False)
    parser.add_argument('--flip', type=str2bool, default=False)
    parser.add_argument('--normalization', type=str2bool, default=False)
    parser.add_argument('--log_path', type=str, default='./log/records/')
    parser.add_argument('--best_pkl_path', type=str, default='./pretrain/colorx2.pkl', required=True)
    parser.add_argument('--result_path', type=str, default='./result_')

    return parser.parse_args()

def main(config):
    task_scale = {'color': 1, 'colorx2': 2, 'srx2': 2, 'srx4': 4, 'srx8': 8}
    scale = task_scale.get(config.task)

    # 1. dataloader
    if config.task == 'color':
        test_data_loader = Color_BGR_Data_Loader(img_path=config.img_test_path, batch_size=1,
                                                 normalzero2one=config.normalization, shuf=False, has_name=True).loader()
    else:
        test_data_loader = SR_BGR_Data_Loader(img_path=config.img_test_path, batch_size=1,
                                              normalzero2one=config.normalization, shuf=False,
                                              sr_factor=scale, has_name=True).loader()

    # 2 model and counter
    model = net.Kong(scale=scale).to(config.device)
    checkpoint = torch.load(config.best_pkl_path)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception:
        print('can not load the checkpoint! please check the pkl')
    finally:
        pass

    test_counter = fjn_util.Model_Statistics(config.log_path,['Test_psnr', 'Test_ssim', 'Test_mae', 'Test_mse', 'Test_pearsonr_corr'])

    model.eval()
    with tqdm(total=len(test_data_loader), ascii=True) as t:
        t.set_description(('Test'))

        for step, (label, inpu, img_name) in enumerate(test_data_loader):
            inpu = inpu.float().to(config.device)
            label = label.float().to(config.device)
            img_name = img_name[0]

            with torch.no_grad():
                out_bgr_hr = model.forward(inpu)[0]

            out_bgr_hr_cpu = out_bgr_hr.cpu().data.numpy()
            label_cpu = label.cpu().data.numpy()

            normal_factor = 255 if config.normalization else 1

            out_bgr_hr_cpu = np.clip(out_bgr_hr_cpu[0] * normal_factor, 0, 255).transpose((1, 2, 0)).astype(np.uint8)
            img_bgr = np.clip(label_cpu[0] * normal_factor, 0, 255).transpose((1, 2, 0)).astype(np.uint8)


            cv2.imwrite(config.result_path + img_name, np.hstack([img_bgr, out_bgr_hr_cpu]))

            (psnr, ssim, mse, mae, pearsonr_corr) = fjn_util.cal_all_index(img_bgr, out_bgr_hr_cpu)

            test_counter.average_count['Test_psnr'].update(psnr)
            test_counter.average_count['Test_ssim'].update(ssim)
            test_counter.average_count['Test_mse'].update(mse)
            test_counter.average_count['Test_mae'].update(mae)
            test_counter.average_count['Test_pearsonr_corr'].update(pearsonr_corr)

            t.set_postfix({
                'psnr': '{0:1.5f}'.format(test_counter.average_count['Test_psnr'].avg),
                'ssim': '{0:1.5f}'.format(test_counter.average_count['Test_ssim'].avg),
                'mse': '{0:1.5f}'.format(test_counter.average_count['Test_mse'].avg),
                'mae': '{0:1.5f}'.format(test_counter.average_count['Test_mae'].avg),
                'pearsonr_corr': '{0:1.5f}'.format(test_counter.average_count['Test_pearsonr_corr'].avg),
            })
            t.update(1)

    print('psnr', test_counter.average_count['Test_psnr'].avg)
    print('ssim', test_counter.average_count['Test_ssim'].avg)
    print('mse', test_counter.average_count['Test_mse'].avg)
    print('mae', test_counter.average_count['Test_mae'].avg)
    print('pearsonr_corr', test_counter.average_count['Test_pearsonr_corr'].avg)



if __name__ == "__main__":
    config = get_parameters()

    assert config.task in ['color', 'colorx2', 'srx2', 'srx4', 'srx8'], 'wrong task'

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    # torch.cuda.manual_seed_all(config.seed)  # if you are using multi-GPU.
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config.result_path = config.result_path + config.task + '/'

    fjn_util.make_folder(config.log_path ,config.result_path)
    main(config)


