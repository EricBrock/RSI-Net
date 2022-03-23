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

    # FOR TRAIN AGAIN
    parser.add_argument('--task', type=str, default='srx2', choices=['color', 'colorx2', 'srx2', 'srx4', 'srx8'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--train_all_epoch', type=int, default=50)

    # DATA_DETAIL
    parser.add_argument('--img_train_path', type=str, default='./data/train/label/')
    parser.add_argument('--img_test_path', type=str, default='./data/test/label/')
    parser.add_argument('--romate', type=str2bool, default=True)
    parser.add_argument('--flip', type=str2bool, default=True)
    parser.add_argument('--normalization', type=str2bool, default=False)
    parser.add_argument('--batch_size', type=int, default=4)

    # SAVE_PATH
    parser.add_argument('--train_pkl_path', type=str, default='./log/pkl/train/')
    parser.add_argument('--best_pkl_path', type=str, default='./log/pkl/best/')
    parser.add_argument('--log_path', type=str, default='./log/records/')
    parser.add_argument('--process_path', type=str, default='./log/process/')
    parser.add_argument('--result_path', type=str, default='./log/result/')

    # TRAIN
    parser.add_argument('--init', type=str2bool, default=True)
    parser.add_argument('--test_step', type=int, default=5)

    return parser.parse_args()


def main(config):
    task_scale = {'color': 1, 'colorx2': 2, 'srx2': 2, 'srx4': 4, 'srx8': 8}
    scale = task_scale.get(config.task)

    # 1.1 dataloader
    if config.task == 'color':
        train_data_loader = Color_BGR_Data_Loader(img_path=config.img_train_path, batch_size=config.batch_size,
                                                  normalzero2one=config.normalization, shuf=True).loader()
    else:
        train_data_loader = SR_BGR_Data_Loader(img_path=config.img_train_path, batch_size=config.batch_size,
                                               normalzero2one=config.normalization, shuf=True,
                                               sr_factor=scale).loader()

    # 1.2 model
    train_model = net.Kong(scale=scale).to(config.device)
    loss_fun = nn.L1Loss()
    optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                                 amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_all_epoch)

    # 1.3 statistics counter
    train_best_ssim = 0
    best_psnr = 0
    train_counter = fjn_util.Model_Statistics(config.log_path, ['Train_loss', 'Train_ssim'])
    ssim_fun = torchssim.SSIM(data_range=1 if config.normalization else 255, channel=3)

    # 2 train
    for epoch in range(1, config.train_all_epoch + 1):
        train_counter.reset_all_counter()

        train_model.train()

        with tqdm(total=len(train_data_loader), ascii=True) as t:
            t.set_description(('epoch: {}/{}'.format(epoch, config.train_all_epoch)))
            for (img_bgr, img_gray_lr) in train_data_loader:
                img_gray_lr = img_gray_lr.float().to(config.device)
                img_bgr = img_bgr.float().to(config.device)

                out_bgr_hr = train_model(img_gray_lr)[0].to(config.device)

                loss = loss_fun(out_bgr_hr, img_bgr).to(config.device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_counter.update_counter('Train_loss', loss.item())
                train_counter.update_counter('Train_ssim', ssim_fun(out_bgr_hr, img_bgr).item())

                t.set_postfix({
                    'Train_loss': '{0:1.5f}'.format(train_counter.average_count['Train_loss'].avg),
                    'Train_ssim': '{0:1.5f}'.format(train_counter.average_count['Train_ssim'].avg)})
                t.update(1)

                # del temporary outputs and loss
                del img_gray_lr, img_bgr, out_bgr_hr, loss

        scheduler.step()
        train_counter.update_list()
        train_counter.draw(['Train_loss'], True, 0)
        train_counter.draw(['Train_ssim'], True, 1)

        # 2.1 save train model
        if train_best_ssim < train_counter.average_count['Train_ssim'].avg:
            torch.save({
                'model_state_dict': train_model.state_dict(),
                'epoch': epoch,
            }, config.train_pkl_path + 'train-' + str(epoch).zfill(4) + '.pkl')
            train_best_ssim = train_counter.average_count['Train_ssim'].avg

        # 2.2 test and save best model
        if epoch % config.test_step == 0:
            train_counter.write_all_to_txt()
            best_psnr = test(config, epoch, best_psnr, train_model)


def test(config, epoch, best_psnr, model, training=True):
    task_scale = {'color': 1, 'colorx2': 2, 'srx2': 2, 'srx4': 4, 'srx8': 8}
    scale = task_scale.get(config.task)

    # 1. dataloader
    if config.task == 'color':
        test_data_loader = Color_BGR_Data_Loader(img_path=config.img_test_path, batch_size=1,
                                                 normalzero2one=config.normalization, shuf=True, has_name=True).loader()
    else:
        test_data_loader = SR_BGR_Data_Loader(img_path=config.img_test_path, batch_size=1,
                                              normalzero2one=config.normalization, shuf=True,
                                              sr_factor=scale, has_name=True).loader()

    # 2 counter
    test_counter = fjn_util.Model_Statistics(config.log_path,
                                             ['Test_psnr', 'Test_ssim', 'Test_mae', 'Test_mse', 'Test_pearsonr_corr'])

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

            if training:
                if step < 10:
                    cv2.imwrite(config.process_path + 'train' + str(epoch).zfill(3) + '_' + str(step).zfill(2) + '.png',
                                np.hstack([img_bgr, out_bgr_hr_cpu]))
            else:
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

    if training:
        if test_counter.average_count['Test_psnr'].avg > best_psnr:
            best_psnr = test_counter.average_count['Test_psnr'].avg
            torch.save({
                'best_epoch': epoch,
                'best_psnr': best_psnr,
                'best_ssim': test_counter.average_count['Test_ssim'].avg,
                'model_state_dict': model.state_dict(),
            }, config.best_pkl_path + 'best-' + str(epoch).zfill(4) + '.pkl')

    if not training:
        print('psnr', test_counter.average_count['Test_psnr'].avg)
        print('ssim', test_counter.average_count['Test_ssim'].avg)
        print('mse', test_counter.average_count['Test_mse'].avg)
        print('mae', test_counter.average_count['Test_mae'].avg)
        print('pearsonr_corr', test_counter.average_count['Test_pearsonr_corr'].avg)

    return best_psnr


if __name__ == "__main__":
    config = get_parameters()

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    # torch.cuda.manual_seed_all(config.seed)  # if you are using multi-GPU.
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    fjn_util.make_folder(config.train_pkl_path, config.best_pkl_path, config.log_path, config.process_path,
                         config.result_path)

    assert config.task in ['color', 'colorx2', 'srx2', 'srx4', 'srx8'], 'wrong task'
    main(config)
