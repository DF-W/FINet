import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from lib.FINet import FINet
from utils.dataloader import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import logging


def log_setting(save_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    file_handler = logging.FileHandler(os.path.join(save_path, "log.log"), encoding='utf8')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


def train(train_loader, model, optimizer, epoch):
    model.train()

    loss_1_record = AvgMeter()
    loss_2_record = AvgMeter()
    loss_3_record = AvgMeter()
    loss_4_record = AvgMeter()
    loss_5_record = AvgMeter()

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        # ---- data prepare ----
        images, gts, egs = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()
        egs = Variable(egs).cuda()

        eg1 = F.interpolate(egs, size=(config['image_size']//4, config['image_size']//4), mode='bilinear', align_corners=True)
        eg2 = F.interpolate(egs, size=(config['image_size']//8, config['image_size']//8), mode='bilinear', align_corners=True)
        eg3 = F.interpolate(egs, size=(config['image_size']//16, config['image_size']//16), mode='bilinear', align_corners=True)
        eg4 = F.interpolate(egs, size=(config['image_size']//32, config['image_size']//32), mode='bilinear', align_corners=True)
        # ---- forward ----
        P, e1, e2, e3, e4 = model(images)
        # ---- loss function ----
        loss_1 = structure_loss(P, gts)
        loss_2 = dice_loss(e1, eg1)
        loss_3 = dice_loss(e2, eg2)
        loss_4 = dice_loss(e3, eg3)
        loss_5 = dice_loss(e4, eg4)

        loss = loss_1 + 0.45*loss_2 + 0.45*loss_3 + 0.45*loss_4 + 0.45*loss_5
        # ---- backward ----
        loss.backward()
        clip_gradient(optimizer, config['clip'])
        optimizer.step()
        # ---- recording loss ----
        # if rate == 1:
        loss_1_record.update(loss_1.data, config['batchsize'])
        loss_2_record.update(loss_2.data, config['batchsize'])
        loss_3_record.update(loss_3.data, config['batchsize'])
        loss_4_record.update(loss_4.data, config['batchsize'])
        loss_5_record.update(loss_5.data, config['batchsize'])
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            logger.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:0.8f}, lateral-1: [{:0.4f}], lateral-2: [{:0.4f}], lateral-3: [{:0.4f}], lateral-4: [{:0.4f}], lateral-5: [{:0.4f}]]'.format(
                datetime.now(), epoch, config['epoch'], i, total_step, optimizer.param_groups[0]['lr'], loss_1_record.show(), loss_2_record.show(), loss_3_record.show(), loss_4_record.show(), loss_5_record.show()))
    # save model 
    save_path = train_save
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) > 250:
        torch.save(model.state_dict(), save_path + '/' + 'model_{}.pth'.format(str(epoch)))
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=300, help='epoch number')
    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')
    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')
    parser.add_argument('--batchsize', type=int,
                        default=32, help='training batch size')
    parser.add_argument('--image_size', type=int,
                        default=224, help='training dataset size')
    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int,
                        default=200, help='every n epochs decay learning rate')
    parser.add_argument('--dataset_root', type=str,
                            default='/opt/data/private/datasets/ultrasound', 
                            help='path to train dataset')
    parser.add_argument('--dataset_name', type=str,
                            default='breast/BUSI1', 
                            help='dataset')
    parser.add_argument('--model_name', type=str,
                            default='FINet', 
                            help='model name')

    config = vars(parser.parse_args())
    train_save = './model_pth/{}/{}'.format(config['dataset_name'], config['model_name']) 
    logger = log_setting(train_save)

    logger.info('-' * 50)
    for key, value in config.items():
        logger.info('{} : {}'.format(key, value))
    logger.info('-' * 50)
    logger.info('--{}--{}--{}'.format(config['dataset_name'], config['model_name'], '224_pretrained'))
    
    # ---- build models ----
    torch.cuda.set_device(0)  # set your gpu device
    model = FINet().cuda()
    # model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    train_path = os.path.join(config['dataset_root'], config['dataset_name'].split('_')[0], 'train')
    train_loader = get_loader(train_path, batch_size=config['batchsize'], image_size=config['image_size'],
                            augmentation=config['augmentation'])
    total_step = len(train_loader)
    logger.info("---------------------Start Training----------------------")
    for epoch in range(1, config['epoch']):
        adjust_lr(optimizer, config['lr'], epoch, config['decay_rate'], config['decay_epoch'])
        train(train_loader, model, optimizer, epoch)
