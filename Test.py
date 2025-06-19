import torch
import torch.nn.functional as F
import torch.utils.data as data
import os, argparse
from lib.FINet import FINet
from utils.dataloader import test_dataset
import cv2

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=224, help='testing size')
    parser.add_argument('--dataset_root', type=str,
                        default='/opt/data/private/datasets/ultrasound', 
                        help='path to train dataset')
    parser.add_argument('--dataset_name', type=str,
                            default='breast/BUSI1', 
                            help='dataset')
    parser.add_argument('--pth_path', type=str, 
                        default='/opt/data/private/Second/FINet/model_pth/breast/BUSI1/opt/data/private/datasets/ultrasound/model.pth')
    parser.add_argument('--test_save', type=str,
                        default='./result_map/breast/BUSI1/FINet')  # 224_no_pretrained  224_pretrained
    
    parser.add_argument('--batch_size', type=int,
                        default=1, help='training batch size')
    parser.add_argument('--num_workers', type=int,
                        default=8, help='num_workers') 
    
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    
    config = vars(parser.parse_args())
    
    
    # build models
    torch.cuda.set_device(0)  # set your gpu device
    model = FINet().cuda()
    model.load_state_dict(torch.load(config['pth_path'], map_location='cpu'))
    model.eval()

    test_path = os.path.join(config['dataset_root'], config['dataset_name'], 'test')
    dataset = test_dataset(test_path, config['image_size'])
    test_loader = data.data_loader = data.DataLoader(dataset=dataset,
                            batch_size=1,
                            shuffle=False)
    

    for i, pack in enumerate(test_loader):
        image, gt, name = pack
        image = image.cuda()
        gt = gt.cuda()
        P, _, _, _, _ = model(image)
        res = F.interpolate(P, size=(gt.shape[-2], gt.shape[-1]), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        

        if not os.path.exists(config['test_save']):
            os.makedirs(config['test_save'])
        cv2.imwrite(os.path.join(config['test_save'], ''.join(name).split('/')[-1]), res * 255)

    print('Finish!')
