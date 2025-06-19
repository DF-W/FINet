import os
import numpy as np
from PIL import Image
from tabulate import tabulate
from utils.eval_functions import *
from utils.utils import *


def evaluate():

    dataset = 'breast/BUSI1'
    method = 'FINet'
    result_path = './result_map/'
    Thresholds = np.linspace(1, 0, 256)
    headers = ['meanDic', 'meanIoU', 'wFm', 'Sm', 'meanEm', 'mae', 'maxEm', 'maxDic', 'maxIoU', 'meanSen', 'maxSen', 'meanSpe', 'maxSpe']
    results = []
    
    pred_root = os.path.join(result_path, dataset, method) 
    gt_root = os.path.join('/opt/data/private/datasets/ultrasound', dataset, 'test', 'masks')

    preds = os.listdir(pred_root)
    gts = os.listdir(gt_root)
    preds.sort()
    gts.sort()

    threshold_Fmeasure = np.zeros((len(preds), len(Thresholds)))
    threshold_Emeasure = np.zeros((len(preds), len(Thresholds)))
    threshold_IoU = np.zeros((len(preds), len(Thresholds)))
    threshold_Sensitivity = np.zeros((len(preds), len(Thresholds)))
    threshold_Specificity = np.zeros((len(preds), len(Thresholds)))
    threshold_Dice = np.zeros((len(preds), len(Thresholds)))

    Smeasure = np.zeros(len(preds))
    wFmeasure = np.zeros(len(preds))
    MAE = np.zeros(len(preds))

    samples = enumerate(zip(preds, gts))

    for i, sample in samples:
        pred, gt = sample
        assert os.path.splitext(pred)[0] == os.path.splitext(gt)[0].split('_')[0]

        pred_mask = np.array(Image.open(os.path.join(pred_root, pred)))
        gt_mask = np.array(Image.open(os.path.join(gt_root, gt)).convert('1'))
        
        if len(pred_mask.shape) != 2:
            pred_mask = pred_mask[:, :, 0]
        if len(gt_mask.shape) != 2:
            gt_mask = gt_mask[:, :, 0]
        
        assert pred_mask.shape == gt_mask.shape

        gt_mask = gt_mask.astype(np.float64) 
        pred_mask = pred_mask.astype(np.float64) / 255

        Smeasure[i] = StructureMeasure(pred_mask, gt_mask)
        wFmeasure[i] = original_WFb(pred_mask, gt_mask)
        MAE[i] = np.mean(np.abs(gt_mask - pred_mask))

        threshold_E = np.zeros(len(Thresholds))
        threshold_F = np.zeros(len(Thresholds))
        threshold_Pr = np.zeros(len(Thresholds))
        threshold_Rec = np.zeros(len(Thresholds))
        threshold_Iou = np.zeros(len(Thresholds))
        threshold_Spe = np.zeros(len(Thresholds))
        threshold_Dic = np.zeros(len(Thresholds))

        for j, threshold in enumerate(Thresholds):
            threshold_Pr[j], threshold_Rec[j], threshold_Spe[j], threshold_Dic[j], threshold_F[j], threshold_Iou[j] = Fmeasure_calu(pred_mask, gt_mask, threshold)

            Bi_pred = np.zeros_like(pred_mask)
            Bi_pred[pred_mask >= threshold] = 1
            threshold_E[j] = EnhancedMeasure(Bi_pred, gt_mask)
        
        threshold_Emeasure[i, :] = threshold_E
        threshold_Fmeasure[i, :] = threshold_F
        threshold_Sensitivity[i, :] = threshold_Rec
        threshold_Specificity[i, :] = threshold_Spe
        threshold_Dice[i, :] = threshold_Dic
        threshold_IoU[i, :] = threshold_Iou

    result = []

    mae = np.mean(MAE)
    Sm = np.mean(Smeasure)
    wFm = np.mean(wFmeasure)

    column_E = np.mean(threshold_Emeasure, axis=0)
    meanEm = np.mean(column_E)
    maxEm = np.max(column_E)

    column_Sen = np.mean(threshold_Sensitivity, axis=0)
    meanSen = np.mean(column_Sen)
    maxSen = np.max(column_Sen)

    column_Spe = np.mean(threshold_Specificity, axis=0)
    meanSpe = np.mean(column_Spe)
    maxSpe = np.max(column_Spe)

    column_Dic = np.mean(threshold_Dice, axis=0)
    meanDic = np.mean(column_Dic)
    maxDic = np.max(column_Dic)

    column_IoU = np.mean(threshold_IoU, axis=0)
    meanIoU = np.mean(column_IoU)
    maxIoU = np.max(column_IoU)

    result.extend([meanDic, meanIoU, wFm, Sm, meanEm, mae, maxEm, maxDic, maxIoU, meanSen, maxSen, meanSpe, maxSpe])
    results.append([dataset, method, *result])
    

    csv = os.path.join(result_path, 'result' + '.csv')
    if os.path.isfile(csv) is True:
        csv = open(csv, 'a')
    else:
        csv = open(csv, 'w')
        csv.write(', '.join(['dataset', 'method', *headers]) + '\n')

    out_str = dataset + ',' + method + ','
    for metric in result:
        out_str += '{:.4f}'.format(metric) + ','
    out_str += '\n'

    csv.write(out_str)
    csv.close()

    
    tab = tabulate(results, headers=['dataset', 'method', *headers], floatfmt=".3f")
    return tab

if __name__ == "__main__":
    tab = evaluate()
    print(tab)
