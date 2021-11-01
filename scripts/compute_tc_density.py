import argparse

from utils import utils
from collections import Counter
from utils import plotters
from sklearn.metrics import cohen_kappa_score, classification_report, balanced_accuracy_score, accuracy_score
from pathlib import Path
import pandas as pd


def main(args=None):
    # TODO should maybe organise these files into a csv. In fact, yes I will...
    # Get ground truths
    test_list = open('models/TestData.txt')
    test_list_patches = [image.partition('\t')[0] for image in test_list.readlines()]
    ground_truth = [int(cls[-5]) for cls in test_list_patches]
    test_images = open('models/MIM_Test_list.txt').read().split()

    # Get model predictions from specified results file.
    y_preds = open('models/Model_2/model_2_round_2_result_2021-08-20 14.56.40.184362.txt').read().split()
    y_preds = [int(pred) for pred in y_preds]

    tcd_list = []
    it_preds = iter(y_preds)
    for i, image in enumerate(test_images):
        svs_name = []
        t = []
        pred = []
        for patch in test_list_patches:
            if image.split('.')[0] == patch.split('_')[0]:
                svs_name.append(patch.partition('_')[0])
                t.append(int(patch[-5]))
                pred.append(next(it_preds))
        tcd_list.append([svs_name, t, pred])

    tcd_scores_t = []
    tcd_scores_y = []
    acc_per_image = []

    # Per image TCD
    for svs_patch_preds in tcd_list:
        svs_patch_preds[1] = utils.merge_classes([3, 4, 5, 6, 7, 8], data=svs_patch_preds[1], target=2)
        svs_patch_preds[2] = utils.merge_classes([3, 4, 5, 6, 7, 8], data=svs_patch_preds[2], target=2)
        acc_per_image.append(accuracy_score(svs_patch_preds[1], svs_patch_preds[2]))
        sums_per_class_t = Counter(svs_patch_preds[1])
        sums_per_class_y = Counter(svs_patch_preds[2])
        tcd_scores_t.append((sums_per_class_t[1] / (sums_per_class_t[1] + sums_per_class_t[2])) * 100)
        tcd_scores_y.append((sums_per_class_y[1] / (sums_per_class_y[1] + sums_per_class_y[2])) * 100)

    plotters.bland_altman_plot(tcd_scores_t, tcd_scores_y)
    plotters.accuracy_per_image_plot(acc_per_image)

    # 0 vs 1 vs 2-8 i.e. 3 groups with all of the informative non-tumour cell groups combined.
    # 0+6 vs 1 vs 2/3/4/5/7/8 i.e. same as above but including tumour lumen in the non informative category on the basis
    # that it is all blank space

    y_merged = utils.merge_classes([6], data=y_preds, target=0)
    gt_merged = utils.merge_classes([6], data=ground_truth, target=0)

    component = ['0: Non-Inf',
                 '1: Tumour',
                 '2: Str/Fib',
                 '3: Necr',
                 '4: Vessel',
                 '5: Infl',
                 '7: Mucin',
                 '8: Muscle']

    print(cohen_kappa_score(gt_merged, y_merged))
    print(balanced_accuracy_score(gt_merged, y_merged))
    print(classification_report(gt_merged, y_merged))
    plotters.plot_confusion(y_true=gt_merged, y_pred=y_merged, fmt='d', labels=component,
                            title='Model 2 - [0,6] vs [1] vs [2,3,4,5,7,8]')

    y_merged = utils.merge_classes([2, 3, 4, 5, 6, 7, 8], data=y_preds, target=2)
    gt_merged = utils.merge_classes([2, 3, 4, 5, 6, 7, 8], data=ground_truth, target=2)

    component = ['0: Non-Inf',
                 '1: Tumour',
                 '2: Str/Fib']

    print(cohen_kappa_score(gt_merged, y_merged))
    print(balanced_accuracy_score(gt_merged, y_merged))
    print(classification_report(gt_merged, y_merged))
    plotters.plot_confusion(y_true=gt_merged, y_pred=y_merged, fmt='d', labels=component,
                            title='Model 2 - [0] vs [1] vs [2-8]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Computes the Tumour Cell Density metrics per image.')

    parser.add_argument("-t", "--test-list", type=Path,
                        help=r'The test image list. This is used to align the images with the predictions',
                        default='config\\default_configuration.yaml')

    parser.add_argument('-s', '--svs-listing', type=str,
                        help=r'Extract from svs in this svs file listing.',
                        default=False)

    parser.add_argument('-p', '--predictions', type=Path,
                        help=r'A file containing the list of model predictions',
                        default=None)

    arguments = parser.parse_args()

    main(args=arguments)
