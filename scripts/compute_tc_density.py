from utils import utils
from collections import Counter
from utils import plotters
from sklearn.metrics import cohen_kappa_score, classification_report, balanced_accuracy_score,accuracy_score
import matplotlib.pyplot as plt
import numpy as np

test_list = open('models/TestData.txt')
test_list_patches = [image.partition('\t')[0] for image in test_list.readlines()]
ground_truth = [int(cls[-5]) for cls in test_list_patches]
y_preds = open('models/Model_2/model_2_round_2_result_2021-08-20 14.56.40.184362.txt').read().split()
y_preds = [int(pred) for pred in y_preds]
test_images = open('models/MIM_Test_list.txt').read().split()

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

for svs_patch_preds in tcd_list:
    svs_patch_preds[1] = utils.merge_classes([3, 4, 5, 6, 7, 8], data=svs_patch_preds[1], target=2)
    svs_patch_preds[2] = utils.merge_classes([3, 4, 5, 6, 7, 8], data=svs_patch_preds[2], target=2)
    acc_per_image.append(accuracy_score(svs_patch_preds[1], svs_patch_preds[2]))
    sums_per_class_t = Counter(svs_patch_preds[1])
    sums_per_class_y = Counter(svs_patch_preds[2])
    tcd_scores_t.append((sums_per_class_t[1] / (sums_per_class_t[1] + sums_per_class_t[2])) * 100)
    tcd_scores_y.append((sums_per_class_y[1] / (sums_per_class_y[1] + sums_per_class_y[2])) * 100)

plotters.bland_altman_plot(tcd_scores_t, tcd_scores_y)
md = np.mean(acc_per_image)
sd = np.std(acc_per_image, axis=0)
plt.hist(acc_per_image, bins=20, histtype='stepfilled')
plt.xlabel('Balanced Accuracy')
plt.ylabel('Per image count')
plt.title('Model 2 - Distribution of accuracy per image')
plt.axvline(md, color='red', linestyle='--', label='mean')
plt.axvline(md + 1.96 * sd, color='gray', linestyle='--', label='SD +1')
plt.axvline(md - 1.96 * sd, color='gray', linestyle='--', label='SD -1')
plt.legend()
plt.show()
print('----')

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
plotters.plot_confusion(y_true=gt_merged, y_pred=y_merged, fmt='d', labels=component, title='Model 2 - [0,6] vs [1] vs [2,3,4,5,7,8]')

y_merged = utils.merge_classes([2, 3, 4, 5, 6, 7, 8], data=y_preds, target=2)
gt_merged = utils.merge_classes([2, 3, 4, 5, 6, 7, 8], data=ground_truth, target=2)

component = ['0: Non-Inf',
             '1: Tumour',
             '2: Str/Fib']

print(cohen_kappa_score(gt_merged, y_merged))
print(balanced_accuracy_score(gt_merged, y_merged))
print(classification_report(gt_merged, y_merged))
plotters.plot_confusion(y_true=gt_merged, y_pred=y_merged, fmt='d', labels=component, title='Model 2 - [0] vs [1] vs [2-8]')
