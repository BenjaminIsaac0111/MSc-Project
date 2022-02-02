from datetime import datetime
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
import seaborn as sns

# %%
RESULTS_PATH = r'E:\JNET_Results_3\\'
results_files = [RESULTS_PATH + file for file in os.listdir(RESULTS_PATH) if file.endswith('.csv')]
# %%
select_results = 5

model_selection = os.path.split(results_files[select_results])[-1].split('_')[2].split('.')[0]
results_df = pd.read_csv(filepath_or_buffer=results_files[select_results], encoding='utf-8')
results_df.loc[results_df['Patient_id'] == '721x', 'Patient_id'] = '7201'
results_df.loc[results_df['Patient_no'] == '1x', 'Patient_no'] = '01'  # Quick fix for issue with this patient id.

results_df['Merged_Truth'] = results_df['Truth']
results_df.loc[results_df['Merged_Truth'].isin([3, 4, 5, 6, 7, 8]), 'Merged_Truth'] = 2
results_df['Merged_Preds'] = results_df['Preds']
results_df.loc[results_df['Merged_Preds'].isin([3, 4, 5, 6, 7, 8]), 'Merged_Preds'] = 2

print('Cohen Kappa - Unmerged and Merged scores\n', cohen_kappa_score(results_df['Truth'], results_df['Preds']))
print('', cohen_kappa_score(results_df['Merged_Truth'], results_df['Merged_Preds']))

print('Balanced Acc - Unmerged and Merged scores\n', balanced_accuracy_score(results_df['Truth'], results_df['Preds']))
print('', balanced_accuracy_score(results_df['Merged_Truth'], results_df['Merged_Preds']))

# %%
tcd_results = {}
for patient_id in set(results_df['Patient_id']):
    t_counts = results_df[results_df['Patient_id'] == patient_id]['Merged_Truth'].value_counts()
    p_counts = results_df[results_df['Patient_id'] == patient_id]['Merged_Preds'].value_counts()

    kappa_TCD_merged = cohen_kappa_score(results_df[results_df['Patient_id'] == patient_id]['Merged_Truth'],
                                         results_df[results_df['Patient_id'] == patient_id]['Merged_Preds'])

    bal_acc_TCD_merged = balanced_accuracy_score(results_df[results_df['Patient_id'] == patient_id]['Merged_Truth'],
                                                 results_df[results_df['Patient_id'] == patient_id]['Merged_Preds'])

    kappa = cohen_kappa_score(results_df[results_df['Patient_id'] == patient_id]['Truth'],
                              results_df[results_df['Patient_id'] == patient_id]['Preds'])

    bal_acc = balanced_accuracy_score(results_df[results_df['Patient_id'] == patient_id]['Truth'],
                                      results_df[results_df['Patient_id'] == patient_id]['Preds'])

    Image_id = set(results_df[results_df['Patient_id'] == patient_id]['Image_id'])
    if len(set(Image_id)) != 1:
        print('Found multiple image ids for Patient: {}! Image_ids found : {}'.format(patient_id, Image_id))

    tcd_results[patient_id] = {'Institute_id': str(patient_id)[:2],
                               'Patient_no': str(patient_id)[2:],
                               'Image_id': Image_id.pop(),
                               'Kappa_Score_TCD': kappa_TCD_merged,
                               'Balanced_Accuracy_TCD': bal_acc_TCD_merged,
                               'Kappa_Score': kappa,
                               'Balanced_Accuracy': bal_acc,
                               'TCD_Truth': ((t_counts[1] / (t_counts[1] + t_counts[0])) * 100),
                               'TCD_Preds': ((p_counts[1] / (p_counts[1] + p_counts[0])) * 100)}

tcd_results = pd.DataFrame.from_dict(tcd_results).T
tcd_results['Mean'] = np.mean([tcd_results['TCD_Truth'], tcd_results['TCD_Preds']], axis=0)
tcd_results['Diff'] = tcd_results['TCD_Truth'] - tcd_results['TCD_Preds']
# %%
TCD_results_dir = RESULTS_PATH + '\\TCD_RESULTS\\'
if not os.path.isdir(TCD_results_dir):
    os.mkdir(TCD_results_dir)
TCD_csv_file = TCD_results_dir
TCD_csv_file += 'CRO7_AUTOMATED_TCD_' + model_selection + '_' + str(datetime.now()).replace(':', '.') + '.csv '
# tcd_results.to_csv(TCD_csv_file)
# %%
md = np.mean(tcd_results['Diff'])  # Mean of the difference
sd = np.std(tcd_results['Diff'], axis=0)  # Standard deviation of the difference
sns.scatterplot(data=tcd_results, x='Mean', y='Diff', hue='Institute_id')
plt.xlim(min(tcd_results["Mean"]) - 5, max(tcd_results["Mean"]) + 5)
plt.ylim(min(tcd_results["Diff"]) - 5, max(tcd_results["Diff"]) + 5)
plt.axhline(md, color='red', linestyle='--', label='mean')
plt.axhline(md + 1.96 * sd, color='gray', linestyle='--', label='SD +1')
plt.axhline(md - 1.96 * sd, color='gray', linestyle='--', label='SD -1')
plt.xlabel('Average Between Observer and Model')
plt.ylabel('Difference Between TCD Truth and TCD Predictions')
plt.legend(loc='center left', title='Legend')
plt.title('Bland-Altman Plot of TCD Agreement | Trained on {}'.format(model_selection))
plt.show()
# %%
n = 12
for g, df in tcd_results.sort_values(by='TCD_Preds', axis=0).groupby(
        np.arange(len(tcd_results.sort_values(by='TCD_Preds', axis=0))) // n):
    df.to_csv(TCD_csv_file[:-4] + 'q' + str(g + 1) + '.csv')
