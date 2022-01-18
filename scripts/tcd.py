import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score
import seaborn as sns

# %%
MODEL_PATH = r'E:\JNET_Results_1\Collated_16px\\'
results_files = [MODEL_PATH + file for file in os.listdir(MODEL_PATH) if file.endswith('.csv')]
tcd_results = {}
whole_results_df = pd.DataFrame(columns=['Truth', 'Preds', 'Patch', 'Institute_id', 'Patient_id', 'Image_id',
                                         'Patch_no', 'TCD_T', 'TCD_P'])
# %%
for results_file in results_files:
    institute_id = os.path.split(results_file)[-1].split('_')[3]
    results_df = pd.read_csv(filepath_or_buffer=results_file, encoding='utf-8', index_col=[0])
    results_df[['Institute_id',
                'Patient_id',
                'Image_id',
                'Patch_no']] = results_df['Patch'].str.split('_', expand=True).drop(columns=[4, 5])

    results_df['TCD_T'] = results_df['Truth']
    results_df.loc[results_df['TCD_T'].isin([3, 4, 5, 6, 7, 8]), 'TCD_T'] = 2

    results_df['TCD_P'] = results_df['Preds']
    results_df.loc[results_df['TCD_P'].isin([3, 4, 5, 6, 7, 8]), 'TCD_P'] = 2

    whole_results_df = whole_results_df.append(results_df)

    for image_id in set(results_df['Image_id']):
        t_counts = results_df[results_df['Image_id'] == image_id]['TCD_T'].value_counts()
        p_counts = results_df[results_df['Image_id'] == image_id]['TCD_P'].value_counts()
        kappa_score = cohen_kappa_score(results_df[results_df['Image_id'] == image_id]['TCD_T'],
                                        results_df[results_df['Image_id'] == image_id]['TCD_P'])
        balanced_acc_score = balanced_accuracy_score(results_df[results_df['Image_id'] == image_id]['TCD_T'],
                                                     results_df[results_df['Image_id'] == image_id]['TCD_P'])

        tcd_results[image_id] = {'Institute': institute_id,
                                 'Kappa_Score': kappa_score,
                                 'Balanced_Accuracy': balanced_acc_score,
                                 'TCD_Truth': ((t_counts[1] / (t_counts[1] + t_counts[0])) * 100),
                                 'TCD_Preds': ((p_counts[1] / (p_counts[1] + p_counts[0])) * 100),
                                 }
tcd_results = pd.DataFrame.from_dict(tcd_results).T

tcd_results['Mean'] = np.mean([tcd_results['TCD_Truth'], tcd_results['TCD_Preds']], axis=0)
tcd_results['Diff'] = tcd_results['TCD_Truth'] - tcd_results['TCD_Preds']

# %%
md = np.mean(tcd_results['Diff'])  # Mean of the difference
sd = np.std(tcd_results['Diff'], axis=0)  # Standard deviation of the difference
sns.scatterplot(data=tcd_results, x='Mean', y='Diff', hue='Institute')
plt.xlim(min(tcd_results["Mean"]) - 5, max(tcd_results["Mean"]) + 5)
plt.ylim(min(tcd_results["Diff"]) - 5, max(tcd_results["Diff"]) + 5)
plt.axhline(md, color='red', linestyle='--', label='mean')
plt.axhline(md + 1.96 * sd, color='gray', linestyle='--', label='SD +1')
plt.axhline(md - 1.96 * sd, color='gray', linestyle='--', label='SD -1')
plt.xlabel('Average Between Observer and Model')
plt.ylabel('Difference Between TCD Truth and TCD Predictions')
plt.legend(loc='center left', title='Legend')
plt.title('Bland-Altman Plot of TCD Agreement | Trained on 16px')
plt.show()
# %%
balanced_accuracy_score(whole_results_df['Truth'].astype(int), whole_results_df['Preds'].astype(int))
balanced_accuracy_score(whole_results_df['TCD_T'].astype(int), whole_results_df['TCD_P'].astype(int))
cohen_kappa_score(whole_results_df['Truth'].astype(int), whole_results_df['Preds'].astype(int))
cohen_kappa_score(whole_results_df['TCD_T'].astype(int), whole_results_df['TCD_P'].astype(int))
# %%
report_results = {}
for model in ['16', '24', '32', '48', '64']:
    reports = [report for report in os.listdir(r'E:\JNET_Results_1\Collated_{}px'.format(model)) if
               report.endswith('Report.txt')]
    kappa = []
    b_acc = []
    for report in reports:
        with open(r'E:\JNET_Results_1\Collated_{}px\{}'.format(model, report)) as report_file:
            txt = report_file.readlines()
            kappa.append(float(txt[0].split(':')[1].strip()))
            b_acc.append(float(txt[2].split(':')[1].strip()))

    report_results[model] = {'Balanced Accuracy': np.mean(b_acc), 'Kappa Score': np.mean(kappa)}
# %%
pd.DataFrame(report_results).T.to_csv(r'E:\Models_Report.csv')
