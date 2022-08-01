import os


def build_training_ds(directory=os.getcwd()):
    patches = [patch + '\t' + str(patch[-5]) + '\n' for patch in os.listdir(directory) if patch.endswith('.png')]
    with open(f'{directory}\\TrainingData.txt', 'w') as file:
        file.writelines(patches)
        file.flush()
