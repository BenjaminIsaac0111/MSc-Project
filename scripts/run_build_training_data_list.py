from utils import utils

if __name__ == '__main__':
    utils.write_list([file + '\t' + cla for file, cla in utils.get_classes_from_data_dir('../data/HGDL3')],
                     '../data/HGDL3/TrainingData.txt')
