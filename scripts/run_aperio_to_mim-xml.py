from utils import xmlformats
from Processing import SVSLoader

if __name__ == '__main__':
    loader = SVSLoader()
    for svs in loader.svs_files:
        print(xmlformats.aperio2mimxml(loader=loader,
                                       image_id=svs,
                                       layer_prefix='whole_score',
                                       subsamplefactor=2,
                                       separatelayers=True,
                                       flip_xy=False,
                                       desc_as_name=None))
