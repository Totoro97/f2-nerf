# To install hloc, see: https://github.com/cvg/Hierarchical-Localization
from hloc import extract_features, match_features, reconstruction, visualization
from hloc import pairs_from_retrieval, pairs_from_exhaustive
from pathlib import Path
import click


@click.command()
@click.option('--data_dir', type=str)
@click.option('--match_type', type=str, default='exhaustive')
def hello(data_dir, match_type):
    images = Path(data_dir) / 'images/'
    outputs = Path(data_dir)

    sfm_pairs = outputs / 'pairs-sfm.txt'
    loc_pairs = outputs / 'pairs-loc.txt'
    sfm_dir = outputs / 'hloc_sfm'
    features = outputs / 'features.h5'
    matches = outputs / 'matches.h5'

    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    assert match_type in ['exhaustive', 'local']
    if match_type == 'exhaustive':
        references = [p.relative_to(images).as_posix() for p in images.iterdir()]
        print(len(references), "mapping images")
        extract_features.main(feature_conf, images, image_list=references, feature_path=features)
        pairs_from_exhaustive.main(sfm_pairs, image_list=references)
        match_features.main(matcher_conf, sfm_pairs, features=features, matches=matches)
        reconstruction.main(sfm_dir, images, sfm_pairs, features, matches, image_list=references, image_options={'camera_model': 'OPENCV'})
    else:
        retrieval_conf = extract_features.confs['netvlad']
        retrieval_path = extract_features.main(retrieval_conf, images, outputs)
        pairs_from_retrieval.main(retrieval_path, sfm_pairs, num_matched=20)
        feature_path = extract_features.main(feature_conf, images, outputs)
        match_path = match_features.main(matcher_conf, sfm_pairs, feature_conf['output'], outputs)

        reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path, image_options={'camera_model': 'OPENCV'})



if __name__ == '__main__':
    hello()
