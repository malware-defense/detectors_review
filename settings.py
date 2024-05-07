import os

_project_path = '/tmp/pycharm_project_843/'


def _project(base):
    return os.path.join(_project_path, base)


config = {
    # Experiment settings
    'saved_models': _project('model_results/apg/models'),
    'saved_features': _project('model_results/apg/features'),
    # 'labels': _project('train_data/apg/apg-y.json'),
    # 'meta_data': _project('train_data/apg/apg-meta-new-local.json'),
    'meta_data': _project('train_data/Data-MD/train.txt'),
    'labels': _project('train_data/Data-MD/test.txt'),
    'android_sdk': '/home/android-sdk/',
    'tmp_dir': '/root/autodl-tmp/gnip/tmp/',
    'results_dir': '/home/nfs/gnip/ADZ/results/',
    # 'source_apk_path': '/home/nfs/gnip/apg/apg_apps/',
    'source_apk_path': '/root/autodl-fs/apg/apg_apps/',
    'slice_database': '/home/nfs/gnip/ADZ/slices_database/',
    "resigner": _project("java-components/apk-signer.jar"),
    'data_source': '/root/autodl-fs/sample/',
    'base_clf_dir': '/root/autodl-fs/models/',
    'model_save_dir': _project('checkpoints/'),

    # sharpness
    'sharpness_result_dir': _project('results/sharpness/'),
    'fs_result_dir': _project('results/fs/'),
    'lid_result_dir': _project('results/lid/'),
    'magnet_result_dir': _project('results/magnet/'),

    # Misc
    'nproc_feature': 10,
    'nproc_slicer': 10,
    'nproc_attacker': 10,
    'sign': False,
    'extract_feature': False,  # Extract the feature
    'serial': False,  # Attack in serial

    'extract_attack_feature': False,



    # Detect_sharpness
    'adv_init_mag': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],

}
