# ----------------------
# EXPERIMENT SAVE PATHS
# ----------------------
data_root = '/home/aalekh/csayn-data-root'
exp_root = f'{data_root}/open_set_recognition'        # directory to store experiment output (checkpoints, logs, etc)
save_dir = f'{data_root}/open_set_recognition/methods/baseline/ensemble_entropy_test'    # Evaluation save dir

# evaluation model path (for openset_test.py and openset_test_fine_grained.py, {} reserved for different options)
root_model_path = data_root + '/open_set_recognition/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}.pth'
root_criterion_path = data_root + '/open_set_recognition/methods/ARPL/log/{}/arpl_models/{}/checkpoints/{}_{}_{}_criterion.pth'

# -----------------------
# DATASET ROOT DIRS
# -----------------------
cifar_10_root = f'{data_root}/datasets/cifar10'                                          # CIFAR10
cifar_100_root = f'{data_root}/datasets/cifar100'                                        # CIFAR100
cub_root = f'{data_root}/datasets/CUB'                                                   # CUB
aircraft_root = f'{data_root}/datasets/aircraft/fgvc-aircraft-2013b'                      # FGVC-Aircraft
mnist_root = f'{data_root}/datasets/mnist/'                                              # MNIST
pku_air_root = f'{data_root}/datasets/pku-air-300/AIR'                                   # PKU-AIRCRAFT-300
car_root = data_root + "/datasets/stanford_car/cars_{}/"                                 # Stanford Cars
meta_default_path = data_root + "/datasets/stanford_car/devkit/cars_{}.mat"              # Stanford Cars Devkit
svhn_root = data_root + '/datasets/svhn'                                                 # SVHN
tin_train_root_dir = data_root + '/datasets/tinyimagenet/tiny-imagenet-200/train'        # TinyImageNet Train
tin_val_root_dir = data_root + '/datasets/tinyimagenet/tiny-imagenet-200/val/images'     # TinyImageNet Val

# ----------------------
# FGVC OSR SPLITS
# ----------------------
osr_split_dir = data_root + '/open_world_learning/open_set_recognition/data/open_set_splits'

# ----------------------
# PRETRAINED RESNET50 MODEL PATHS (For FGVC experiments)
# Weights can be downloaded from https://github.com/nanxuanzhao/Good_transfer
# ----------------------
imagenet_moco_path = f'{data_root}/pretrained_models/imagenet/moco_v2_800ep_pretrain.pth.tar'
places_moco_path = f'{data_root}/pretrained_models/places/moco_v2_places.pth'
places_supervised_path = f'{data_root}/pretrained_models/places/supervised_places.pth'
imagenet_supervised_path = f'{data_root}/pretrained_models/imagenet/supervised_imagenet.pth'
