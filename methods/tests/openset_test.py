from test.utils import EvaluateOpenSet, ModelTemplate
from utils.utils import strip_state_dict

import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from data.open_set_datasets import get_datasets, get_class_splits
from models.model_utils import get_model
import sys, os

from utils.utils import str2bool

from config import save_dir, root_model_path, root_criterion_path

class EnsembleModelEntropy(ModelTemplate):

    def __init__(self, all_models, mode='entropy', num_classes=4, use_softmax=False):

        super(ModelTemplate, self).__init__()

        self.all_models = all_models
        self.max_ent = torch.log(torch.Tensor([num_classes])).item()
        self.mode = mode
        self.use_softmax = use_softmax

    def entropy(self, preds):

        logp = torch.log(preds + 1e-5)
        entropy = torch.sum(-preds * logp, dim=-1)

        return entropy

    def forward(self, imgs):

        all_closed_set_preds = []

        for m in self.all_models:

            closed_set_preds = m(imgs, return_feature=False)

            if self.use_softmax:
                closed_set_preds = torch.nn.Softmax(dim=-1)(closed_set_preds)

            all_closed_set_preds.append(closed_set_preds)

        closed_set_preds = torch.stack(all_closed_set_preds).mean(dim=0)

        if self.mode == 'entropy':
            open_set_preds = self.entropy(closed_set_preds)
        elif self.mode == 'max_softmax':
            open_set_preds = -closed_set_preds.max(dim=-1)[0]

        else:
            raise NotImplementedError

        return closed_set_preds, open_set_preds

def load_models(path, args):

    model = get_model(args, evaluate=True)

    if args.loss == 'ARPLoss':

        state_dict_list = [torch.load(p) for p in path]
        model.load_state_dict(state_dict_list)

    else:

        state_dict = strip_state_dict(torch.load(path[0]))
        model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
    parser.add_argument('--device', default='None', type=str, help='Which GPU to use')
    parser.add_argument('--osr_mode', default='max_softmax', type=str, help='{entropy, max_softmax}')
    parser.add_argument('--seed', default=0, type=int)

    # Model
    parser.add_argument('--model', type=str, default='classifier32')
    parser.add_argument('--loss', type=str, default='ARPLoss')
    parser.add_argument('--feat_dim', default=128, type=int)
    parser.add_argument('--max_epoch', default=599, type=int)
    parser.add_argument('--cs', default=True, type=str2bool)

    # Data params
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', type=str, default='tinyimagenet')
    parser.add_argument('--transform', type=str, default='rand-augment')

    # Train params
    args = parser.parse_args()
    args.save_dir = save_dir
    device = torch.device('cuda:0')

    # exp_ids = [
    #     '(11.04.2021_|_16.699)',
    #     '(11.04.2021_|_27.623)',
    #     '(11.04.2021_|_25.507)',
    #     '(11.04.2021_|_17.466)',
    #     '(11.04.2021_|_07.958)'
    # ]

    exp_ids = [
        '(19.08.2021_|_26.227)',
        '(20.08.2021_|_34.790)',
        '(20.08.2021_|_45.082)',
        '(20.08.2021_|_51.491)',
        '(21.08.2021_|_42.871)'
    ]

    if args.cs:
        dataset_cs = args.dataset + 'cs'
    else:
        dataset_cs = args.dataset

    # Define paths
    all_paths_combined = [[x.format(i, args.dataset, dataset_cs, args.max_epoch, args.loss)
                           for x in (root_model_path, root_criterion_path)] for i in exp_ids]

    all_preds = []

    for split_idx in range(5):

        # ------------------------
        # DATASETS
        # ------------------------

        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, split_idx=split_idx)

        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                image_size=args.image_size, balance_open_set_eval=False,
                                split_train_val=False, open_set_classes=args.open_set_classes)

        # ------------------------
        # DATALOADERS
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers)

        # ------------------------
        # MODEL
        # ------------------------
        print('Loading arpl_models...')

        all_models = [load_models(path=all_paths_combined[split_idx], args=args)]

        model = EnsembleModelEntropy(all_models=all_models, mode=args.osr_mode, num_classes=len(args.train_classes))
        model.eval()
        model = model.to(device)

        # ------------------------
        # EVALUATE
        # ------------------------
        evaluate = EvaluateOpenSet(model=model, known_data_loader=dataloaders['test_known'],
                                   unknown_data_loader=dataloaders['test_unknown'], device=device, save_dir=args.save_dir)

        # Make predictions on test sets
        evaluate.predict()
        preds = evaluate.evaluate(evaluate)
        all_preds.append(preds)

    all_preds = np.array(all_preds)
    means = np.mean(all_preds, axis=0)
    stds = np.std(all_preds, axis=0)
    print(f'Mean: {means[0]:.4f} pm {stds[0]:.4f} | AUROC: {means[-2]:.4f} pm {stds[-2]:.4f} | AUPR: {means[-1]:.4f} pm {stds[-1]:.4f}')
