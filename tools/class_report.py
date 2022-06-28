from typing import Optional

import numpy as np
import pandas as pd
from argparse import ArgumentParser, Namespace
from ignite.utils import setup_logger
from sklearn.metrics import average_precision_score, roc_auc_score
from tabulate import tabulate


def parse_args() -> Namespace:
    parser = ArgumentParser('Show classification report')
    parser.add_argument('pred_file', help='Prediction csv file.')
    parser.add_argument('--out-file', help='File path for saving the classification report.')

    args = parser.parse_args()
    return args


def class_report(pred_file: str, out_file: Optional[str] = None) -> None:
    logger = setup_logger('imba-explain')
    if not pred_file.endswith('.csv'):
        raise ValueError(f'pred_file must be a csv file, but got {pred_file}.')
    if out_file is not None:
        if not out_file.endswith('.csv'):
            raise ValueError(f'out_file must be a csv file, but got {out_file}.')

    df = pd.read_csv(pred_file)
    pred_cols = [x for x in df.columns if x.startswith('p-')]
    gt_cols = [x for x in df.columns if x.startswith('t-')]
    class_names = [x.split('-')[1] for x in gt_cols]

    pred = df[pred_cols].values
    target = df[gt_cols].values

    avg_pred = np.mean(pred, axis=0)
    roc_auc = roc_auc_score(target, pred, average=None)
    ap = average_precision_score(target, pred, average=None)
    num_pos = np.sum(target, axis=0)

    tabular_data = {
        'Classes': class_names,
        'Samples': num_pos.astype(int),
        'Avg Prob': avg_pred.round(4),
        'ROC AUC': roc_auc.round(4),
        'AP': ap.round(4)
    }
    table = tabulate(tabular_data, headers='keys', tablefmt='pretty', numalign='left', stralign='left')

    log_str = 'Classification Report:\n'
    log_str += f'{table}'
    logger.info(log_str)

    if out_file is not None:
        out_df = pd.DataFrame(tabular_data)
        out_df.to_csv(out_file)
        logger.info(f'Classification report has been saved to {out_file}.')


def main():
    args = parse_args()
    class_report(pred_file=args.pred_file, out_file=args.out_file)


if __name__ == '__main__':
    main()
