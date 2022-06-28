optimizer = dict(type='Adam', lr=4e-4, weight_decay=1e-6)

class_metrics = dict(
    accuracy=dict(is_multilabel=True), roc_auc=dict(check_compute_fn=False), ap=dict(check_compute_fn=False))

cosine_annealing = True
val_interval = 1
max_epochs = 50
n_saved = 2
log_interval = 20
