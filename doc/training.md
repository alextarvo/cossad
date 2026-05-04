As of 01/2026, the contrastive encoder is under active experimentation. Significant portions of its code
and configuration parameters may be experimental and not recommended for production use.

Make sure that your [environment is set](environment.md) up correctly.

# Training data
The trainer reads tuples of patches from the path specified by `data.patches_path`. Throughout the configs, we expect
data to be located in the `./data` folder, which can be a symbolic link. The `./data` folder must contain folders with 
pre-processed input patches for the `real3dad` and `shapenet` datasets.

Currently, training data is hosted on Cloudflare R2.
Download `<training_set_name>_real3dad_norot.tar` and `<training_set_name>_shapenet_norot.tar` to your computer,
as described in the [dataset](dataset.md#ready-to-use-dataset) document,
and extract their contents to the `./data` folder. You should have
`<training_set_name>_real3dad_norot` and `<training_set_name>_shapenet_norot` folders.
Specify the path to the training data in the configuration file (see below) or from the command line as: 

```
data.patches_path=./data/<training_set_name>
```

# Trainer configuration

We expect that you have a Weights and Biases (wandb) account configured for training COSSAD. Set the 
WANDB_API_KEY environment variable. We are working on making wandb optional for training.

The trainer is configured through Hydra. Hydra configs are located in the `./configs` folder. The current configuration file
is `config_riconv_msloss_v3.yaml`. It has extensive comments that describe the semantics of the configuration parameters. 
The following parameters should be overridden from the command line during training:

- `data.dataset`: A list of datasets used to train the model. Specify one dataset (`real3dad` or `shapenet`)
or both. For both, use `data.dataset=[shapenet,real3dad]`; the model will be trained on combined data from both datasets. 
- `experiment.tags`: A comma-separated list of self-describing, human-readable strings for this training session. These will be associated   
with this run in wandb.
- `experiment.run_id`: An identifier for this training run. When set, weights are saved to
`{model_output_path}/{run_id}/{split}/`. When `null` (default), a per-invocation timestamp is used instead.
The pipeline runner and `scripts/training_v3.bash` both use the format `Train_YYYYMMDD_HHMMSS`.
- `paths.model_output_path`: A base path where model weights will be saved. The run ID and split name are appended
as subdirectories. Best weights are named `best_<model.name>.pth`.
- `paths.logs_output_path`: Similarly, a base path where TensorBoard logs will be saved.
- `data.split`: A comma-separated list of data splits used to train models. Hydra will start a separate training session 
for each split and train a separate model on that data split. These data splits are used for 12-fold cross-validation
of COSSAD.
The names of the objects in a split are stored 
in the `class_splits_YYYY_MM_DD.py` file; the current version is `class_splits_2025_11_13.py`. `train_classes_all` is a special 
split name that includes all objects in a dataset. 

Here is an example command to train the feature extractor on data split 1:
```bash
python contrastive_learner_v3.py --config-path configs \
        --config-name config_riconv_msloss_v3 -m \
    experiment.run_name=riconv_split_1 \
    experiment.run_id='Train_20260320_150000' \
    data.dataset=[shapenet,real3dad] \
    data.patches_path=./data/supcon_train_r2_2025_11_04 \
    data.split=train_classes_1 \
    model.name=riconv2 \
    experiment.tags=[v3,1_tensor_per_batch,simple_miner]
```

Model weights are saved following the layout `{model_output_path}/{run_id}/{split}/`. For the example above, weights will be at:

`/data/model_weights/Train_20260320_150000/train_classes_1/best_riconv2.pth`

This is the same layout used by the pipeline runner, so weights trained manually with an explicit `run_id` can be evaluated by pointing `--weights-run-id` at that ID.


# Training on multiple data splits
We use 12-fold cross-validation to evaluate COSSAD (see [evaluation](evaluation.md#cross-validation)). 
Thus, for a proper evaluation, we need to train a separate
model on each split.
`./scripts/training_v3.bash` is a convenience script that trains the v3 version of the model across all 12 splits.
It accepts an optional `RUN_ID` argument (defaults to `Train_<timestamp>`):

```bash
scripts/training_v3.bash                           # auto-generated ID
scripts/training_v3.bash Train_20260320_150000     # explicit ID
```

Weights are saved to `/data/model_weights/<RUN_ID>/train_classes_N/best_riconv2.pth` — the same layout as the pipeline runner. To evaluate afterwards, use `python scripts/run_train_eval_pipeline.py eval --weights-run-id <RUN_ID>` or the standalone evaluation scripts.


# Training via the pipeline runner

The [pipeline runner](training_validation_pipeline.md) (`scripts/run_train_eval_pipeline.py`) can train, evaluate, and compare models in a single command. For training specifically, it:

- Iterates over the requested cross-validation splits, invoking `contrastive_learner_v3.py` for each one
- Generates a `WEIGHTS_RUN_ID` (e.g. `Train_20260320_151240`) that determines the output directory for all splits
- Skips splits whose `best_riconv2.pth` already exists, enabling partial re-runs after interruptions
- Logs the full Hydra command line for each split so the run is reproducible by copy-pasting

To train all 12 splits:

```bash
python scripts/run_train_eval_pipeline.py full          # train + eval + compare
python scripts/run_train_eval_pipeline.py train          # train only, all 12 splits
```

To train a subset of splits (useful during development):

```bash
python scripts/run_train_eval_pipeline.py train --splits 2,3 --epochs 10
```

To preview the exact commands without running them:

```bash
python scripts/run_train_eval_pipeline.py train --splits 2,3 --dry-run
```

All training parameters (config name, model name, experiment tags, patches subdirectory) are defined as constants at the top of the script. See [pipeline runner documentation](training_validation_pipeline.md) for the full reference.

