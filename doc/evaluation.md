# Evaluating COSSAD 

Inference and evaluation are performed through the `pipeline_reg_based.py` script. It will load PCs for given object(s) from the 
COSSAD dataset, run the spatially-aware anomaly detection algorithm on them, and validate the results against the
anomaly masks defined for these objects. `pipeline_reg_based.py` computes the object-level and point-level accuracy metrics
as areas under ROC curve (AUROC), and saves these into .csv file. For each object, it also outputs the predicted anomaly 
scores for each point within a point cloud for that object. These predictions can be later visualized. 

Currently, a single script performs both inference and evaluation.

The inference/evaluation script runs a previously trained contrastive encoder model,
and thus requires a GPU to run. Make sure that your [environment is set](environment.md) up correctly.

## Command line arguments for the evaluation

The inference script requires the `--cossad_data_path` argument to specify the path to the 
COSSAD dataset, as described in the [dataset](dataset.md). At a minimum,
the `template` and `train` folders with the corresponding set of point clouds in COSSAD format must be present. We typically assume
this dataset is located in the `./data` folder (it can be a symlink to a different location).

`--object_class` specifies the name of a class (or comma-separated class names) for which inference and
evaluation will be performed. The script loads all point clouds for this class from the `train` folder of the COSSAD dataset, runs inference
on them, and evaluates the accuracy. Please read the "Cross-validation" section below carefully. We do not automatically 
enforce train/test splits as it hinders experimentation. You must manually pass the weights and object class that
represent a valid data split.


The encoder architecture is specified through the `--encoder` parameter. Currently, we support the `riconv2` encoder
based on the RIConv++ network. 
`--contrastive_encoder_path` specifies the path to the trained model weights. For convenience, 
pre-trained weights are provided in the `./weights/2025_12_11` folder.

The values of `--patch_radius`, `--points_per_patch`, and `--embedding_dim` must exactly match the values used during
contrastive encoder training.

In the Real3D-AD dataset, training objects contain only half of a point cloud—either a top-down or bottom-up view.
When running inference on such objects, use the `--filter_by_point_count` flag. We use simple heuristics to
select only patches that lie fully within that partial point cloud. **Do not** use the `--filter_by_point_count` flag
on full 3D shapes, such as those in the Anomaly-ShapeNet dataset.

The example below shows how to run inference on the `airplane` object from the Real3D-AD dataset using our pre-trained weights:

```bash
python pipeline_reg_based.py \
        --cossad_data_path=./data \
        --object_class="airplane" \
        --encoder=riconv2 \
        --contrastive_encoder_path="./weights/2025_12_11/riconv2_split3.pth" \
        --patch_radius=2 \
        --filter_by_point_count \
        --output_results_path="./data/predictions/real3dad/test"
```

The `--output_results_path` specifies where results will be saved. Results include
a CSV file with accuracy scores and NPZ files with predicted anomaly scores for each point 
in the point cloud. 

## Cross-validation {#cross-validation}

We use 12-fold cross-validation to train and validate COSSAD. Data splits are generated based on
object name. We train the encoder on a subset that includes data from 11/12 of all objects and validate it on the
remaining 1/12 of objects. This evaluates how COSSAD handles previously unseen geometries. Training and validating
COSSAD on point clouds from the same object would cause data leakage of training data into the evaluation set. 

The current structure of the training/evaluation splits is defined in the `class_splits_2025_11_13.py`
file. To generate different data splits, use the `scripts/generate_datasplits.py` script.

If you train the network yourself and use newly trained weights for evaluation, ensure 
proper data splitting. In the example below, we train the COSSAD encoder on data split #2 (which does not
include the Real3D-AD "shell" object) using joint data from both Real3D-AD and ShapeNet datasets:

```bash
python contrastive_learner_v3.py --config-path configs \
--config-name config_riconv_msloss_v3 -m experiment.run_name=riconv_split2 \
data.dataset=[shapenet,real3dad] \
data.patches_path=./data/supcon_train_r2_2025_11_04 \
paths.model_output_path=./data/model_weights/shapenet_real3dad/riconv_train_classes_2 \
paths.logs_output_path=./data/model_logs/shapenet_real3dad/riconv_train_classes_2 \
data.split=train_classes_2 model.name=riconv2
```

Then, to validate the COSSAD on this split, run:

```bash
FULL_MODEL_PATH=$(ls ./data/model_weights/shapenet_real3dad/riconv_train_classes_2/*/best_riconv2.pth)

python pipeline_reg_based.py --cossad_data_path=./data \
--object_class="shell" --encoder=riconv2 \
--contrastive_encoder_path="$FULL_MODEL_PATH" \
--patch_radius=2 --filter_by_point_count \
--output_results_path=./data/predictions/real3dad/split2
```

Note that the full path to the model contains a timestamped folder within it. 

Currently, we do not enforce consistency of data splits between training and inference, as it complicates
rapid experimentation. We maintain scripts that run COSSAD on all data splits using
proper pre-trained weights for Real3D-AD and Anomaly-ShapeNet: `./scripts/evaluation_pretrained_v3_real3dad.sh` and `./scripts/evaluation_pretrained_v3_shapenet.sh`.

The objects that belong to a data split are specified in the `object_groups` variable; the corresponding weights 
are in `encoder_paths`.
`object_groups` and `encoder_paths` are arrays; each i-th entry represents objects that belong to the i-th
data split and weights trained on the i-th data split. If you train the encoder yourself, edit 
`encoder_paths` to set the path to your trained weights.

For quick validation, we provide convenience scripts.
Run `./scripts/training_v3.bash` to train the model on all splits and save weights into the proper folder structure.
Then run either `./scripts/evaluation_v3_real3dad.sh` or `./scripts/evaluation_v3_shapenet.sh`
to load the newly trained weights and perform evaluation.

## Evaluation via the pipeline runner

The [pipeline runner](training_validation_pipeline.md) (`scripts/run_train_eval_pipeline.py`) automates evaluation across all cross-validation splits. For evaluation specifically, it:

- Iterates over the requested splits, invoking `pipeline_reg_based.py` for each one with the correct weights path, dataset-specific flags (e.g. `--filter_by_point_count` for Real3D-AD), and output directory
- Verifies that trained weights exist for each split before running — fails with a clear error rather than silently skipping
- Generates a fresh `EVAL_RUN_ID` (e.g. `Eval_20260320_151240`) so each evaluation run gets its own output directory
- After all splits complete, automatically runs `scripts/compare_models.py` to perform a Mann-Whitney U statistical comparison against the baseline
- Logs the full command line for each split invocation for reproducibility

To evaluate with existing weights:

```bash
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260320_151240
```

To evaluate a subset of splits or with fewer attempts (faster):

```bash
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260320_151240 \
    --splits 2,3 --attempts 1
```

To evaluate on a different dataset:

```bash
python scripts/run_train_eval_pipeline.py eval \
    --weights-run-id Train_20260320_151240 \
    --eval-dataset shapenet
```

The runner applies dataset-specific flags automatically — `--filter_by_point_count` is added for `real3dad` and omitted for `shapenet`. These mappings are defined in the `DATASET_EVAL_FLAGS` constant at the top of the script. See [pipeline runner documentation](training_validation_pipeline.md) for the full reference.