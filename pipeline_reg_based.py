import copy
import logging
import os
from typing import Any
import math

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

import visualization.utils_visualization as util_vis
import registration.icp_registration as icp_reg
from registration.random_perturbation import  RandomPerturbation
import registration.multiple_attempts_registration as mul_reg
from feature_extractors.feature_retrievers import ContrastiveFeatureRetriever, FPFHPerPatchFeatureRetriever

from utils.pipeline_args import set_pipeline_args
from utils.debug import in_debugger, assert_nans_nparray, set_random_seeds
from utils.logging_util import setup_logging, log_nans_nparray_error
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

import constants as consts
import dataloaders.cossad_dataloaders as cossad_dloaders
import dataloaders.real3d_dataloaders_legacy as real3d_legacy_dloaders
import dataloaders.util_dataloaders as util_dloaders
import patch_generators as pgen
import pmb as pmb

from visualization import utils_visualization as uvis
import time

# Enable for reproducible debugging
# set_random_seeds(42)


# The desired number of points in the downsampled PC that will be used for computing
# per-point accuracy. Computing accuracy using all the points is prohibitively expensive.
DSAMPLED_NPOINTS_FOR_AUC = 10000

# The desired number of voxels for PC registrtaion
REGISTRATION_VOXEL_COUNT = 5000

def downsample(points: np.ndarray, labels: np.ndarray, n_samples: int = 5000):
    """Downsamples the points and labels arrays to a desired number of samples"""
    assert points.shape[0] == labels.shape[0], "points and labels must have the same length"
    N = points.shape[0]
    # if we already have <= n_samples, do nothing
    if N <= n_samples:
        return points, labels
    idx = np.random.choice(N, size=n_samples, replace=False)
    return points[idx], labels[idx]


def update_object_stats(stats_path, stats_data, class_name, aggregated_for_class=False):
    """
    Save object statistics to a CSV file.

    Args:
        stats_path (str): Directory path where stats CSV will be saved.
        stats_data (dict): Dictionary of statistics {key: value}.
        class_name (str): name of the class of an object. if we are saving unaggregated stats
            for individual objects of that class, it is used to form CSV filename for that entry.
            If we are saving aggregated per-class stats - it is an an entry name in a global results table.
        aggregated_for_class: aggregate stats across all objects in a class
    """
    # Ensure directory exists
    os.makedirs(stats_path, exist_ok=True)

    # Define file path
    file_path = os.path.join(stats_path, f"{class_name}.csv")

    # Log stats
    dont_print_fields = []
    if aggregated_for_class:
        # This is an aggregated statistics for all objects
        logging.info(f"Statistics on the object {class_name}:")
    else:
        logging.info(f"Statistics on file {stats_data['input_file_test']}:")
        dont_print_fields = ['input_file_test']

    for k, v in stats_data.items():
        if k not in dont_print_fields:
            logging.info(f"  {k}: {v}")

    # Convert dict to DataFrame (one row)
    row_df = pd.DataFrame([stats_data])

    # If file exists, append; otherwise, create with header
    if os.path.exists(file_path):
        row_df.to_csv(file_path, mode="a", index=False, header=False)
    else:
        row_df.to_csv(file_path, mode="w", index=False, header=True)

    # Ensure flush to disk
    with open(file_path, "a") as f:
        f.flush()
        os.fsync(f.fileno())

def register(np_template, np_target, class_name):
    """Registers the template PC against the target PC.

    Args:
        np_template: a point cloud as a numpy array to be registered into the  frame of the target PC.
        np_target: a target PC that represent the reference frame
        class_name: the name of the class of an object. Used to get the minimum allowed accuracy for the registration

    Returns:
        a tuple that contains (registration summary statistics, a deep copy of the registered PC)
    """

    reg_threshold = consts.registration_mean_distances_thresholds[class_name]
    reg_factory = icp_reg.ICPRegistrationFactory(
        np_pc_template=np_template, np_pc_target=np_target, desired_num_voxels=REGISTRATION_VOXEL_COUNT)
    reg = mul_reg.MultipleAttemptRegistrationEugenRotation(reg_factory, reg_threshold)
    reg_success = reg.compute_registration_transform()
    if not reg_success:
        logging.warning(f'Likely wrong registration, skipping')
        return None, None
    assert reg.get_accuracy().mean_dist_src_to_target <= reg_threshold
    logging.info(f'Registered target PC {np_target.shape} against template {np_template.shape}.')
    logging.info(f'Best accuracy: {reg.get_accuracy()}; resulting PC {reg.get_registered_target_pc().shape}')
    return reg.get_accuracy(), copy.deepcopy(reg.get_registered_target_pc())


def anomaly_score_for_points(np_pc, patch_center_indices_test, effect_sizes, patch_radius):
    """
    Predict anomaly score for each point, as an aggregation of the anomaly scores for each patch it belongs to.

    Args:
        np_pc: Point cloud as (N,3) array
        patch_center_indices_test: Patch center indices as (M) array
        effect_sizes: Effect sizes for each patch as  (M) array
        patch_radius: the radius of a patch
    Returns:
         anomaly_scores_max: maximum anomaly score for all patches the point belongs to
         anomaly_scores_avg: an average of all anomaly scores for all patches the point belongs to
         anomaly_scores_gauss: a Gaussian kernel is applied to distances to all the patches, and then averaged.
    """
    # Use library code to get pariwise distance between patch centers and rest of the points
    patch_centers = np_pc[patch_center_indices_test]
    dist2 = cdist(np_pc, patch_centers, metric='sqeuclidean')

    # The most accurate scoring use a Gaussian kernel
    norm_constant = 1 / (math.sqrt(2 * math.pi) * patch_radius)
    sigma = patch_radius  # use sigma (i.e. stddev) = patch radius
    gauss_kernel_values = np.exp(-dist2 / (2 * (sigma ** 2)))
    ascores_gauss_num = np.sum(np.array(effect_sizes) * gauss_kernel_values, axis=1)
    ascores_gauss_denom = np.sum(gauss_kernel_values, axis=1)
    anomaly_scores_gauss = np.where(ascores_gauss_denom > 0, ascores_gauss_num / ascores_gauss_denom,
                                    0.0) * norm_constant
    # return anomaly_scores_max, anomaly_scores_avg, anomaly_scores_gauss
    return anomaly_scores_gauss


def anomaly_score_for_points_torch(np_pc, patch_center_indices_test, effect_sizes, patch_radius, device_idx):
    """
    Predict anomaly score for each point, as an aggregation of the anomaly scores for each patch it belongs to.
    Use Torch

    Args:
        np_pc: Point cloud as (N,3) array
        patch_center_indices_test: Patch center indices as (M) array
        effect_sizes: Effect sizes for each patch as  (M) array
        patch_radius: the radius of a patch
        device_idx: the index of CUDA device to perform computations
    Returns:
         anomaly_scores_max: maximum anomaly score for all patches the point belongs to
         anomaly_scores_avg: an average of all anomaly scores for all patches the point belongs to
         anomaly_scores_gauss: a Gaussian kernel is applied to distances to all the patches, and then averaged.
    """
    # Use library code to get pariwise distance between patch centers and rest of the points
    patch_centers = np_pc[patch_center_indices_test]
    pc_t = torch.from_numpy(np_pc).float().to(f'cuda:{device_idx}')
    centers_t = torch.from_numpy(patch_centers).float().to(f'cuda:{device_idx}')
    dist2 = torch.cdist(pc_t, centers_t).pow(2)

    # The most accurate scoring use a Gaussian kernel
    norm_constant = 1 / (math.sqrt(2 * math.pi) * patch_radius)
    sigma = patch_radius  # use sigma (i.e. stddev) = patch radius
    effect_sizes_t = torch.tensor(effect_sizes).float().to(f'cuda:{device_idx}')

    kernel_values = torch.exp(-dist2 / (2 * sigma**2))
    scores_num = (effect_sizes_t * kernel_values).sum(dim=1)
    scores_denom = kernel_values.sum(dim=1)
    result = torch.where(scores_denom > 0, scores_num / scores_denom, 0.0) * norm_constant
    return result.cpu().numpy()


def build_knn_patch_graph(np_pc, patch_center_indices_filtered, feature_vectors_filtered, k):
    """Builds a graph representation of a shape. Graph nodes are the patch centers; edges are formed by taking kNN nodes

    Args:
        np_pc: Point cloud as (N,3) array, a representation of the object's shape
        patch_center_indices_filtered: indices of the patch centers selected as vertices
        k: the number of nearest neighbors to be connected by edges
    """
    patch_centers_3d = np_pc[patch_center_indices_filtered]
    M = patch_centers_3d.shape[0]
    k_actual = min(k, M - 1)
    tree = cKDTree(patch_centers_3d)
    _, neighbor_indices = tree.query(patch_centers_3d, k=k_actual + 1)
    neighbor_indices = neighbor_indices[:, 1:]
    adjacency = np.zeros((M, M), dtype=np.uint8)
    for i in range(M):
        adjacency[i, neighbor_indices[i]] = 1
    node_features = feature_vectors_filtered.numpy() if hasattr(feature_vectors_filtered, 'numpy') else np.asarray(feature_vectors_filtered)
    return {
        'adjacency': adjacency,
        'node_features': node_features,
        'patch_centers_3d': patch_centers_3d,
        'point_coordinates': np_pc,
        'k': np.array(k_actual),
    }


# A type of an entry we expect from the dataloader
ItemDict = dict[str, np.ndarray | list[Any] | None]


def pipeline(args):
    logging.info('pipeline_reg_based:\n' + '\n'.join(f'  {k}: {v}' for k, v in vars(args).items()))
    object_classes = util_dloaders.get_object_classes_names(args.object_classes)

    # Setting num_workers to 0 in Torch DataLoader is necessary to allow debuggin in PyCharm
    num_workers = 0 if in_debugger() else 4

    for real_class in object_classes:
        for attempt in range(args.attempts):
            try:
                class_stats = {}
                logging.warning(f'Testing object class {real_class} (attempt {attempt + 1}/{args.attempts})')
                # This is  a registration template to register everything into the same coordinate frame.
                basic_template = None

                template_dataset = cossad_dloaders.DatasetCossad(
                    args.cossad_data_path, 'template', dataset_tpl='.*', class_tpl=real_class,
                    anomaly_tpl='good', index_tpl='.*', normalize=False)
                collate_fn = util_dloaders.dloader_dict_collate
                # template_dataset = real3d_legacy_dloaders.Dataset3DADLegacyTrain(
                #     args.real3dad_data_path, real_class, normalize=True)
                # collate_fn = real3d_legacy_dloaders.real3d_dict_collate
                template_loader = DataLoader(template_dataset, num_workers=num_workers,
                                             batch_size=1, shuffle=True, drop_last=True, collate_fn=collate_fn)
                pbar = tqdm(enumerate(template_loader), total=len(template_loader))

                # This is a set of template PCs registered against each other
                template_pcs = []

                # First, generate a memory bank of features from the "good" objects
                for idx_template, item in pbar:
                    if item['np_pointcloud'] is None:
                        continue
                    # Make sure we loaded only a simngle point cloud.
                    assert (len(item['np_pointcloud']) == 1)
                    np_pointcloud = item['np_pointcloud'][0]
                    # Temporary code for extensive debugging. Leave it here temporarily, until a better solution is found.
                    # np.save(f'np_pointcloud_{idx_template}.npy', np_pointcloud)
                    input_file = item['input_file'][0]
                    logging.warning(f'Processing object loaded from {input_file}')
                    if np_pointcloud is None:
                        logging.warning(f'For training item {idx_template}, received no data')
                        continue
                    if idx_template >= args.obj_per_class:
                        # Stop after processing a small number of objects per class. Used for debugging
                        break
                    if idx_template == 0:
                        # The very first item in the test set. Use as a registration template.
                        basic_template = copy.deepcopy(np_pointcloud)
                        template_pcs.append(basic_template)
                    else:
                        reg_stats, registered_pc = register(basic_template, np_pointcloud, real_class)
                        if registered_pc is None:
                            class_stats[f'reg_good_to_template_{idx_template}'] = ''
                            continue
                        class_stats[f'reg_good_to_template_{idx_template}'] = reg_stats.mean_dist_src_to_target
                        template_pcs.append(registered_pc)
                        if args.do_visualize:
                            uvis.show_registered_np_pointcloud(registered_pc, basic_template)

                    # Temporary code for extensive debugging. Leave it here temporarily, until a better solution is found.
                    # np.save(f'registered_pc_train_{idx_template}.npy', registered_pc_train)
                    # np.save(f'patch_center_indices_train_{idx_template}.npy', patch_center_indices)
                    # np.save(f'feature_vectors_train_{idx_template}.npy', feature_vectors)
                    # np.save(f'patch_center_indices_filtered_train_{idx_template}.npy', patch_center_indices_filtered)


                # Reset label arrays
                image_predictions = []
                image_labels = []
                pixel_predictions = []
                pixel_labels = []
                inference_times = []
                reg_accuracies = []

                time_start = time.time()
                if args.pmb_type == 'indexed':
                    memory_bank = pmb.IndexedPatchMemoryBank(top_k=args.top_k, aggregation_method=args.top_k_aggregation_method)
                elif args.pmb_type == 'patchcore':
                    memory_bank = pmb.PatchcoreMemoryBank()
                else:
                    raise ValueError(f'Unknown pmb_type: {args.pmb_type}')

                # This code is for ablation studies only. It compares a shape to a single template
                # memory_bank = pmb.IndexedPatchMemoryBankSingleTemplate()

                # Prepare feature set retrievers / encoders based on the first template
                # Patch generator samples the surface of the input shape and generates the coordinates of patch centers
                patch_generator = pgen.FPSFixedPatchGenerator(
                    num_points=args.num_patches, np_template_pc=template_pcs[0], radius=args.patch_radius)
                # Feature retriever takes patches and returns their features
                if args.encoder == 'fpfh':
                    # This is for ablation studies only; don't use it in prod
                    feature_retriever = FPFHPerPatchFeatureRetriever(patch_generator, patch_radius=args.patch_radius)
                else:
                    feature_retriever = ContrastiveFeatureRetriever(
                        patch_generator, args.encoder, args.contrastive_encoder_path,
                        embedding_dim=args.embedding_dim, points_per_patch=args.points_per_patch,
                        patch_radius=args.patch_radius, standardize_patch=True, device_idx=args.device
                    )

                # Extract features from all the template PCs and store them in a memory bank
                for template_idx, registered_pc_template in enumerate(template_pcs):
                    patch_center_indices, feature_vectors, patch_center_indices_filtered, feature_vectors_filtered, _ = feature_retriever.get_features(
                        registered_pc_template)
                    np_pc_allcenters = registered_pc_template[patch_center_indices_filtered]
                    if args.do_visualize:
                        uvis.show_registered_np_pointcloud(np_pc_allcenters, patch_generator.patch_centers)
                    memory_bank.add_patches(feature_vectors)
                    if args.extract_graph and args.output_results_path is not None:
                        # We requested to build a graph representation of the shape, where the edges are selected
                        # according tot he kNN principle
                        graph = build_knn_patch_graph(
                            registered_pc_template, patch_center_indices_filtered,
                            feature_vectors_filtered, args.graph_k)
                        graph_dir = os.path.join(args.output_results_path, 'graphs', real_class)
                        os.makedirs(graph_dir, exist_ok=True)
                        np.savez(os.path.join(graph_dir, f'template_{template_idx}.npz'), **graph)
                memory_bank.compute_pairwise_distances()
                # TODO(alexta): We include only these patches that are of a high quality. They have a full
                #  neighbourhood of points (i.e. are not at the boundary of an object).
                # Once we will have a more robust way to compare patches, remove this!
                patch_generator.set_filter_by_point_count(args.filter_by_point_count, args.point_count_coefficient)
                time_end = time.time()
                logging.warning(f'PMB setup time: {time_end - time_start}')

                test_dataset = cossad_dloaders.DatasetCossad(
                    args.cossad_data_path, 'train', dataset_tpl='.*', class_tpl=real_class,
                    anomaly_tpl='.*', index_tpl='.*', normalize=False)
                collate_fn = util_dloaders.dloader_dict_collate
                test_loader = DataLoader(test_dataset, num_workers=num_workers,
                                         batch_size=1, shuffle=True, drop_last=True,
                                         collate_fn=collate_fn)

                pbar = tqdm(enumerate(test_loader), total=len(test_loader))
                for idx, item in pbar:  # type: ItemDict
                    if item['np_pointcloud'] is None:
                        continue
                    assert (len(item['np_pointcloud']) == 1)
                    np_pointcloud_test = item['np_pointcloud'][0]
                    input_file_test = item['input_file'][0]
                    label_test = item['object_label'][0]
                    # print(f'path: {input_file}, label: {label}')
                    # np.save(f'np_pointcloud_test_{idx}.npy', np_pointcloud_test)
                    object_stats = {}

                    if np_pointcloud_test is None:
                        logging.warning(f'For item {idx}, received no data')
                        continue
                    if idx >= args.obj_per_class:
                        break

                    time_start = time.time()
                    reg_stats_test, registered_pc_test = register(basic_template, np_pointcloud_test, real_class)
                    if registered_pc_test is None:
                        continue

                    if args.perturb_rotation_deg != 0.0 or args.perturb_translation_percent != 0.0:
                        logging.warning(
                            f'Applying random perturbation to registered test PC '
                            f'(rotation_deg={args.perturb_rotation_deg}, '
                            f'translation_percent={args.perturb_translation_percent}). '
                            'This is for ABLATION STUDIES ONLY — do not use in prod runs.'
                        )
                        perturb = RandomPerturbation(
                            rotation_deg=args.perturb_rotation_deg,
                            translation_percent=args.perturb_translation_percent,
                        )
                        registered_pc_test = perturb.perturb(registered_pc_test)

                    patch_center_indices_test, feature_vectors_test, patch_center_indices_filtered_test, _, patch_centers_indices_skipped = \
                        feature_retriever.get_features(registered_pc_test)
                    object_stats['reg_test_to_template'] = reg_stats_test.mean_dist_src_to_target
                    reg_accuracies.append(reg_stats_test.mean_dist_src_to_target)

                    # This is a debugging code.
                    # np.save(f'registered_pc_test_{idx}.npy', registered_pc_test)
                    # np.save(f'patch_center_indices_test_{idx}.npy', patch_center_indices_test)
                    # np.save(f'feature_vectors_test_{idx}.npy', feature_vectors_test)
                    # np.save(f'patch_center_indices_filtered_test_{idx}.npy', patch_center_indices_filtered_test)

                    if args.do_visualize:
                        uvis.show_registered_np_pointcloud(registered_pc_test, basic_template)
                        np_allcenters_test = registered_pc_test[patch_center_indices_filtered_test]
                        uvis.show_registered_np_pointcloud(np_allcenters_test, patch_generator.patch_centers)

                    # Do a PC-wide prediction
                    max_effect_size, effect_sizes = memory_bank.detect_anomaly(feature_vectors_test, patch_center_indices_test)
                    image_predictions.append(max_effect_size)
                    image_labels.append(label_test)
                    object_stats['input_file_test'] = input_file_test
                    object_stats['npz_file'] = item['npz_file'][0]
                    object_stats['true_label'] = label_test
                    object_stats['risk_score'] = max_effect_size
                    object_stats['anomaly_type'] = item['anomaly_type'][0]
                    object_stats['inference_points'] = len(patch_center_indices_filtered_test)
                    object_stats['skipped_points'] = len(patch_centers_indices_skipped)

                    # Do a per-point risk prediction
                    anomaly_scores_gauss = anomaly_score_for_points(
                        registered_pc_test, patch_center_indices_filtered_test, effect_sizes, args.patch_radius)
                    # anomaly_scores_gauss = anomaly_score_for_points_torch(
                    #     registered_pc_test, patch_center_indices_filtered_test, effect_sizes, args.patch_radius, args.device)
                    # assert np.allclose(anomaly_scores_gauss, anomaly_scores_gauss_torch, rtol=1e-4, atol=1e-6), \
                    #     'Divergence between anomaly scores computed on GPU and CPU'

                    time_end = time.time()
                    inference_time = time_end - time_start
                    object_stats["inference_time"] = inference_time
                    inference_times.append(inference_time)

                    anomaly_mask = item['point_mask'][0]  # Actual values, where 1 means anomaly
                    try:
                        pixel_rocauc_gauss = roc_auc_score(anomaly_mask, anomaly_scores_gauss)
                        object_stats['pixel_rocauc_gauss'] = pixel_rocauc_gauss
                    except Exception as e:
                        logging.warning(f'Failed to compute pixel ROC AUC: {e}')
                        object_stats['pixel_rocauc_gauss'] = 0

                    # For per-point AUROC, keep the number of points small (~5K per object).
                    # So the final points array, accumulated across all objects, will be of a reasonable size
                    pixel_predictions_dsampled, anomaly_mask_dsampled = downsample(
                        anomaly_scores_gauss, anomaly_mask, n_samples=DSAMPLED_NPOINTS_FOR_AUC)
                    pixel_predictions.extend(pixel_predictions_dsampled)
                    pixel_labels.extend(anomaly_mask_dsampled)

                    if args.do_visualize:
                        np_allcenters_test = registered_pc_test[patch_center_indices_filtered_test]
                        uvis.show_registered_np_pointcloud(np_allcenters_test, patch_generator.patch_centers)
                        uvis.show_prediction(registered_pc_test, anomaly_mask, anomaly_scores_gauss)

                    if args.output_results_path is not None:
                        update_object_stats(args.output_results_path, object_stats, real_class, aggregated_for_class=False)
                        # Output prediction results, if we requested so
                        out_item = {}
                        for key in item.keys():
                            out_item[key] = item[key][0]
                            out_item['risk_score'] = max_effect_size
                            out_item['patch_risk_scores'] = effect_sizes
                            out_item['point_risk_scores'] = anomaly_scores_gauss
                        out_file_name = f'prediction_{os.path.basename(item["npz_file"][0])}'
                        np.savez(os.path.join(args.output_results_path, out_file_name), **out_item)
                        # Output statistics to the /stats subfolder
                        update_object_stats(os.path.join(args.output_results_path, 'stats'), object_stats, real_class,
                                            aggregated_for_class=False)

                image_labels_stack = np.stack(image_labels)
                image_predictions_stack = np.stack(image_predictions)
                logging.info(f'Labels stack: {image_labels_stack}')
                logging.info(f'Predictions stack: {image_predictions_stack}')

                mask = np.isfinite(image_predictions_stack)

                image_rocauc = roc_auc_score(image_labels_stack[mask], image_predictions_stack[mask])
                image_aupr = average_precision_score(image_labels_stack[mask], image_predictions_stack[mask])
                class_stats['class_name'] = real_class
                class_stats['O-ROCAUC'] = image_rocauc
                class_stats['O-PRAUC'] = image_aupr

                pixel_labels_stack = np.stack(pixel_labels)
                pixel_predictions_stack = np.stack(pixel_predictions)
                pixel_rocauc = roc_auc_score(pixel_labels_stack, pixel_predictions_stack)
                pixel_aupr = average_precision_score(pixel_labels_stack, pixel_predictions_stack)
                class_stats['P-ROCAUC'] = pixel_rocauc
                class_stats['P-PRAUC'] = pixel_aupr

                class_stats['mean_inference_time'] = np.mean(np.array(inference_times))
                class_stats['mean_chamfer'] = np.mean(np.array(reg_accuracies))

                # logging.warning(f'SciPy: AUC for {real_class}: {image_rocauc}')
                # logging.warning(f'SciPy: Average precision score for {real_class}: {image_aupr}')
                update_object_stats(os.path.join(args.output_results_path, 'stats'), class_stats, 'all',
                                    aggregated_for_class=True)

            except Exception as e:
                logging.error(
                    f"Inference failed: object={real_class} attempt={attempt + 1}/{args.attempts}: {e}",
                    exc_info=True
                )



if __name__ == "__main__":
    parser = set_pipeline_args()
    parser.add_argument("--num_patches", type=int, default=1000,
                        help='The number of patches to sample from the point cloud.')
    parser.add_argument("--filter_by_point_count", action='store_true', default=False,
                        help='Enable filtering of the candidate patch center by the number of points it has.'
                             'If the number of patch points less than the mean number of points for all PC patches, '
                             'the patch is discarded. Used for real3dad dataset; DO NOT USE for ashapenet')
    parser.add_argument('--output_results_path', type=str, default=None,
                        help='Path to the folder where the prediction will be stored into as .npz files, '
                             'as well as the statistics on predictions in form of .csv classes')
    parser.add_argument('--device', type=int, default=0,
                        help='The index of CUDA device for running the contrastive encoder.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='The number of patches to be used to compute the final anomaly score.')
    parser.add_argument('--top_k_aggregation_method', type=str, default='max',
                        help='The aggregation function to be taken across the k patches to compute '
                             'the final anomaly score.')
    parser.add_argument('--point_count_coefficient', type=float, default=0.0,
                        help='Coefficient for the point count filtering threshold. '
                             'Threshold = mean + std * coefficient. '
                             'Positive values make filtering stricter, negative - more lenient.')
    parser.add_argument('--pmb_type', type=str, default='indexed',
                        choices=['indexed', 'patchcore'],
                        help='ABLATION STUDIES ONLY. Type of the patch memory bank. '
                             '"indexed" is the COSSAD spatially-aware bank, '
                             '"patchcore" is the PatchCore FaissNN-based bank.')
    parser.add_argument('--attempts', type=int, default=1,
                        help='Number of independent inference attempts per object class '
                             '(for mean/std estimation). Each attempt re-builds the memory bank '
                             'from scratch, appending results to the shared stats CSV.')
    parser.add_argument('--extract_graph', action='store_true', default=False,
                        help='ABLATION: Extract kNN patch graph over template patches and save to output dir.')
    parser.add_argument('--graph_k', type=int, default=6,
                        help='Number of nearest neighbors for patch graph construction.')
    parser.add_argument('--perturb_rotation_deg', type=float, default=0.0,
                        help='ABLATION STUDIES ONLY. Max random rotation (degrees) applied to each '
                             'registered test PC before feature extraction. '
                             'If both this and --perturb_translation_percent are 0.0, no perturbation is applied.')
    parser.add_argument('--perturb_translation_percent', type=float, default=0.0,
                        help='ABLATION STUDIES ONLY. Max random translation (as a fraction of the PC extent) '
                             'applied to each registered test PC before feature extraction. '
                             'If both this and --perturb_rotation_deg are 0.0, no perturbation is applied.')

    args = parser.parse_args()
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_filename = f'pipeline_reg_based_{args.object_classes}_{timestamp}.log'
    setup_logging(log_filename, do_sync=True)
    pipeline(args)
