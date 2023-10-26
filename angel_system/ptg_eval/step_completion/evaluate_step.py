import logging
import yaml
import argparse
import logging
from pathlib import Path
from typing import List
from typing import Optional
import kwcoco

import numpy as np
import numpy.typing as npt

from angel_system.data.common.load_data import (
    steps_from_dive_csv,
    steps_from_ros_export_json,
    objs_as_dataframe,
    add_inter_steps_to_step_gt,
    sanitize_str,
)
from angel_system.data.common.discretize_data import discretize_data_to_windows
from angel_system.ptg_eval.common.visualization import (
    EvalVisualization,
)
from angel_system.activity_hmm.core import ActivityHMM
from angel_system.ptg_eval.common.compute_scores import EvalMetrics


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ptg_eval_step")

def compute_steps_from_activities_json(video_dset, labels):

    dt = 1 / 30
    class_str = labels
    N =  len(class_str)
    med_class_duration = np.ones(N) * 5
    class_mean_conf = np.ones(N) * 0.5
    class_std_conf = np.ones(N) * 0.1

    model = ActivityHMM(
            dt,
            class_str,
            med_class_duration,
            num_steps_can_jump_fwd=0,
            num_steps_can_jump_bck=0,
            class_mean_conf=class_mean_conf,
            class_std_conf=class_std_conf,
            )

    model.step

    return 0

def gt_from_activities_json(video_dset, labels):

    return 0

    

def run_eval(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time_window = args.time_window
    uncertainty_pad = args.uncertainty_pad

    with open(args.config_fn, "r") as stream:
        config = yaml.safe_load(stream)
    labels = [sanitize_str(l["description"]) for l in config["steps"]]
    if "background" not in labels:
        labels.insert(0, "background")
    print("Labels: ", labels)

    coco = kwcoco.CocoDataset(args.activity_mscoco_fpath)
    video_ids = np.unique(np.asarray(coco.images().lookup('video_id')))

    # Loop over gt/pred pairs, gathering input data for eval.
    gt_true_mask: Optional[npt.NDArray] = None
    dets_per_valid_time_w: Optional[npt.NDArray] = None
    #for i, (gt_fpath, pred_fpath) in enumerate(args.step_gt_pred_pair):
    for video_id in video_ids:
        log.info(f"Loading data from video {video_id}")
        image_ids = coco.index.vidid_to_gids[video_id]
        video_dset = coco.subset(gids=image_ids, copy=True)
        label_vec, activity_seq = compute_steps_from_activities(video_dset, labels)

        #detections = steps_from_ros_export_json(pred_fpath.as_posix())
        detections = compute_steps_from_activities_json(video_dset, labels)
        #gt = steps_from_dive_csv(gt_fpath.as_posix(), labels)
        gt = gt_from_activities_json(video_dset, labels)

        import ipdb; ipdb.set_trace()

        min_start_time = min(
            min(gt, key=lambda a: a.start).start,
            min(detections, key=lambda a: a.start).start,
        )
        max_end_time = max(
            max(gt, key=lambda a: a.end).end,
            max(detections, key=lambda a: a.start).start,
        )
        if args.add_inter_steps or args.add_before_finished_steps:
            gt = add_inter_steps_to_step_gt(
                gt,
                labels,
                min_start_time,
                max_end_time,
                add_inter_steps=args.add_inter_steps,
                add_before_after_steps=args.add_before_finished_steps,
            )

        # Make gt/detections pd.DataFrame instance to be consistent with downstream
        # implementation.
        gt = objs_as_dataframe(gt)
        detections = objs_as_dataframe(detections)

        # Local masks for this specific file pair
        (
            l_gt_true_mask,
            l_dets_per_valid_time_w,
            l_time_windows,
        ) = discretize_data_to_windows(
            labels,
            gt,
            detections,
            time_window,
            uncertainty_pad,
            min_start_time,
            max_end_time,
        )

        # for each pair, output separate activity window plots
        log.info("Visualizing this detection set against respective ground-truth.")
        pair_out_dir = output_dir / f"pair_{gt_fpath.stem}_{pred_fpath.stem}"
        vis = EvalVisualization(labels, None, None, output_dir=pair_out_dir)
        vis.plot_activities_confidence(gt, detections, min_start_time, max_end_time)

        # Stack with global set
        if gt_true_mask is None:
            gt_true_mask = l_gt_true_mask
        else:
            gt_true_mask = np.concatenate([gt_true_mask, l_gt_true_mask])
        if dets_per_valid_time_w is None:
            dets_per_valid_time_w = l_dets_per_valid_time_w
        else:
            dets_per_valid_time_w = np.concatenate(
                [dets_per_valid_time_w, l_dets_per_valid_time_w]
            )

    assert gt_true_mask is not None, "No ground truth loaded."
    assert dets_per_valid_time_w is not None, "No predictions loaded"

    # ============================
    # Metrics
    # ============================
    metrics = EvalMetrics(
        labels, gt_true_mask, dets_per_valid_time_w, output_dir=output_dir
    )
    metrics.precision_recall()

    log.info(f"Saved metrics to {metrics.output_dir}")

    # ============================
    # Plot
    # ============================
    vis = EvalVisualization(
        labels, gt_true_mask, dets_per_valid_time_w, output_dir=output_dir
    )
    vis.confusion_mat()

    log.info(f"Saved plots to {vis.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-pred-pair",
        help=(
            "Specification of a pair of filepaths that refer to the "
            "ground-truth CSV file and prediction result JSON file, "
            "respectively. This option may be repeated any number of times "
            "for independent pairs."
        ),
        dest="step_gt_pred_pair",
        type=Path,
        nargs=2,
        default=[],
        action="append",
    )
    parser.add_argument(
        "--activity-mscoco-fpath",
        help=(
            "Activity mscoco filepath with gt and activity predictions.",
        ),
        dest = "activity_mscoco_fpath",
        type=Path,
        default = "ros_bags/val_activity_preds_epoch40.mscoco.json",
    )
    parser.add_argument(
        "--config_fn",
        type=Path,
        help="Task configuration file",
    )
    parser.add_argument(
        "--time_window",
        type=float,
        default=1,
        help="Time window in seconds to evaluate results on.",
    )
    parser.add_argument(
        "--uncertainty_pad",
        type=float,
        default=0.5,
        help="Time in seconds to pad the ground-truth regions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval/step",
        help="Folder to save results and plots to",
    )
    parser.add_argument(
        "--add_inter_steps",
        action="store_true",
        help="Adds interstitial steps to the ground truth",
    )
    parser.add_argument(
        "--add_before_finished_steps",
        action="store_true",
        help="Adds before and finished steps to the ground truth",
    )

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
