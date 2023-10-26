import os
import argparse
import glob
import yaml
import numpy as np
from pathlib import Path
import kwcoco

from angel_system.activity_hmm.core import load_and_discretize_data


# ----------------------------------------------------------------------------
def load_data(args):
    # Load from real system.
    dt = 0.25
    gt = []
    for sdir in os.listdir(args.base_dir) :
        if not os.path.isdir(f"{args.base_dir}/{sdir}"):
            continue
        #print(glob.glob(f"{args.base_dir}/{sdir}/*activity_detection_data.json"))
        activity_gt = glob.glob(f"{args.base_dir}/{sdir}/*.csv")[0]
        extracted_activity_detections = glob.glob(
                f"{args.base_dir}/{sdir}/*activity_detection_data.json")[0]
        gt.append(
            [
                sdir,
                load_and_discretize_data(
                    activity_gt, extracted_activity_detections, dt, 0.5
                ),
            ]
        )

    # For the purpose of fitting mean and std outside of the HMM, we can just
    # append everything together.

    X = []
    activity_ids = []
    time_windows = []
    valid = []
    for gt_ in gt:
        time_windowsi, class_str, Xi, activity_idsi, validi = gt_[1]
        time_windows.append(time_windowsi)
        X.append(Xi)
        activity_ids.append(activity_idsi)
        valid.append(validi)

    time_windows = np.vstack(time_windows)
    X = np.vstack(X)
    valid = np.hstack(valid)
    activity_ids = np.hstack(activity_ids)

    # ----------------------------------------------------------------------------
    # What was loaded is activity_id ground truth, but we want step ground truth.

    # Map activities to steps
    with open(args.config_fname, "r") as stream:
        config = yaml.safe_load(stream)

    dest = config["activity_mean_and_std_file"]

    activity_id_to_step = {}
    for step in config["steps"]:
        if isinstance(step["activity_id"], str):
            a_ids = [int(s) for s in step["activity_id"].split(",")]
        else:
            a_ids = [step["activity_id"]]

        for i in a_ids:
            activity_id_to_step[i] = step["id"]

    activity_id_to_step[0] = 0
    steps = sorted(list(set(activity_id_to_step.values())))
    assert steps == list(range(max(steps) + 1))

    true_step = [activity_id_to_step[activity_id] for activity_id in activity_ids]

    return time_windows, true_step, dest, X, steps

def load_data_kwcoco(args):
    coco = kwcoco.CocoDataset(args.kwcoco_fname)

    X = np.asarray(coco.images().lookup('activity_conf'))

    # time_windows: used to be the time stamps for each frame.
    # TODO: currently assuming framerate is 30. May need to add something
    #      that actually detects framerate at some point, but that info 
    #      is not available in the CocoDataset AFAIK.
    frame_indexes = coco.images().lookup('frame_index')
    time_windows = [[frame_ind/30, (frame_ind+1)/30] for frame_ind in frame_indexes]
    

    # true_step
    true_step = coco.images().lookup('activity_gt')

    # dest
    with open(args.config_fname, "r") as stream:
        config = yaml.safe_load(stream)

    dest = config["activity_mean_and_std_file"]

    # steps
    activity_id_to_step = {}
    for step in config["steps"]:
        if isinstance(step["activity_id"], str):
            a_ids = [int(s) for s in step["activity_id"].split(",")]
        else:
            a_ids = [step["activity_id"]]

        for i in a_ids:
            activity_id_to_step[i] = step["id"]

    activity_id_to_step[0] = 0
    steps = sorted(list(set(activity_id_to_step.values())))
    assert steps == list(range(max(steps) + 1))


    return time_windows, true_step, dest, X, steps, activity_id_to_step


# ----------------------------------------------------------------------------
def fit(time_windows, true_step, X, steps, activity_id_to_step):
    # Fit HMM.
    num_classes = max(true_step) + 1
    class_mean_conf = []
    class_std_conf = []
    med_class_duration = []
    true_mask = np.diag(np.ones(num_classes, dtype=bool))[true_step]
    for i in range(num_classes):
        class_mean_conf.append(np.mean(X[true_mask[:, i], :], axis=0))
        class_std_conf.append(np.std(X[true_mask[:, i], :], axis=0))

        indr = np.where(np.diff(true_mask[:, i].astype(np.int8)) < 0)[0]
        indl = np.where(np.diff(true_mask[:, i].astype(np.int8)) > 0)[0]

        #import ipdb; ipdb.set_trace()
        print(f"i = {i}")
        if true_mask[0, i] and indl[0] != 0:
            indl = np.hstack([0, indl])

        if true_mask[-1, i] and indr[-1] != len(true_mask) - 1:
            indr = np.hstack([indr, len(true_mask) - 1])

        # wins has shape (2, num_instances) where wins[0, i] indicates when the ith
        # instance starts and wins[1, i] indicates when the ith instance ends.
        wins = np.array(list(zip(indl, indr))).T

        # During (seconds) of each instance.
        if i == 17:
            continue
        twins = (wins[1] - wins[0]) / 30

        med_class_duration.append(np.mean(twins))

    med_class_duration = np.array(med_class_duration)
    class_mean_conf = np.array(class_mean_conf)
    class_std_conf = np.array(class_std_conf)

    # ----------------------------------------------------------------------------
    if False:
        # Fit dummy mean and cov.
        num_steps = len(steps)
        num_activities = len(activity_id_to_step)
        class_mean_conf2 = np.zeros((num_steps, num_activities))
        class_std_conf = np.ones((num_steps, num_activities)) * 0.05

        for key in activity_id_to_step:
            class_mean_conf2[activity_id_to_step[key], key] = 1

        ind = np.argmax(class_mean_conf2 * class_mean_conf, axis=1)
        class_mean_conf2[:] = 0
        for i, ii in enumerate(ind):
            class_mean_conf2[i, ii] = 1

        class_mean_conf = class_mean_conf2

    return class_mean_conf, class_std_conf, med_class_duration


# ----------------------------------------------------------------------------
def save(dest, class_mean_conf, class_std_conf, med_class_duration):
    np.save(dest, np.array([class_mean_conf, class_std_conf, med_class_duration], dtype=object))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_fname",
        dest="config_fname",
        type=Path,
        default="config/tasks/task_steps_cofig-recipe-coffee-shortstrings.yaml",
        #default="config/tasks/task_steps_config-recipe_coffee_trimmed_v3.yaml",
    )
    parser.add_argument(
        "--kwcoco_fname",
        dest="kwcoco_fname",
        type=Path,
        default="ros_bags/val_activity_preds_epoch40.mscoco.json"
        #default="/data/PTG/cooking/training/activity_classifier/TCN_HPL/logs/coffee_feat_v5_move_pts_norm/runs/2023-09-28_14-47-57/val_activity_preds_epoch40.mscoco.json"
    )
    parser.add_argument(
        "--base_dir",
        dest="base_dir",
        type=Path,
        default="ros_bags/rosbag2_2023_09_29-15_39_42_successful_TCN_all_acts_10/",
    )

    args = parser.parse_args()

    #time_windows, true_step, dest, X = load_data(args)
    time_windows, true_step, dest, X, steps, activity_id_to_step = load_data_kwcoco(args)
    #import ipdb; ipdb.set_trace()
    class_mean_conf, class_std_conf, med_class_duration = fit(time_windows, true_step, X, steps,
            activity_id_to_step)
    save(dest, class_mean_conf, class_std_conf, med_class_duration)


if __name__ == "__main__":
    main()
