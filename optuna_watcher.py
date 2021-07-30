"""Optuna study watcher.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import warnings

warnings.warn = lambda *args, **kwargs: None

import argparse
from pprint import pprint
from typing import List

import numpy as np
import optuna
from optuna.trial import TrialState

from model_searcher.optuna_utils import create_load_study_with_config


def print_study_list(storage: str):
    studies = optuna.get_all_study_summaries(storage=storage)

    print("No | .. Start Date .. | nTry | . Score .| ... Name ... |")
    for i, study in enumerate(studies):
        msg = "{:02d} | ".format(i + 1)
        msg += "{} | ".format(study.datetime_start.strftime("%Y %m-%d %H:%M"))
        msg += "{:04d} | ".format(study.n_trials)
        msg += (
            "{:.03f}".format(study.best_trial.value)
            if study.best_trial is not None
            else " None"
        )
        msg += (
            "(⬆) | "
            if study.direction == optuna.study.StudyDirection.MAXIMIZE
            else "(⬇)"
            if study.direction == optuna.study.StudyDirection.MINIMIZE
            else "(·)"
        )
        msg += "{} |".format(study.study_name)

        print(msg)


def remove_study(storage: str, study_name: str, force: bool = False):
    print(
        "\x1b[1;103;90m :: Warning :: \x1b[30;41m You are about to DELETE \x1b[7m\x1b[103;90m\x1b[5m\x1b[4m {} \x1b[0m\x1b[1m\x1b[7;30;39m study.\x1b[0m".format(
            study_name
        )
    )
    if force is False:
        name = input("Re-type study name to delete: ")
        if name == study_name:
            optuna.delete_study(name, storage=storage)
            print("{} study has ben deleted.".format(name))
        else:
            print("Wrong name. {} study has NOT been deleted".format(study_name))
    else:
        optuna.delete_study(study_name, storage=storage)
        print("{} study has ben deleted.".format(study_name))


def query_trial(trial: optuna.trial.FrozenTrial):
    print(f"Index: {trial.number}, State: {trial.state}, Value: {trial.value}")

    print("-" * 10, "Params", "-" * 10)
    pprint(trial.params)

    print("-" * 10, "User Attributes", "-" * 10)
    pprint(trial.user_attrs)


def show_importance(i_study: optuna.Study):
    importance = optuna.importance.get_param_importances(i_study)
    pprint(importance)


def show_records(trials: List[optuna.trial.FrozenTrial], args: argparse.Namespace):
    states_to_show = [TrialState.COMPLETE]
    if args.show_prune:
        states_to_show += [TrialState.PRUNED]

    trials = [t for t in trials if t.state in states_to_show]
    if args.sort_date:
        trials = sorted(
            trials,
            key=lambda t: t.datetime_complete
            if t.state == TrialState.COMPLETE
            else t.datetime_start,
        )
    elif args.sort_attr == "":
        trials = sorted(trials, key=lambda t: t.value)
    else:
        trials = [t for t in trials if args.sort_attr in t.user_attrs.keys()]
        trials = sorted(trials, key=lambda t: t.user_attrs[args.sort_attr])

    workers = np.array([t.user_attrs["worker"] for t in trials])
    unique_workers = np.unique([t.user_attrs["worker"] for t in trials])

    print(
        ", ".join(
            [f"{worker}: {np.sum(worker == workers):,d}" for worker in unique_workers]
        )
    )
    print(f"Total: {len(trials):,d}")

    if args.direction == "maximize":
        trials = trials[::-1]

    td2str = lambda x: "{:02d}:{:02d}:{:02d}".format(
        int(x / 60 / 60), int(x / 60) % 60, x % 60
    )
    for i, t in enumerate(trials):
        if i >= args.n_top:
            break

        p_msg = ""
        a_msg = ""

        if args.params:
            params = {k: v for k, v in t.params.items() if k in args.params}

            p_keys = [k for k in params.keys()]
            p_keys = [
                ".".join(
                    [
                        kk[: 2 if i < (len(k.split(".")) - 1) else None]
                        for i, kk in enumerate(k.split("."))
                    ]
                )
                for k in p_keys
            ]
            params = {k: v for k, v in zip(p_keys, params.values())}
            p_msg = " | ".join([f"{k}: {v}" for k, v in params.items()])

        if args.attrs:
            attrs = {k: v for k, v in t.user_attrs.items() if k in args.attrs}

            a_msg = " | ".join([f"{k}: {v}" for k, v in attrs.items()])

        msg = f"{t.state.name} | {t.number:6,d} | {t.value if t.value else -float('inf'):7.5f} | {p_msg} | {a_msg}"
        if args.verbose > 0:
            msg += f" | {t.datetime_complete.strftime('%Y-%m%d-%H:%M:%S')} (runtime - {td2str(t.duration.seconds)})"
        print(msg)


def print_study_info(study: optuna.study.Study, storage: str):
    print("----- Study Information -----")
    print(f"  - name: {study.study_name}")
    print(f"  - direction: {study.direction}")
    print(f"  - Attributes")
    print("\n".join([f"    - {k}: {v}" for k, v in study.user_attrs.items()]))

    s_summary = [
        s
        for s in optuna.get_all_study_summaries(storage)
        if s.study_name == study.study_name
    ][0]
    print(f"  - Number of trials: {s_summary.n_trials:,d}")


def main(args: argparse.Namespace):
    if args.ls:
        print_study_list(args.storage)
        return
    if args.rm:
        remove_study(args.storage, args.study_name, force=args.force_rm)
        return

    if args.create or args.overwrite_attr:
        study = create_load_study_with_config(
            args.study_conf,
            args.study_name,
            storage=args.storage,
            load_if_exists=True,
            overwrite_user_attr=True,
        )
        if study is None:
            return
        if args.create:
            print(f"Study created in {args.storage}")
        else:
            print("Study user attributes has been overwritten.")
    else:
        study = optuna.load_study(args.study_name, storage=args.storage)

    if args.info:
        print_study_info(study, args.storage)
        return

    if args.importance:
        show_importance(study)
        return

    print("Loading ...")
    if args.query > 0:
        query_trial(study.trials[args.query])
        return

    trials = study.get_trials()

    show_records(trials, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optuna CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "storage",
        type=str,
        help="postgresql://your_id:your_password@server_address.com/optuna",
    )
    parser.add_argument(
        "--study-name", default="", type=str, help="Study name to query."
    )
    parser.add_argument("--verbose", default=1, type=int, help="Verbosity level")
    parser.add_argument(
        "--n-top", default=20, type=int, help="Number of top results to show"
    )
    parser.add_argument(
        "--ls",
        dest="ls",
        action="store_true",
        default=False,
        help="Show list of studies in storage",
    )
    parser.add_argument(
        "--rm",
        dest="rm",
        action="store_true",
        default=False,
        help="Remove ${study-name}",
    )
    parser.add_argument(
        "--force-rm",
        dest="force_rm",
        action="store_true",
        default=False,
        help="Force remove ${study-name}",
    )
    parser.add_argument(
        "--show-prune",
        action="store_true",
        default=False,
        help="Include pruned results",
    )
    parser.add_argument(
        "--params", nargs="*", type=str, help="Parameter names to display (Multiple)"
    )
    parser.add_argument(
        "--attrs",
        nargs="*",
        type=str,
        help="User attribute names to display (Multiple)",
    )
    parser.add_argument(
        "--importance",
        action="store_true",
        default=False,
        help="Show parameter importance of the study",
    )
    parser.add_argument(
        "--query",
        default=-1,
        type=int,
        help="Query index to display detailed parameters and user attributes",
    )
    parser.add_argument(
        "--sort-attr", default="", type=str, help="Sort by user attribute values."
    )
    parser.add_argument(
        "--sort-date", default=False, action="store_true", help="Sort by recent studies"
    )
    parser.add_argument(
        "--create",
        default=False,
        action="store_true",
        help="Create study named ${study-name}",
    )
    parser.add_argument(
        "--direction",
        default="minimize",
        type=str,
        help="Objective score direction. (minimize or maximize",
    )
    parser.add_argument(
        "--study-conf",
        default="model_searcher/config/study_conf.yaml",
        type=str,
        help="Study configuration yaml file path.",
    )
    parser.add_argument(
        "--overwrite-attr",
        default=False,
        action="store_true",
        help="Overwrite study user attributes.",
    )
    parser.add_argument(
        "--info", default=False, action="store_true", help="Show study information."
    )

    p_args = parser.parse_args()

    main(p_args)
