"""Write description here.

- Author: Jongkuk Lim
- Contact: limjk@jmarple.ai
"""
import argparse

from model_searcher.model_optimizer import ModelSearcher

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Searching for the best model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d", "--data", type=str, default="data/coco.yaml", help="Dataset yaml file"
    )
    parser.add_argument(
        "-nt",
        "--n-trials",
        type=int,
        default=1000,
        help="Number of trials for the searching",
    )
    parser.add_argument(
        "-ts",
        "--test-step",
        type=int,
        default=1,
        help="Perform test on every ${test_step} epochs",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=8,
        help="maximum number of dataloader workers",
    )
    parser.add_argument(
        "-is",
        "--img-size",
        nargs="+",
        type=int,
        default=[640, 640],
        help="[train, test] image sizes",
    )
    parser.add_argument(
        "-noo",
        "--no-optimize-option",
        action="store_true",
        help="No optimization of training options.",
    )
    parser.add_argument(
        "-noh",
        "--no-optimize-hyp",
        action="store_true",
        help="No optimization of hyper-parameters.",
    )
    parser.add_argument(
        "-noa",
        "--no-optimize-aug",
        action="store_true",
        help="No optimization of augmentation.",
    )
    parser.add_argument(
        "-nsm",
        "--no-search-model",
        action="store_true",
        help="No searching for the model.",
    )
    parser.add_argument(
        "--override-optimization",
        action="store_true",
        help="Override optimization method with above 4 arguments (noo, noh, noa, nsm)",
    )
    parser.add_argument(
        "--cfg",
        default="",
        type=str,
        help="Use fixed model if provided and ${no_search_model} flag is on.",
    )
    parser.add_argument(
        "-tnp",
        "--threshold-n-param",
        type=float,
        default=1.0,
        help="Threshold of parameter number from baseline parameter.",
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of epochs on each trial."
    )
    parser.add_argument(
        "--no-prune", action="store_true", help="No prune model while training."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="total batch size for all GPUs"
    )
    parser.add_argument(
        "--no-single-cls",
        dest="single_cls",
        default=True,
        action="store_false",
        help="train as multi-class dataset",
    )
    parser.add_argument(
        "--single-cls",
        dest="single_cls",
        default=True,
        action="store_true",
        help="train as single-class dataset",
    )
    parser.add_argument("--logdir", type=str, default="runs/", help="logging directory")
    parser.add_argument(
        "--name",
        default="",
        help="renames experiment folder exp{N} to exp{N}_{name} if supplied",
    )
    parser.add_argument(
        "--study-name",
        default="auto_yolo",
        type=str,
        help="Name of the study for Optuna",
    )
    parser.add_argument(
        "--study-conf",
        default="model_searcher/config/study_conf.yaml",
        type=str,
        help="Study configuration yaml file.",
    )
    parser.add_argument("--storage", default="", type=str, help="Optuna storage URL")
    parser.add_argument(
        "--wlog", default=False, action="store_true", help="Use wandb for logging."
    )
    parser.add_argument(
        "--wlog-project", default="auto_yolo", type=str, help="Wandb project name"
    )
    parser.add_argument(
        "--wlog-tags",
        nargs="*",
        type=str,
        default=[],
        help="Wandb custom tags (multiple)",
    )
    parser.add_argument(
        "--no-cache",
        default=False,
        action="store_true",
        help="train as multi-class dataset",
    )
    parser.add_argument("--n-skip", type=int, default=0, help="skip every n data.")
    args = parser.parse_args()

    searcher = ModelSearcher(
        args.data,
        img_size=args.img_size,
        device=args.device,
        epoch=args.epochs,
        batch_size=args.batch_size,
        single_cls=args.single_cls,
        log_root=args.logdir,
        model_name=args.name,
        optimize_option=not args.no_optimize_option,
        optimize_hyp=not args.no_optimize_hyp,
        optimize_augment=not args.no_optimize_aug,
        search_model=not args.no_search_model,
        fixed_model=args.cfg if args.cfg != "" else None,
        param_threshold=args.threshold_n_param,
        workers=args.workers,
        n_trials=args.n_trials,
        test_step=args.test_step,
        prune=not args.no_prune,
        wandb=args.wlog,
        wandb_tags=args.wlog_tags,
        wandb_project=args.wlog_project,
        not_cache=args.no_cache,
        override_optimization=args.override_optimization,
        n_skip=args.n_skip,
    )

    searcher.optimize(
        study_name=args.study_name,
        storage=args.storage if args.storage != "" else None,
        study_conf=args.study_conf,
    )
