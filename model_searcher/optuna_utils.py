"""Utility module for Optuna.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import Union

import optuna
import yaml


def load_study_conf(study_conf):
    with open(study_conf) as f:
        study_conf = yaml.load(f, yaml.FullLoader)

    if "hyp_config" in study_conf:
        with open(study_conf["hyp_config"]) as f:
            hyp_conf = yaml.load(f, yaml.FullLoader)
            study_conf["study_attr"]["hyp_param"] = hyp_conf["param"]
            study_conf["study_attr"]["hyp_augment"] = hyp_conf["augment"]

    return study_conf


def create_load_study_with_config(
    study_conf,
    study_name,
    storage: Union[str, optuna.storages.RDBStorage, None] = None,
    load_if_exists: bool = True,
    overwrite_user_attr: bool = False,
):
    study_conf = load_study_conf(study_conf)

    assert study_name != "", "Study name must be given."
    assert study_conf["direction"] in [
        "minimize",
        "maximize",
    ], "Direction must be either 'minimize' or 'maximize'"

    try:
        study = optuna.create_study(
            study_name=study_name,
            direction=study_conf["direction"],
            storage=storage,
            load_if_exists=load_if_exists,
        )
    except optuna.exceptions.DuplicatedStudyError:
        print("Study already exists!")
        return None

    for k, v in study_conf["study_attr"].items():
        if overwrite_user_attr or k not in study.user_attrs:
            study.set_user_attr(k, v)

    return study


class OptunaParameterManager:
    def __init__(self, study_conf):
        self.local_conf = load_study_conf(study_conf)

        self.local_study_attr = self.local_conf["study_attr"]
        self.trial: Union[None, optuna.Trial] = None

        for k in self.local_study_attr.keys():
            self.__setattr__(k, self.__get_property(k))

    def set_trial(self, trial):
        self.trial = trial
        for k in self.trial.study.user_attrs.keys():
            self.__setattr__(k, self.__get_property(k))

    def __get_property(self, name):
        if self.trial and name in self.trial.study.user_attrs:
            return self.trial.study.user_attrs[name]
        else:
            return self.local_study_attr[name]
