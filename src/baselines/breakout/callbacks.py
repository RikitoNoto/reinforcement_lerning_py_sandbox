from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import os
import numpy as np


class ModelSaveCallback(BaseCallback):

    def __init__(
        self,
        log_dir_path: str,
        save_dir_path: str,
        verbose: int = 0,
        save_epoch: int = 10,
        save_only_best=True,
        overwrite=True,
    ):
        super().__init__(verbose)

        self.__best_mean_reward = -np.inf  # 最高の報酬
        self.__log_dir_path = log_dir_path  # ログファイルのディレクトリ
        self.__save_dir_path = save_dir_path  # 保存先のディレクトリ
        self.__save_epoch = save_epoch  # 保存間隔
        self.__save_only_best = save_only_best  # 最高値の場合のみ保存
        self.__overwrite = overwrite

        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        if self.num_timesteps % self.__save_epoch > 0:
            return True

        _, y = ts2xy(load_results(self.__log_dir_path), "timesteps")

        if len(y) <= 0:
            return True

        # 平均報酬がベスト平均報酬以上の時はモデルを保存
        mean_reward = np.mean(y[-self.__save_epoch :])

        if not self.__save_only_best or (mean_reward > self.__best_mean_reward):
            os.makedirs(self.__save_dir_path, exist_ok=True)
            if mean_reward > self.__best_mean_reward:
                self.__best_mean_reward = mean_reward
            self.locals["self"].save(
                os.path.join(self.__save_dir_path, "breakout_model")
            )
            if not self.__overwrite:
                self.locals["self"].save(
                    os.path.join(
                        self.__save_dir_path, f"breakout_model_{self.num_timesteps}"
                    )
                )

        return True
