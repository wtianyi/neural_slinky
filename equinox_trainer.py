from typing import Any, Union, Optional, List
import inspect
from argparse import _ArgumentGroup, ArgumentParser, Namespace
from pytorch_lightning.utilities.argparse import (
    _defaults_from_env_vars,
    add_argparse_args,
    from_argparse_args,
    parse_argparser,
    parse_env_variables,
)
from pytorch_lightning.callbacks import Callback


class EquinoxTrainer:
    @_defaults_from_env_vars
    def __init__(
        self,
        logger=True,
        enable_checkpointing: bool = True,
        callbacks: Optional[Union[List[Callback], Callback]] = None,
        enable_progress_bar: bool = True,
        enable_model_summary: bool = True,
        max_epochs: Optional[int] = None,
        min_epochs: Optional[int] = None,
        fast_dev_run: Union[int, bool] = False,
        check_val_every_n_epoch: Optional[int] = 1,
        val_check_interval: Optional[Union[int, float]] = None,
        log_every_n_steps: int = 50,
    ):
        """
        Args:
            callbacks: Callbacks to use during training.
                Default: ``None``.

            check_val_every_n_epoch: Perform a validation loop every after every `N` training epochs. If ``None``,
                validation will be done solely based on the number of training batches, requiring ``val_check_interval``
                to be an integer value.
                Default: ``1``.

            enable_checkpointing: Whether to enable checkpointing.
                Default: ``True``.

            enable_model_summary: Whether to enable model summarization by default.
                Default: ``True``.

            enable_progress_bar: Whether to enable the progress bar.
                Default: ``True``.

            fast_dev_run: Runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                of train, val and test to find any bugs (ie: a sort of unit test).
                Default: ``False``.

            max_epochs: Stop training once this number of epochs is reached. Disabled by default (None).
                If both max_epochs and max_steps are not specified, defaults to ``max_epochs = 1000``.
                To enable infinite training, set ``max_epochs = -1``.

            min_epochs: Force training for at least these many epochs. Disabled by default (None).

            logger: Logger for experiment tracking.
                A ``True`` value defaults to a ``WandbLogger``.
                Default: ``True``.

            log_every_n_steps: How often to log within steps (batches).
                Default: ``50``.

            val_check_interval: How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check
                after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training
                batches. An ``int`` value can only be higher than the number of training batches when
                ``check_val_every_n_epoch=None``, which validates after every ``N`` training batches
                across epochs or during iteration-based training.
                Default: ``1.0``.
        """
        pass

    @classmethod
    def default_attributes(cls) -> dict:
        init_signature = inspect.signature(cls)
        return {k: v.default for k, v in init_signature.parameters.items()}

    @classmethod
    def from_argparse_args(
        cls: Any, args: Union[Namespace, ArgumentParser], **kwargs: Any
    ) -> Any:
        return from_argparse_args(cls, args, **kwargs)

    @classmethod
    def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
        return parse_argparser(cls, arg_parser)

    @classmethod
    def match_env_arguments(cls) -> Namespace:
        return parse_env_variables(cls)

    @classmethod
    def add_argparse_args(
        cls, parent_parser: ArgumentParser, **kwargs: Any
    ) -> Union[_ArgumentGroup, ArgumentParser]:
        return add_argparse_args(cls, parent_parser, **kwargs)
