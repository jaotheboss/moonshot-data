from __future__ import annotations

import asyncio
import time
from typing import Any

from moonshot.src.redteaming.attack.attack_module import AttackModule
from moonshot.src.redteaming.attack.attack_module_arguments import AttackModuleArguments
from moonshot.src.utils.log import configure_logger

# Create a logger for this module
logger = configure_logger(__name__)


class DatasetAugmentation:
    async def generate(
        self,
        event_loop: Any,
        runner_args: dict,
        cancel_event: asyncio.Event,
    ) -> dict | None:
        """
        Asynchronously generates augmented data by loading and running attack modules.

        Args:
            event_loop (Any): The event loop to run asynchronous tasks.
            runner_args (dict): Arguments for the runner, including attack strategies.
            cancel_event (asyncio.Event): Event to signal cancellation of the process.

        Returns:
            dict: The response from the first attack module, or None if an error occurs.
        """

        self.event_loop = event_loop
        self.runner_args = runner_args
        self.cancel_event = cancel_event

        logger.info("[Data Augmentation] Starting data augmentation...")

        # ------------------------------------------------------------------------------
        # Part 1: Load attack module
        # ------------------------------------------------------------------------------
        logger.debug("[Data Augmentation] Part 1: Loading Attack Module...")
        loaded_attack_modules = []
        try:
            # load red teaming modules
            for attack_strategy_args in self.runner_args.get("attack_strategies", None):
                attack_module_attack_arguments = AttackModuleArguments(
                    dataset_prompts=attack_strategy_args.get("dataset_prompts", []),
                    augment_dataset=True,
                    cancel_event=self.cancel_event,
                )
                loaded_attack_module = AttackModule.load(
                    am_id=attack_strategy_args.get("attack_module_id"),
                    am_arguments=attack_module_attack_arguments,
                )
                loaded_attack_modules.append(loaded_attack_module)
        except Exception as e:
            logger.error(
                f"[Red teaming] Unable to load attack modules in attack strategy: {str(e)}"
            )

        # ------------------------------------------------------------------------------
        # Part 2: Run attack module(s)
        # ------------------------------------------------------------------------------
        logger.debug("[Data Augmentation] Part 2: Running Attack Module(s)...")

        responses_from_attack_module = []
        for attack_module in loaded_attack_modules:
            logger.info(
                f"[Data Augmentation] Starting to run attack module [{attack_module.name}]"
            )
            start_time = time.perf_counter()

            attack_module_response = await attack_module.perform_data_augmentation()
            logger.info(
                f"[Data Augmentation] Running attack module [{attack_module.name}] took "
                f"{(time.perf_counter() - start_time):.4f}s"
            )
            responses_from_attack_module.append(attack_module_response)

        # currently we only run one attack_module, so we return the first element in the list
        return responses_from_attack_module[0]
