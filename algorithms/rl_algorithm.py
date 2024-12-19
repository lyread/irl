from abc import ABC, abstractmethod

from tensorboardX import SummaryWriter


class ReinforcementLearningAlgorithm(ABC):
    """
    Abstract base class representing a reinforcement learning algorithm.

    This class defines the basic structure and interface for reinforcement learning algorithms.

    Args:
        name (str): Name of the reinforcement learning algorithm.
        init_method (str, optional): Initialization method. Defaults to None.
        exploration (str, optional): Exploration method. Defaults to None.
        tensorboard (bool, optional): Whether to use TensorBoard for logging. Defaults to False.
    """

    def __init__(
        self,
        name: str,
        init_method: str = None,
        exploration: str = None,
        tensorboard: bool = False,
    ):
        self.name = name
        self.init_method = init_method
        self.exploration = exploration
        self.tensorboard = tensorboard
        if self.tensorboard:
            self.writer = SummaryWriter()
            self.learning_iter = 0

    @abstractmethod
    def reset(self):
        """
        Reset the internal state of the algorithm.
        """
        pass

    @abstractmethod
    def save(self, file_name, suffix, chkpt, time_stampt):
        """
        Save the algorithm's state to a file.

        Args:
            file_name: Name of the file to save.
            suffix: Additional suffix for the file name.
            chkpt: Whether to save as a checkpoint.
            time_stampt: Whether to include a timestamp in the file name.
        """
        pass

    @abstractmethod
    def load(self, file_name, resume):
        """
        Load the algorithm's state from a file.

        Args:
            file_name: Name of the file to load.
            resume: Whether to resume training from the loaded state.
        """
        pass

    @abstractmethod
    def select_action(self, state, method):
        """
        Select an action based on the current state and method.

        Args:
            state: Current state of the environment.
            method: Method for action selection.
        """
        pass

    @abstractmethod
    def learn(
        self,
        state,
        action,
        next_state,
        next_action,
        reward,
        terminated,
        truncated,
        info,
    ):
        """
        Update the algorithm based on the transition information.

        Args:
            state: Current state.
            action: Action taken in the current state.
            next_state: Next state.
            next_action: Action taken in the next state.
            reward: Reward received.
            terminated: Whether the episode terminated after this transition.
            truncated: Whether the episode was truncated after this transition.
            info: Additional information provided by the environment.
        """
        pass
