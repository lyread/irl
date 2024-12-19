from abc import ABC, abstractmethod


class Logger(ABC):
    @abstractmethod
    def init_logger(self):
        pass

    @abstractmethod
    def init_counter(self):
        pass

    @abstractmethod
    def update_learning_log_during_episode(
        self, state, action, reward, next_state, terminated, truncated, info
    ):
        pass

    @abstractmethod
    def update_learning_log_after_episode(self):
        pass

    @abstractmethod
    def update_test_log_during_episode(
        self, state, action, reward, next_state, terminated, truncated, info
    ):
        pass

    @abstractmethod
    def update_test_log_after_episode(self):
        pass

    @abstractmethod
    def update_multiple_run_log(self):
        pass

    @abstractmethod
    def export_single_run_data(self):
        pass

    @abstractmethod
    def export_multiple_run_data(self):
        pass

    @abstractmethod
    def export_plot_data(self):
        pass
