from abc import ABC, abstractmethod


class AbstractMove(ABC):
    @abstractmethod
    def transform(previous_sample):
        raise Exception("To be overriden in concerte implementation")
