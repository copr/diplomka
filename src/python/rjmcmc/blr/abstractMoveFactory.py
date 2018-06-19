from abc import ABC, abstractmethod


class AbstractMoveFactory(ABC):
    '''
    Implementace tohoto interfacu by mely zajistovat vytvareni moznych
    prechodu mezi dimenzemi.
    '''

    @abstractmethod
    def get_moves_from(self, k):
        raise Exception("To be overriden in concerte factory")
