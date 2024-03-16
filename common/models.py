from queue import Queue
from enum import Enum


class DataSet(Enum):
    """
    Dataset type enum.
    """
    WIKI_PEDIA = "wikipedia"
    FB60K_NYT10 = "FB60K-NYT10"
    UMLS_PUB_MED = "UMLS-PubMed"


class BaseModel(object):
    """
    Base class for all models.
    """
    def __init__(self,
                 name_: str,
                 input_: bool = False,
                 output_: bool = True):
        """
        :param name_: Model name.
        :param input_: whether the model has input.
        :param output_: whether the model has output.
        """
        self.name = name_
        self.input = input_
        self.output = output_
        # TODO: add logger

    def __exec_input(self, *args, **kwargs):
        pass

    def __exec_process(self, *args, **kwargs):
        pass

    def __exec_output(self):
        pass

    def run_task(self, *args, **kwargs) -> any:
        """Main function to run task."""
        if self.input:
            self.__exec_input(*args, **kwargs)
        self.__exec_process(args, kwargs)
        if self.output:
            return self.__exec_output()


class MultiThreadRequest(BaseModel):
    """
    Multi-thread request model.
    """
    def __init__(self,
                 input_: bool,
                 name_: str,
                 queue_size: int,
                 produce_thread: int,
                 consumer_thread: int):
        """
        :param input_: whether the model has input.
        :param name_: model name.
        :param queue_size: multi-thread queue size.
        :param produce_thread: produce thread number.
        :param consumer_thread: consumer thread number.
        """
        super(MultiThreadRequest, self).__init__(name_=name_,
                                                 input_=input_,
                                                 output_=True)
        self.queue = Queue(queue_size)
        self.result_queue = Queue(queue_size)
        self.produce_thread = produce_thread
        self.consumer_thread = consumer_thread
        self.params = None
        # TODO: add logger

    def __produce(self, *args, **kwargs):
        pass

    def __consume(self, *args, **kwargs):
        pass

    def __get_result(self) -> any:
        pass

    def __exec_process(self, *args, **kwargs):
        pass

    def __exec_output(self) -> any:
        return self.__get_result()
