from queue import Queue
from enum import Enum
from common.config import LOG_TASK
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class DataSet(Enum):
    """
    Dataset type enum.
    """
    WIKI_PEDIA = "wikipedia"
    DB15K = "DB15K"
    FB15K = "FB15K"
    FB15K_237 = "FB15K-237"
    YAGO15K = "YAGO15K"


class BaseModel(object):
    """
    Base class for all models.
    """

    def __init__(self,
                 name_: str,
                 input_: bool = False,
                 output_: bool = True,
                 cost_time: bool = True):
        """
        :param name_: Model name.
        :param input_: whether the model has input.
        :param output_: whether the model has output.
        """
        self.name = name_
        self.input = input_
        self.output = output_
        self.cost_time = cost_time
        LOG_TASK.info(f"{self.name} (input: {self.input}, output: {self.output}) is initialized.")

    def exec_input(self, *args, **kwargs):
        """
        override this function to execute input.
        """
        pass

    def exec_process(self, *args, **kwargs):
        """
        override this function to execute process.
        """
        raise NotImplementedError

    def exec_output(self):
        """
        override this function to execute output.
        """
        pass

    def run_task(self, *args, **kwargs) -> any:
        """Main function to run task."""
        start_time = time.time()
        if self.input:
            self.exec_input(*args, **kwargs)
        self.exec_process(*args, **kwargs)
        end_time = time.time()
        if self.cost_time:
            LOG_TASK.info(f"{self.name} finished. Cost time: {end_time - start_time}s")
        if self.output:
            return self.exec_output()


class MultiThreadRequest(BaseModel):
    """
    Multi-thread request model.
    """

    def __init__(self,
                 name_: str,
                 queue_size: int,
                 produce_thread: int,
                 consumer_thread: int,
                 input_: bool = False,
                 cost_time: bool = True):
        """
        :param input_: whether the model has input.
        :param name_: model name.
        :param queue_size: multi-thread queue size.
        :param produce_thread: produce thread number.
        :param consumer_thread: consumer thread number.
        """
        super().__init__(name_=name_,
                         input_=input_,
                         output_=True,
                         cost_time=cost_time)
        self.queue = Queue(queue_size)
        self.cache_queue = Queue()
        self.result_queue = Queue()
        self.produce_thread = produce_thread
        self.consumer_thread = consumer_thread
        self.params = None
        LOG_TASK.info(f"Queue size: {queue_size}, produce thread: {produce_thread}, consumer thread: {consumer_thread}")

    def produce(self, *args, **kwargs):
        """
        override this function to produce data.
        """
        raise NotImplementedError

    def consume(self, *args, **kwargs):
        """
        override this function to consume data.
        """
        raise NotImplementedError

    def exec_process(self, *args, **kwargs):
        with ThreadPoolExecutor(max_workers=self.produce_thread + 1) as executor_p:
            producer_future = []
            for _ in range(self.produce_thread):
                future = executor_p.submit(self.produce, *args, **kwargs)
                producer_future.append(future)
            with ThreadPoolExecutor(max_workers=self.consumer_thread + 1) as executor_c:
                consumer_future = []
                for _ in range(self.consumer_thread):
                    future = executor_c.submit(self.consume, *args, **kwargs)
                    consumer_future.append(future)
                for future in as_completed(consumer_future):
                    future.result()
            for future in as_completed(producer_future):
                future.result()
