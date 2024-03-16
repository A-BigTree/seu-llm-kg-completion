from queue import Queue


class BaseModel(object):

    def __init__(self,
                 name_: str,
                 input_: bool = False,
                 output_: bool = True):
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
        if self.input:
            self.__exec_input(*args, **kwargs)
        self.__exec_process(args, kwargs)
        if self.output:
            return self.__exec_output()


class MultiThreadRequest(BaseModel):
    def __init__(self,
                 input_: bool,
                 name_: str,
                 queue_size: int,
                 produce_thread: int,
                 consumer_thread: int):
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
