
class BaseModel(object):

    def __init__(self, input_: bool = False, output_: bool = True):
        self.input = input_
        self.output = output_

    def __exec_input(self):
        pass

    def __exec_process(self):
        pass

    def __exec_output(self):
        pass

    def run_task(self) -> any:
        if self.input:
            self.__exec_input()
        self.__exec_process()
        if self.output:
            return self.__exec_output()


class MultiThreadRequest(BaseModel):
    pass
