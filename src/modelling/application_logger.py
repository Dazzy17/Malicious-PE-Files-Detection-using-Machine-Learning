from datetime import datetime


class CustomApplicationLogger:
    def __init__(self) -> None:
        pass

    def logger(self, file_obj, msg):
        """ 
        Input: 
            file_obj: file object 
            msg: message to be logged 
        Output: 
            Loggings the message to the file object 
        """
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_obj.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + msg + "\n"
        )


if __name__ == "__main__":
    pass
