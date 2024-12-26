from dvclive import Live
import os
def dvc_test():
    with Live() as live:
        live.log_artifact("./models/model.pt", type="model")

if __name__ == '__main__':
    dvc_test()