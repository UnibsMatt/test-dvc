import random

from dvclive import Live
import os
os.chdir("..")
def dvc_test():
    with Live() as live:
        live.log_param("epochs", 50)
        for epoch in range(50):
            for metric_name, value in {"val": 2, "tra": 1}.items():
                live.log_metric(metric_name, value * random.randint(0,200))
                live.next_step()
        import numpy as np
        img_numpy = np.ones((500, 500), np.uint8) * 255
        live.log_image("numpy.png", img_numpy)
        live.log_artifact("./models/model.pt",
                          type="model",
                          name="pr1",
                          desc="Fine-tuned Resnet50",
                          labels=["resnet", "imagenet"],)
        live.make_report()
if __name__ == '__main__':
    dvc_test()