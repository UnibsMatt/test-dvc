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
        live.log_artifact("./models/model.pt", type="model", name="pr1", desc="Fine-tuned Resnet50",
      labels=["resnet", "imagenet"],)
if __name__ == '__main__':
    dvc_test()