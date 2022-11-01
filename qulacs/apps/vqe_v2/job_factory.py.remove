import uuid
from job import Job
from utils import to_string
import json


class JobFactory:
    def __init__(self, config):
        self.config = config
        pass

    def create(
        self,
        current_time,
        start_time,
        end_time,
        cost_history,
        param_history,
        iter_history,
    ):
        if self.config["gate"]["type"] == "direct":
            job = Job(
                str(uuid.uuid4()),
                current_time,
                end_time - start_time,
                self.config["nqubit"],
                self.config["depth"],
                self.config["gate"]["type"],
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                str(cost_history[-1]),
                str(param_history[-1]),
                str(iter_history[-1]),
                str(cost_history),
                to_string(param_history),
                str(iter_history),
                self.config["gate"]["noise"]["singlequbit"]["enabled"],
                self.config["gate"]["noise"]["singlequbit"]["value"]
                if "value" in self.config["gate"]["noise"]["singlequbit"]
                else None,
                self.config["gate"]["noise"]["twoqubit"]["enabled"],
                self.config["gate"]["noise"]["twoqubit"]["value"]
                if "value" in self.config["gate"]["noise"]["twoqubit"]
                else None,
                json.dumps(self.config),
            )
        else:
            job = Job(
                str(uuid.uuid4()),
                current_time,
                end_time - start_time,
                self.config["nqubit"],
                self.config["depth"],
                self.config["gate"]["type"],
                str(self.config["gate"]["parametric_rotation_gate_set"]),
                str(self.config["gate"]["bn"]["type"]),
                self.config["gate"]["bn"]["range"]
                if "range" in self.config["gate"]["bn"]
                else None,
                str(self.config["gate"]["bn"]["value"]),
                str(self.config["gate"]["cn"]["value"]),
                str(self.config["gate"]["r"]["value"]),
                self.config["gate"]["time"]["type"],
                self.config["gate"]["time"]["max_val"]
                if "max_val" in self.config["gate"]["time"]
                else None,
                self.config["gate"]["time"]["min_val"]
                if "min_val" in self.config["gate"]["time"]
                else None,
                str(self.config["gate"]["time"]["value"])
                if "value" in self.config["gate"]["time"]
                else None,
                str(cost_history[-1]),
                str(param_history[-1]),
                str(iter_history[-1]),
                str(cost_history),
                to_string(param_history),
                str(iter_history),
                self.config["gate"]["noise"]["singlequbit"]["enabled"],
                self.config["gate"]["noise"]["singlequbit"]["value"]
                if "value" in self.config["gate"]["noise"]["singlequbit"]
                else None,
                self.config["gate"]["noise"]["twoqubit"]["enabled"],
                self.config["gate"]["noise"]["twoqubit"]["value"]
                if "value" in self.config["gate"]["noise"]["twoqubit"]
                else None,
                json.dumps(self.config),
            )
        return job
