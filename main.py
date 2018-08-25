import numpy as np
import os
import json
import argparse
from models import ArsAgent


def main():
    parser = argparse.ArgumentParser(description="Run ARS algorithm with given config")
    parser.add_argument("-c",
                        "--config",
                        type=str,
                        metavar="",
                        required=True,
                        help="Config file name - file must be available as .json in ./configs")

    parser.add_argument("-t",
                        "--train",
                        type=str,
                        metavar="",
                        default="true",
                        help="If true, the program will train, save and evaluate the model. If false, the program will "
                             "search for a checkpoint in ./checkpoints to load and evaluate the model.")

    args = parser.parse_args()

    # load config files
    with open(os.path.join(".", "configs", args.config), "r") as read_file:
        ars_config = json.load(read_file)

    np.random.seed(ars_config["general"]["seed_np"])

    ars_agent = ArsAgent(config=ars_config)

    if "true" == args.train.lower():
        ars_agent.train()
        ars_agent.save_model()
        total_reward = ars_agent.evaluate()
        print("Total Reward: {}".format(total_reward))
    else:
        ars_agent.load_model()
        total_reward = ars_agent.evaluate()
        print("Total Reward: {}".format(total_reward))


if __name__ == "__main__":
    main()

