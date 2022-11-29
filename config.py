import argparse

def get_config():
    parser = argparse.ArgumentParser(description='FedAvg')

    """
    
    TODO:

    Implement you own hyperparameters

    """

    config = parser.parse_args()
    return config