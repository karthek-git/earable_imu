import flwr as fl


def fit_config(server_round: int):
    return {"batch_size": 32, "local_epochs": 5}


def main():

    strategy = fl.server.strategy.FedAvgAndroid(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_fn=None,
        on_fit_config_fn=fit_config,
        initial_parameters=None,
    )

    fl.server.start_server(server_address="localhost:8080",
                           strategy=strategy,
                           config=fl.server.ServerConfig(num_rounds=3))


if __name__ == "__main__":
    main()
