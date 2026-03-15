from config import OrchNASConfig
from models.supernet import SuperNet
from nas.search_space import SearchSpace
from federated.client import OrchNASClient
from federated.server import OrchNASServer
from utils.helpers import build_dummy_dataset


def main():
    config = OrchNASConfig()

    global_model = SuperNet(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        max_depth=max(config.depth_choices),
        max_width=max(config.width_choices),
        kernel_choices=config.kernel_choices,
    )

    search_space = SearchSpace(
        depth_choices=config.depth_choices,
        width_choices=config.width_choices,
        kernel_choices=config.kernel_choices,
    )

    clients = []
    for k in range(config.num_clients):
        dataset = build_dummy_dataset(num_samples=128, num_classes=config.num_classes)

        device_state = {
            "energy_budget": 25.0 + 10 * k,
            "memory_budget": 80.0 + 20 * k,
            "compute_budget": 60.0 + 15 * k,
            "alpha": 1.0 + 0.1 * k,
        }

        client = OrchNASClient(
            client_id=k,
            model=global_model,
            train_dataset=dataset,
            config=config,
            device_state=device_state,
        )
        clients.append(client)

    server = OrchNASServer(global_model=global_model, search_space=search_space, config=config)

    for rnd in range(config.rounds):
        output = server.communication_round(clients)
        print(f"Round {rnd + 1}")
        print("Shared backbone:", output["shared_backbone_arch"])
        print("Client subnets:", output["client_subnets"])
        print("-" * 60)


if __name__ == "__main__":
    main()
