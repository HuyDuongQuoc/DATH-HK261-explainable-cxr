def build_loader_kwargs(training_config, pin_memory, batch_size=None):
    num_workers = int(training_config["num_workers"])
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    kwargs = {
        "batch_size": int(
            training_config["batch_size"] if batch_size is None else batch_size
        ),
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        kwargs.update(
            {
                "persistent_workers": bool(
                    training_config["persistent_workers"]
                ),
                "prefetch_factor": int(training_config["prefetch_factor"]),
            }
        )
    return kwargs
