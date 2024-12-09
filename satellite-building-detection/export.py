import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from src.models.UANet import UANet


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Set lighting seed
    L.seed_everything(cfg.seed)

    # Enable tensor cores
    torch.set_float32_matmul_precision("high")

    # Instantiate data module (See: https://lightning.ai/docs/pytorch/stable/data/datamodule.html)
    data_module = instantiate(cfg.dataset)

    # Instantiate model subcomponents
    network = instantiate(cfg.model.network)
    optimizer = instantiate(cfg.model.optimizer)
    criterion = instantiate(cfg.model.criterion)
    accuracy = instantiate(cfg.model.accuracy)

    # Load model from checkpoint, passing instantiated components
    ckpt_path = "./test/grax41h9/checkpoints/epoch=295-step=8584.ckpt"
    model = UANet.load_from_checkpoint(
        ckpt_path,
        network=network,
        optimizer=optimizer,
        criterion=criterion,
        accuracy=accuracy,
    )
    model.eval()

    # Get example input from the data module
    data_module.setup()
    dataloader = data_module.val_dataloader()
    example_input = next(iter(dataloader))[0][0].unsqueeze(
        0
    )  # Get the image batch, select the first imae and add back the batch dim

    # Export model to ONNX
    model.to_onnx(
        "model.onnx",
        example_input,
        export_params=True,
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        # dynamic_axes={
        #    "input": {0: "batch_size"},  # variable length axes
        #    "output": {0: "batch_size"},
        # },
    )


if __name__ == "__main__":
    main()
