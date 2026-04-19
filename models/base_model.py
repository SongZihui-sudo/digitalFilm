import torch


class model_base(torch.nn.Module):
    def __init__(self, rank, global_config: dict) -> None:
        super().__init__()

        self._rank: int = rank
        self._global_config: dict = global_config
        self._model: list[dict[str, torch.nn.Module]] = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("This method has not been implemented!")
    
    def state_dict(self, *args, **kwargs) -> dict:
        cur_state_dict = {}
        for name, module in self.named_children():
            cur_state_dict[name] = module.state_dict()
        
        return cur_state_dict
        
    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        try:
            for name, module in self.named_children():
                if name in state:
                    module.load_state_dict(state[name], strict=strict)
        except Exception as e:
            raise KeyError(f"Weights of missing model components: {name}, Error: {e} .")

        print("Model weights loaded successfully.")

if __name__ == "__main__":
    pass
