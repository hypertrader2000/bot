import torch
import torch.nn as nn

OBS_DIM = 66  # must match your bot's DRL_OBS_WINDOW + 2 (default 64 + 2)

class DummyPolicy(nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [batch, OBS_DIM]
        batch = obs.shape[0]
        logits = torch.zeros((batch, 3), dtype=obs.dtype, device=obs.device)
        logits[:, 1] = 3.0  # strongly prefer BUY
        return logits        # [HOLD, BUY, SELL]

if __name__ == "__main__":
    model = DummyPolicy().eval()
    example = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    traced = torch.jit.trace(model, example)
    traced.save("policy.ts")
    print("Saved models/policy.ts (TorchScript traced)")
