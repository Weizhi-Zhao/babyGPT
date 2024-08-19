import torch

checkpoint: dict[dict] = torch.load("checkpoints/ckpt_gptv.pt", weights_only=False, mmap=True)
for k, v in checkpoint['model'].items():
    if "transformer" in k:
        new_k = k.replace("transformer", "language_model")
        checkpoint['model'][new_k] = v
        checkpoint['model'].pop(k)

torch.save(checkpoint, "checkpoints/ckpt_new_gptv.pt")