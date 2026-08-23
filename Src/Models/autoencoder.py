import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, window_len:int = 201):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=7,stride=2,padding=3),nn.ReLU(),
            nn.Conv1d(16,32, kernel_size=5,stride=2,padding=2),nn.ReLU(),
            nn.Conv1d(32,64,kernel_size=5,stride=2,padding=2),nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64,32,kernel_size=5,stride=2,padding=2),nn.ReLU(),
            nn.ConvTranspose1d(32,16,kernel_size=5,stride=2,padding=2),nn.ReLU(),
            nn.ConvTranspose1d(16,1,kernel_size=7,stride=2,padding=2)
        )
        self.window_len = window_len
    def forward(self, x):
        z= self.encoder(x)
        out = self.decoder(z)
        if out.shape[-1]!=x.shape[-1]:
            out = nn.functional.interpolate(out, size=x.shape[-1], mode="linear", align_corners=False)
        return out

    def reconstruction_error(self, x, k:int = 5):
        with torch.no_grad():
            recon = self(x)
            err= (recon-x)**2
            err = err.squeeze(1)
            greatest_errs, index = torch.topk(err, k=k, dim=-1)

            #I am picking the greatest err values because lighcurves are quiet for the most part

            return greatest_errs.mean(dim=-1)





