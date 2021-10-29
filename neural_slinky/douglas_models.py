import torch

class CompleteQuadratic(torch.nn.Module):
    def __init__(self, input_dim, latent_dim) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(input_dim, latent_dim)
    
    def forward(self, x):
        x_proj = self.projection(x)
        return torch.sum(x_proj * x_proj, dim=-1)


class DouglasModel(torch.nn.Module):
    def __init__(self, c_dxi, c_dxi_sq, c_dz_sq, c_dphi_sq):
        super().__init__()
        self.c_dxi = torch.nn.Parameter(torch.tensor(c_dxi))
        self.c_dxi_sq = torch.nn.Parameter(torch.tensor(c_dxi_sq))
        self.c_dz_sq = torch.nn.Parameter(torch.tensor(c_dz_sq))
        self.c_dphi_sq = torch.nn.Parameter(torch.tensor(c_dphi_sq))

    def forward(self, douglas_dof):
        """
        Args:
            douglas_dof: (... x 3) tensor. The columns are (dxi, dz, dphi)
        """
        dxi = douglas_dof[..., 0]
        dz = douglas_dof[..., 1]
        dphi = douglas_dof[..., 2]

        return (
            self.c_dxi * dxi
            + self.c_dxi_sq * dxi ** 2
            + self.c_dz_sq * dz ** 2
            + self.c_dphi_sq * dphi ** 2
        )

