from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torch import Tensor
from torch.utils.data import Dataset

from sklearn.datasets import make_moons

# Base class for all normalizing flows
class Flow(nn.Module):
    """Base class for transforms with learnable parameters.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute f(x) and log_abs_det_jac(x)."""
        raise NotImplementedError

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute f^-1(y) and inv_log_abs_det_jac(y)."""
        raise NotImplementedError

    def get_inverse(self):
        """Get inverse transformation."""
        return InverseFlow(self)


class InverseFlow(Flow):
    """Change the forward and inverse transformations."""

    def __init__(self, base_flow: Flow):
        """Create the inverse flow from a base flow.

        Args:
            base_flow: flow to reverse.
        """
        super().__init__()
        self.base_flow = base_flow
        if hasattr(base_flow, 'domain'):
            self.codomain = base_flow.domain
        if hasattr(base_flow, 'codomain'):
            self.domain = base_flow.codomain

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the forward transformation given an input x.

        Args:
            x: input sample. shape [batch_size, dim]

        Returns:
            y: sample after forward tranformation. shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward tranformation, shape [batch_size]
        """
        y, log_det_jac = self.base_flow.inverse(x)
        return y, log_det_jac

    def inverse(self, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse tranformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse tranformation, shape [batch_size]
        """
        x, inv_log_det_jac = self.base_flow.forward(y)
        return x, inv_log_det_jac


# Datasets
class CircleGaussiansDataset(Dataset):
    """Create a 2D dataset with Gaussians on a circle.

    Args:
        n_gaussians: number of Gaussians. int
        n_samples: number of sample per Gaussian. int
        radius: radius of the circle where the Gaussian means lie. float
        varaince: varaince of the gaussians. float
        seed: random seed: int
    """
    def __init__(
        self,
        n_gaussians: int = 6,
        n_samples: int = 100,
        radius: float = 3.,
        variance: float = .3,
        seed: int = 0
    ):
        self.n_gaussians = n_gaussians
        self.n_samples = n_samples
        self.radius = radius
        self.variance = variance

        np.random.seed(seed)
        radial_pos = np.linspace(0, np.pi*2, num=n_gaussians, endpoint=False)
        mean_pos = radius * np.column_stack((np.sin(radial_pos), np.cos(radial_pos)))
        samples = []
        for _, mean in enumerate(mean_pos):
            sampled_points = mean[:,None] + (np.random.normal(loc=0, scale=variance, size=n_samples), np.random.normal(loc=0, scale=variance, size=n_samples ))
            samples.append(sampled_points)
        p = np.random.permutation(self.n_gaussians * self.n_samples)
        self.X = np.transpose(samples, (0, 2, 1)).reshape([-1,2])[p]

    def __len__(self) -> int:
        return self.n_gaussians * self.n_samples

    def __getitem__(self, item: int) -> Tensor:
        x = torch.from_numpy(self.X[item]).type(torch.FloatTensor)
        return x


class MoonsDataset(Dataset):
    """Create a 2D dataset with spirals.

    Args:
        n_samples: number of sample per spiral. int
        seed: random seed: int
    """
    def __init__(self, n_samples: int = 1200, seed: int = 0):

        self.n_samples = n_samples

        np.random.seed(seed)
        self.X, _ = make_moons(n_samples=n_samples, shuffle=True, noise=.05, random_state=None)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, item: int) -> Tensor:
        x = torch.from_numpy(self.X[item]).type(torch.FloatTensor)
        return x


class SpiralDataset(Dataset):
    """Create a 2D dataset with spirals.

    Args:
        n_spirals: number of spiral. int
        n_samples: number of sample per spiral. int
        seed: random seed: int
    """
    def __init__(self, n_spirals: int = 2, n_samples: int = 600, seed: int = 0):
        self.n_spirals = n_spirals
        self.n_samples = n_samples

        np.random.seed(seed)
        radial_pos = np.linspace(0, np.pi*2, num=n_spirals, endpoint=False)
        samples = []
        for ix, radius in enumerate(radial_pos):
            n = np.sqrt(np.random.rand(n_samples, 1)) * 540 * (2 * np.pi) / 360
            d1x = -np.cos(n) * n + np.random.rand(n_samples, 1) * 0.5
            d1y = np.sin(n) * n + np.random.rand(n_samples, 1) * 0.5
            x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
            x += np.random.randn(*x.shape) * 0.01
            samples.append(x)

        p = np.random.permutation(self.n_spirals * self.n_samples)
        self.X = np.concatenate(samples, axis=0)[p]

    def __len__(self) -> int:
        return self.n_spirals * self.n_samples

    def __getitem__(self, item) -> Tensor:
        x = torch.from_numpy(self.X[item]).type(torch.FloatTensor)
        return x


# Visualization functions
def plot_density(model, loader=[], batch_size=100, mesh_size=5., device="cpu"):
    """Plot the density of a normalizing flow model. If loader not empty, it plots also its data samples.

    Args:
        model: normalizing flow model. Flow or StackedFlows
        loader: loader containing data to plot. DataLoader
        bacth_size: discretization factor for the mesh. int
        mesh_size: range for the 2D mesh. float
    """
    with torch.no_grad():
        xx, yy = np.meshgrid(np.linspace(- mesh_size, mesh_size, num=batch_size), np.linspace(- mesh_size, mesh_size, num=batch_size))
        coords = np.stack((xx, yy), axis=2)
        coords_resh = coords.reshape([-1, 2])
        log_prob = np.zeros((batch_size**2))
        for i in range(0, batch_size**2, batch_size):
            data = torch.from_numpy(coords_resh[i:i+batch_size, :]).float().to(device)
            log_prob[i:i+batch_size] = model.log_prob(data.to(device)).cpu().detach().numpy()

        plt.scatter(coords_resh[:,0], coords_resh[:,1], c=np.exp(log_prob))
        plt.colorbar()
        for X in loader:
            plt.scatter(X[:,0], X[:,1], marker='x', c='orange', alpha=.05)
        plt.tight_layout()
        plt.show()


def plot_samples(model, num_samples=500, mesh_size=5.):
    """Plot samples from a normalizing flow model. Colors are selected according to the densities at the samples.

    Args:
        model: normalizing flow model. Flow or StackedFlows
        num_samples: number of samples to plot. int
        mesh_size: range for the 2D mesh. float
    """
    x, log_prob = model.rsample(batch_size=num_samples)
    x = x.cpu().detach().numpy()
    log_prob = log_prob.cpu().detach().numpy()
    plt.scatter(x[:,0], x[:,1], c=np.exp(log_prob))
    plt.xlim(-mesh_size, mesh_size)
    plt.ylim(-mesh_size, mesh_size)
    plt.show()
