import torch


def fast_gradient_attack(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor, epsilon: float, norm: str = "2",
                         loss_fn=torch.nn.functional.cross_entropy):
    """
    Perform a single-step projected gradient attack on the input x.
    Parameters
    ----------
    logits: torch.Tensor of shape [B, K], where B is the batch size and K is the number of classes.
        The logits for each sample in the batch.
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the number of channels, and N is the image
       dimension.
       The input batch of images. Note that x.requires_grad must have been active before computing the logits
       (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    epsilon: float
        The desired strength of the perturbation. That is, the perturbation (before clipping) will have a norm of
        exactly epsilon as measured by the desired norm (see argument: norm).
    norm: str, can be ["1", "2", "inf"]
        The norm with which to measure the perturbation. E.g., when norm="1", the perturbation (before clipping)
         will have a L_1 norm of exactly epsilon (see argument: epsilon).
    loss_fn: function
        The loss function used to construct the attack. By default, this is simply the cross entropy loss.

    Returns
    -------
    torch.Tensor of shape [B, C, N, N]: the perturbed input samples.

    """
    norm = str(norm)
    assert norm in ["1", "2", "inf"]

    loss = loss_fn(logits, y)
    loss.backward()
    x_grad = x.grad
    
    if x_grad is None:
        raise ValueError("Gradient w.r.t. the data was None. Set x.requires_grad = True before computing the logits.")
    if norm in ["1", "2"]:
        x_grad_normalized = x_grad / x_grad.norm(p=float(norm), dim=(-1,-2), keepdim=True)
        perturbation = epsilon * x_grad_normalized
    else:
        perturbation = epsilon * torch.sign(x_grad)
    x_pert = x + perturbation
    x_pert = x_pert.clamp(0, 1)
    return x_pert.detach()
