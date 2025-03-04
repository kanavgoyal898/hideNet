from constants import *
import torch

class Model(torch.nn.Module):
    """
    A convolutional neural network (CNN) with multiple convolutional heads, 
    followed by two additional convolutional blocks.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        intermediate_channels (int): Number of channels in the intermediate layers.
        num_heads (int): Number of parallel convolutional heads in the first layer.
        device (str or torch.device): The device to run the model on (e.g., "cpu", "mps", "cuda").

    Attributes:
        conv1 (list of torch.nn.Sequential): List of convolutional heads with different kernel sizes.
        conv2 (torch.nn.Sequential): Intermediate convolutional block.
        conv3 (torch.nn.Sequential): Final convolutional block.
        device (torch.device): Device on which the model is running.

    Methods:
        forward(images, images_=None):
            Performs a forward pass through the network.

            Args:
                images (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
                images_ (torch.Tensor, optional): Ground truth tensor for computing loss.

            Returns:
                tuple: (output tensor, loss tensor or None if `images_` is not provided).
    """

    def __init__(self, in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS, intermediate_channels=INTERMEDIATE_CHANNELS, num_heads=NUM_HEADS, device=DEVICE):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.intermediate_channels = intermediate_channels
        self.num_heads = num_heads
        self.device = device

        self.conv1 = torch.nn.ModuleList()
        for i in range(self.num_heads):
            layer = [
                torch.nn.Conv2d(self.in_channels, self.intermediate_channels, kernel_size=(i)*2+1, stride=1, padding=(i)*2//2),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Conv2d(self.intermediate_channels, self.intermediate_channels, kernel_size=3, stride=1, padding=1),
            ]
            self.conv1.append(torch.nn.Sequential(*layer))
        
        layers = [torch.nn.Conv2d(self.intermediate_channels, self.intermediate_channels, 3, 1, 1)]
        layers.append(torch.nn.LeakyReLU(inplace=True))
        layers.append(torch.nn.Conv2d(self.intermediate_channels, self.intermediate_channels, 3, 1, 1))
        self.conv2 = torch.nn.Sequential(*layers)

        layers = [torch.nn.Conv2d(self.intermediate_channels, self.intermediate_channels, 3, 1, 1)]
        layers.append(torch.nn.LeakyReLU(inplace=True))
        layers.append(torch.nn.Conv2d(self.intermediate_channels, self.out_channels, 3, 1, 1))
        self.conv3 = torch.nn.Sequential(*layers)

    def forward(self, images, images_=None):
        """
        Performs a forward pass through the model.

        Args:
            images (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            images_ (torch.Tensor, optional): Ground truth tensor for computing loss.

        Returns:
            tuple: (output tensor, loss tensor or None if `images_` is not provided).
        """

        out1 = sum([layer(images) for layer in self.conv1])
        out2 = self.conv2(out1)
        out_ = self.conv3(out1 + out2)

        loss = None
        if images_ is not None:
            loss = torch.nn.functional.mse_loss(out_, images_)
        return out_, loss
