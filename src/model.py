from constants import *
import torch
import torchvision

class CNNP(torch.nn.Module):
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
                torch.nn.Conv2d(self.in_channels, self.intermediate_channels, kernel_size=(i)*2+1, stride=1, padding=i),
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
    
class UNetLikeLite(torch.nn.Module):
    """
    A lightweight U-Net-like architecture for image transformation tasks.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_blocks (int): Number of up/downsampling blocks.
        num_channels (int): Initial number of intermediate channels.
        device (str or torch.device): Device to use for computation.
    """

    def __init__(self, in_channels=NUM_CHANNELS, out_channels=NUM_CHANNELS, num_blocks=NUM_BLOCKS, num_channels=INTERMEDIATE_CHANNELS, device=DEVICE):

        class ConvolutionBlockLikeLite(torch.nn.Module):
            """
            A convolutional block for upsampling or downsampling.

            Args:
                in_channels (int): Number of input channels.
                out_channels (int): Number of output channels.
                type (str): Type of operation, 'up' for upsampling, 'down' for downsampling.
            """

            def __init__(self, in_channels, out_channels, type):
                super().__init__()
                
                self.layers = None
                if type == "up" or type == "u":
                    self.layers = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU(inplace=True),
                    )

                if type == "down" or type == "d":
                    self.layers = torch.nn.Sequential(
                        torch.nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                        torch.nn.BatchNorm2d(in_channels),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                        torch.nn.BatchNorm2d(out_channels),
                        torch.nn.ReLU(inplace=True),
                    )

            def forward(self, x):
                """
                Forward pass through the convolution block.

                Args:
                    x (torch.Tensor): Input tensor.

                Returns:
                    torch.Tensor: Processed output tensor.
                """

                return self.layers(x)
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_channels = num_channels
        self.device = device

        self.u_sample = torch.nn.Conv2d(self.in_channels, self.num_channels, 3, 1, 1)

        self.u_sampling_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.u_sampling_blocks.append(
                ConvolutionBlockLikeLite(self.num_channels, self.num_channels*2, "up")
            )
            self.num_channels *= 2

        self.sampling_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.sampling_blocks.append(
                torch.nn.Sequential(
                    ConvolutionBlockLikeLite(self.num_channels, self.num_channels, "down"),
                    ConvolutionBlockLikeLite(self.num_channels, self.num_channels, "up"),
                )
            )

        self.d_sampling_blocks = torch.nn.ModuleList()
        for _ in range(num_blocks):
            self.d_sampling_blocks.append(
                ConvolutionBlockLikeLite(self.num_channels, self.num_channels//2, "down")
            )
            self.num_channels //= 2

        self.d_sample = torch.nn.Conv2d(self.num_channels, self.out_channels, 3, 1, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, images, images_=None):
        """
        Forward pass through the UNet-like model.

        Args:
            images (torch.Tensor): Input images.
            images_ (torch.Tensor, optional): Target images for computing loss.

        Returns:
            tuple: (output tensor, loss if images_ is provided, else None)
        """

        out_ = images

        residual_blocks = []

        out_ = self.u_sample(out_)
        for block in self.u_sampling_blocks:
            out_ = block(out_)
            residual_blocks.append(out_)

        for block in self.sampling_blocks:
            out_ = block(out_) + out_

        for block in self.d_sampling_blocks:
            out = residual_blocks.pop()
            out_ = block(out_ + out)
        out_ = self.d_sample(out_)
        
        out_ = self.sigmoid(out_)

        loss = None
        if images_ is not None:
            loss = torch.nn.functional.mse_loss(out_, images_)
        return out_, loss
