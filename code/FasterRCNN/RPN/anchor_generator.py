# -*- coding:utf-8 -*-
"""
Generate anchors for per features.

Description:
    Scaling feature map according to the scale of the original
    image, and than generate anchors with a fixed width ratio
    at each pixel.

"""
# **** modification history ****  #
# ******************************  #
# 2021/01/17,by Junlu Ding,create #


import torch
from torch.nn import Module


class AnchorGenerator(Module):
    """
    Generate anchor for feature maps and return anchor list
    for batch features.

    Args:
        anchor_sizes(tuple): base anchor size.
        anchor_ratios(tuple): base anchor size ratios.
    Returns:
        anchor list [Tensor(N,M,4),Tensor,..]
    """
    def __init__(self, anchor_sizes=(128, 256, 512), anchor_ratios=(0.5, 1.0, 2.0)):
        super(AnchorGenerator, self).__init__()
        if not isinstance(anchor_sizes[0], (list, tuple)):
            anchor_sizes = tuple((size,) for size in anchor_sizes)
        if not isinstance(anchor_ratios[0], (list, tuple)):
            anchor_ratios = (anchor_ratios,) * len(anchor_sizes)

        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios
        self.base_anchor = None
        return

    def get_base_anchor(self, data_device, data_type):
        """
        generate base anchors with a fixed width ratio.
        Args:
            data_device: tensor device.
            data_type: tensor data type.
        Return:

        """
        if self.base_anchor is not None:
            return
        anchor_sizes = torch.tensor(self.anchor_sizes, device=data_device, dtype=data_type)
        anchor_ratios = torch.tensor(self.anchor_ratios, device=data_device, dtype=data_type)

        base_anchors = []
        for size, ratio in zip(torch.tensor(anchor_sizes), torch.tensor(anchor_ratios)):
            anchor_h = size[:, None].expand((len(ratio), 1))
            anchor_w = (size * ratio)[:, None]
            anchor = torch.cat([-anchor_w, -anchor_h, anchor_w, anchor_h], dim=1) / 2
            base_anchors.append(anchor)
        self.base_anchor = torch.cat(base_anchors, dim=0)
        return

    def generate_anchor(self, feature_sizes, strides, device, data_type):
        """
        base anchors are generated for each pixel position on the basis of
        feature maps after scaling in proportion to the original image.

        Args:
            feature_sizes (List[Tuple]) : different feature maps size list.
            strides (List[Tuple]): The ratios of each feature maps relative
                                   to the original image size.
            device : tensor device.
            data_type : tensor data type.
        Return:
            anchor list [Tensor(M,4),Tensor,..]
        """
        anchor_list = []
        for size, stride in zip(feature_sizes,strides):
            feature_h = size[0]
            feature_w = size[1]

            # get coordinate system
            grid_x = torch.arange(feature_w, device=device, dtype=data_type) * stride[1]
            grid_y = torch.arange(feature_h, device=device, dtype=data_type) * stride[0]
            grid_y, grid_x = torch.meshgrid(grid_y, grid_x)
            grid_y = grid_y.flatten()[:, None]
            grid_x = grid_x.flatten()[:, None]

            coordinate = torch.cat((grid_x, grid_y, grid_x, grid_y),dim=1)
            anchor = coordinate.view(-1, 1, 4) + self.base_anchor
            anchor_list.append(anchor)
        return anchor_list

    def forward(self, batch_image_info, list_features):
        batch_size = batch_image_info[0]
        batch_height = batch_image_info[1]
        batch_width = batch_image_info[2]

        feature_sizes = [feature.size()[2:] for feature in list_features]
        data_device, data_type = list_features[0].device, list_features[0].dtype

        # get stride in every feature map
        strides = [(torch.as_tensor(batch_height/size[0], device=data_device, dtype=data_type),
                    torch.as_tensor(batch_width/size[1], device=data_device, dtype=data_type))
                    for size in feature_sizes]

        # get base anchor
        self.get_base_anchor(data_device, data_type)

        # generate anchors that mapping image size for different feature map
        anchor_list_over_features = self.generate_anchor(feature_sizes, strides, data_device, data_type)

        # generate anchors for every image
        list_anchors = []
        for anchor in anchor_list_over_features:
            list_anchors.append(anchor[None,:].expand(batch_size,anchor.size(0),9,4))
        return list_anchors


if __name__ == '__main__':

    batch_image_info = (2, 256, 256)
    feature_maps = [torch.randn(2, 3, 112, 112), torch.randn(2, 3, 64, 64)]
    anchor_generator = AnchorGenerator()
    list_anchors = anchor_generator(batch_image_info, feature_maps)
    print([anchor.size() for anchor in list_anchors])
