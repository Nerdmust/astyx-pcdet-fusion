import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # spatial_features = data_dict['spatial_features']
        lidar_spatial_features = data_dict['lidar_spatial_features']
        radar_spatial_features = data_dict['radar_spatial_features']
        # ups = []
        lidar_ups = []
        radar_ups = []
        # ret_dict = {}
        lidar_ret_dict = {}
        radar_ret_dict = {}
        # x = spatial_features
        lidar_x = lidar_spatial_features
        radar_x = radar_spatial_features
        #print(x.size()) ###
        # for i in range(len(self.blocks)):
        #     x = self.blocks[i](x)
        #     ########################################
        #     print(f'\nfor loop %d:' % i)###
        #     print(self.blocks[i]) ###
        #     print(x.size()) ###
        #     #######################################
        #     stride = int(spatial_features.shape[2] / x.shape[2])
        #     ret_dict['spatial_features_%dx' % stride] = x
        #     if len(self.deblocks) > 0:
        #         ups.append(self.deblocks[i](x))
        #         ######################################
        #         y=self.deblocks[i](x)
        #         ups.append(y)
        #         print('deblock:')
        #         print(y.size())
        #         #####################################
        #     else:
        #         ups.append(x)
        for i in range(len(self.blocks)):
            lidar_x = self.blocks[i](lidar_x)
            ########################################
            # print(f'\nfor loop %d:' % i)###
            # print(self.blocks[i]) ###
            # print(lidar_x.size()) ###
            #######################################
            lidar_stride = int(lidar_spatial_features.shape[2] / lidar_x.shape[2])
            lidar_ret_dict['lidar_spatial_features_%dx' % lidar_stride] = lidar_x
            if len(self.deblocks) > 0:
                lidar_ups.append(self.deblocks[i](lidar_x))
                ######################################
                # lidar_y=self.deblocks[i](lidar_x)
                # lidar_ups.append(lidar_y)
                # print('deblock:')
                # print(lidar_y.size())
                #####################################
            else:
                lidar_ups.append(lidar_x)
        for i in range(len(self.blocks)):
            radar_x = self.blocks[i](radar_x)
            ########################################
            # print(f'\nfor loop %d:' % i)###
            # print(self.blocks[i]) ###
            # print(radar_x.size()) ###
            #######################################
            radar_stride = int(radar_spatial_features.shape[2] / radar_x.shape[2])
            radar_ret_dict['radar_spatial_features_%dx' % radar_stride] = radar_x
            if len(self.deblocks) > 0:
                radar_ups.append(self.deblocks[i](radar_x))
                ######################################
                # radar_y=self.deblocks[i](radar_x)
                # radar_ups.append(radar_y)
                # print('deblock:')
                # print(radar_y.size())
                #####################################
            else:
                radar_ups.append(radar_x)

        # if len(ups) > 1:
        #     x = torch.cat(ups, dim=1)
        # elif len(ups) == 1:
        #     x = ups[0]
        if len(lidar_ups) > 1:
            lidar_x = torch.cat(lidar_ups, dim=1)
        elif len(lidar_ups) == 1:
            lidar_x = lidar_ups[0]
        if len(radar_ups) > 1:
            radar_x = torch.cat(radar_ups, dim=1)
        elif len(radar_ups) == 1:
            radar_x = radar_ups[0]

        # if len(self.deblocks) > len(self.blocks):
        #     x = self.deblocks[-1](x)
        if len(self.deblocks) > len(self.blocks):
            lidar_x = self.deblocks[-1](lidar_x)
        if len(self.deblocks) > len(self.blocks):
            radar_x = self.deblocks[-1](radar_x)

        # data_dict['spatial_features_2d'] = x
        data_dict['lidar_spatial_features_2d'] = lidar_x
        data_dict['radar_spatial_features_2d'] = radar_x
        data_dict['spatial_features_2d'] = torch.cat((lidar_x, radar_x), 1)

        return data_dict
