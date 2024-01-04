import torch.nn as nn
import torch
import torch.nn.functional as func

from model import net_utils

class FlowPath(nn.Module):
    def __init__(self,
                 device = 'cuda',
                 leaky_relu_alpha=0.05,
                 dropout_rate=0.25,
                 num_channels_upsampled_context=32,
                 num_levels=5,
                 normalize_before_cost_volume=True,
                 channel_multiplier=1.,
                 use_cost_volume=True,
                 use_feature_warp=True,
                 accumulate_flow=True,
                 shared_flow_decoder=False,
                 init_flow = False):
        super(FlowPath, self).__init__()
        
        self._device = device
        self._leaky_relu_alpha = leaky_relu_alpha
        self._drop_out_rate = dropout_rate
        self._num_context_up_channels = num_channels_upsampled_context
        self._num_levels = num_levels
        self._normalize_before_cost_volume = normalize_before_cost_volume
        self._channel_multiplier = channel_multiplier
        self._use_cost_volume = use_cost_volume
        self._use_feature_warp = use_feature_warp
        self._accumulate_flow = accumulate_flow
        self._shared_flow_decoder = shared_flow_decoder
        self._init_flow = init_flow

        self._refine_model = self._build_refinement_model()
        self._flow_layers = self._build_flow_layers()
        if not self._use_cost_volume:
            self._cost_volume_surrogate_convs = self._build_cost_volume_surrogate_convs()
        if self._num_context_up_channels:
            self._context_up_layers = self._build_upsample_layers(
                num_channels=int(self._num_context_up_channels * channel_multiplier))

    def forward(self, feature_pyramid1, feature_pyramid2, training=True):
        """Run the model."""
        context = None
        flow = None
        flow_up = None
        context_up = None
        flows = []

        # Go top down through the levels to the second to last one to estimate flow.
        for level, (features1, features2) in reversed(
                list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[1:]):

            # init flows with zeros for coarsest level if needed
            if self._init_flow and flow_up is None:
                batch_size, _, height, width = features1.shape
                flow_up = torch.zeros([batch_size, height, width, 2]).to(self._device)
                if self._num_context_up_channels:
                    num_channels = int(self._num_context_up_channels *
                                       self._channel_multiplier)
                    context_up = torch.zeros([batch_size, height, width, num_channels]).to(self._device)

            # Warp features2 with upsampled flow from higher level.
            if flow_up is None or not self._use_feature_warp:
                warped2 = features2
            else:
                warp_up = net_utils.flow_to_warp(flow_up)
                warped2 = net_utils.resample(features2, warp_up)

            # Compute cost volume by comparing features1 and warped features2.
            features1_normalized, warped2_normalized = net_utils.normalize_features(
                [features1, warped2],
                normalize=self._normalize_before_cost_volume,
                center=self._normalize_before_cost_volume,
                moments_across_channels=True,
                moments_across_images=True)

            if self._use_cost_volume:
                cost_volume = net_utils.compute_cost_volume(features1_normalized, warped2_normalized, max_displacement=4)
            else:
                concat_features = torch.cat([features1_normalized, warped2_normalized], dim=1)
                cost_volume = self._cost_volume_surrogate_convs[level](concat_features)

            cost_volume = func.leaky_relu(cost_volume, negative_slope=self._leaky_relu_alpha)

            # Compute context and flow from previous flow, cost volume, and features1.
            if flow_up is None:
                x_in = torch.cat([cost_volume, features1], dim=1)
            else:
                if context_up is None:
                    x_in = torch.cat([flow_up, cost_volume, features1], dim=1)
                else:
                    x_in = torch.cat([context_up, flow_up, cost_volume, features1], dim=1)

            # Use dense-net connections.
            x_out = None
            if self._shared_flow_decoder:
                # reuse the same flow decoder on all levels
                flow_layers = self._flow_layers
            else:
                flow_layers = self._flow_layers[level]
            for layer in flow_layers[:-1]:
                x_out = layer(x_in)
                x_in = torch.cat([x_in, x_out], dim=1)
            context = x_out

            flow = flow_layers[-1](context)

            # dropout full layer
            if training and self._drop_out_rate:
                maybe_dropout = (torch.rand([]) > self._drop_out_rate).type(torch.get_default_dtype())
                # note that operation must not be inplace, otherwise autograd will fail pathetically
                context = context * maybe_dropout
                flow = flow * maybe_dropout

            if flow_up is not None and self._accumulate_flow:
                flow += flow_up

            # Upsample flow for the next lower level.
            flow_up = net_utils.upsample(flow, is_flow=True)
            if self._num_context_up_channels:
                context_up = self._context_up_layers[level](context)

            # Append results to list.
            flows.insert(0, flow)

        # Refine flow at level 1.
        # refinement = self._refine_model(torch.cat([context, flow], dim=1))
        refinement = torch.cat([context, flow], dim=1)
        for layer in self._refine_model:
            refinement = layer(refinement)

        # dropout refinement
        if training and self._drop_out_rate:
            maybe_dropout = (torch.rand([]) > self._drop_out_rate).type(torch.get_default_dtype())
            # note that operation must not be inplace, otherwise autograd will fail pathetically
            refinement = refinement * maybe_dropout

        refined_flow = flow + refinement
        flows.insert(0, refined_flow)
        flows[0] = net_utils.upsample(flows[0], is_flow=True)
        #flows[0] = refined_flow
        
        flows.insert(0, flows[0])
        flows[0] = net_utils.upsample(flows[0], is_flow=True)
        #flows.insert(0, net_utils.upsample(flows[0], is_flow=True))
        #flows.insert(0, net_utils.upsample(flows[0], is_flow=True))

        return flows

    def _build_cost_volume_surrogate_convs(self):
        layers = nn.ModuleList()
        for _ in range(self._num_levels):
            layers.append(nn.Sequential(
                nn.ZeroPad2d((2,1,2,1)), # should correspond to "SAME" in keras
                nn.Conv2d(
                    in_channels=int(2 * self._num_channels_upsampled_context * self._channel_multiplier),
                    out_channels=int(2 * self._num_channels_upsampled_context * self._channel_multiplier),
                    kernel_size=(4, 4)))
            )
        return layers

    def _build_upsample_layers(self, num_channels):
        """Build layers for upsampling via deconvolution."""
        layers = []
        for unused_level in range(self._num_levels):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=(4, 4),
                    stride=2,
                    padding=1))
        return nn.ModuleList(layers)

    def _build_flow_layers(self):
        """Build layers for flow estimation."""
        # Empty list of layers level 0 because flow is only estimated at levels > 0.
        result = nn.ModuleList([nn.ModuleList()])

        block_layers = [128, 128, 96, 64, self._num_context_up_channels]

        for i in range(1, self._num_levels):
            layers = nn.ModuleList()
            last_in_channels = (64+32) if not self._use_cost_volume else (81+32)
            if i != self._num_levels-1:
                last_in_channels += 2 + self._num_context_up_channels

            for c in block_layers:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels=last_in_channels,
                            out_channels=int(c * self._channel_multiplier),
                            kernel_size=(3, 3),
                            padding=1),
                        nn.LeakyReLU(
                            negative_slope=self._leaky_relu_alpha)
                    ))
                last_in_channels += int(c * self._channel_multiplier)
            layers.append(
                nn.Conv2d(
                    in_channels=block_layers[-1],
                    out_channels=2,
                    kernel_size=(3, 3),
                    padding=1))
            if self._shared_flow_decoder:
                return layers
            result.append(layers)
        return result

    def _build_refinement_model(self):
        """Build model for flow refinement using dilated convolutions."""
        layers = []
        last_in_channels = self._num_context_up_channels + 2
        for c, d in [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]:
            layers.append(
                nn.Conv2d(
                    in_channels=last_in_channels,
                    out_channels=int(c * self._channel_multiplier),
                    kernel_size=(3, 3),
                    stride=1,
                    padding=d,
                    dilation=d))
            layers.append(
                nn.LeakyReLU(negative_slope=self._leaky_relu_alpha))
            last_in_channels = int(c * self._channel_multiplier)
        layers.append(
            nn.Conv2d(
                in_channels=last_in_channels,
                out_channels=2,
                kernel_size=(3, 3),
                stride=1,
                padding=1))
        return nn.ModuleList(layers)

    def _build_1x1_shared_decoder(self):
        """Build layers for flow estimation."""
        # Empty list of layers level 0 because flow is only estimated at levels > 0.
        result = nn.ModuleList([nn.ModuleList()])
        for _ in range(1, self._num_levels):
            result.append(
                nn.Conv2d(
                    in_channels= 32,
                    out_channels=32,
                    kernel_size=(1, 1),
                    stride=1))
        return result

