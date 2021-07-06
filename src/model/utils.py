import torch
import math


def get_timing_signal(length: int, channels: int,
                      min_timescale: float = 1.0,
                      max_timescale: float = 1.0e4) -> torch.Tensor:
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(
        float(max_timescale) / float(min_timescale)) /
        (float(num_timescales)-1))
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales).type(torch.float32
                                          ) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    m = torch.nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal


def PosEncoder(x: torch.Tensor,
               min_timescale: float = 1.0,
               max_timescale: float = 1.0e4):
    length = x.size(1)
    channels = x.size(2)
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return x + (signal.cuda() if x.is_cuda else signal)


def TreePosEncoder(x):
    ...
