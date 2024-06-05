import torch
from metnet3_pytorch import MetNet3


device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

metnet3 = MetNet3(
    dim = 512,
    num_lead_times = 722,
    lead_time_embed_dim = 32,
    input_spatial_size = 624,
    attn_dim_head = 8,
    hrrr_channels = 617,
    input_2496_channels = 2 + 14 + 1 + 2 + 20,
    input_4996_channels = 16 + 1,
    precipitation_target_bins = dict(
        mrms_rate = 512,
        mrms_accumulation = 512,
    ),
    surface_target_bins = dict(
        omo_temperature = 256,
        omo_dew_point = 256,
        omo_wind_speed = 256,
        omo_wind_component_x = 256,
        omo_wind_component_y = 256,
        omo_wind_direction = 180
    ),
    hrrr_loss_weight = 10,
    hrrr_norm_strategy = 'sync_batchnorm',  # this would use a sync batchnorm to normalize the input hrrr and target, without having to precalculate the mean and variance of the hrrr dataset per channel
    hrrr_norm_statistics = None             # you can also also set `hrrr_norm_strategy = "precalculated"` and pass in the mean and variance as shape `(2, 617)` through this keyword argument
).to(device)


# inputs

lead_times = torch.randint(0, 722, (1,)).to(device)
hrrr_input_2496 = torch.randn((1, 617, 624, 624)).to(device)
hrrr_stale_state = torch.randn((1, 1, 624, 624)).to(device)
input_2496 = torch.randn((1, 39, 624, 624)).to(device)
input_4996 = torch.randn((1, 17, 624, 624)).to(device)

metnet3.eval()

# Dict[str, Tensor], Tensor, Dict[str, Tensor]
surface_preds, hrrr_pred, precipitation_preds = metnet3(
    lead_times = lead_times,
    hrrr_input_2496 = hrrr_input_2496,
    hrrr_stale_state = hrrr_stale_state,
    input_2496 = input_2496,
    input_4996 = input_4996,
)

for name, surface_pred in surface_preds.items():
    print(f"{name}: {surface_pred.shape}")
print(f"hrrr_pred: {hrrr_pred.shape}")
for name, precipitation_pred in precipitation_preds.items():
    print(f"{name}: {precipitation_pred.shape}")
# omo_temperature: torch.Size([1, 256, 128, 128])
# omo_dew_point: torch.Size([1, 256, 128, 128])
# omo_wind_speed: torch.Size([1, 256, 128, 128])
# omo_wind_component_x: torch.Size([1, 256, 128, 128])
# omo_wind_component_y: torch.Size([1, 256, 128, 128])
# omo_wind_direction: torch.Size([1, 180, 128, 128])
# hrrr_pred: torch.Size([1, 617, 128, 128])
# mrms_rate: torch.Size([1, 512, 512, 512])
# mrms_accumulation: torch.Size([1, 512, 512, 512])
