import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, random_split

from sbit.models import sbit_coord, FORWARD 
from sbit.bridge import bridge_matching_loss

HIDDEN_DIM = 1408  
BATCH_SIZE = 16    
MAX_EPOCHS = 130  
checkpoint_dir = 'models_checkpoints_S' 
log_path = os.path.join(checkpoint_dir, 'training_log_S')
device = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(checkpoint_dir, exist_ok=True)

encoder = model.to(device).eval() 

baseline_model = sbit_coord("SBIT-S", data_dim=HIDDEN_DIM).to(device)
sbit_model = sbit_coord("SBIT-S", data_dim=HIDDEN_DIM).to(device)

LAST_EPOCH = 0

# baseline_model.load_state_dict(torch.load(f"{checkpoint_dir}/jepa_baseline_epoch_{LAST_EPOCH}.pt", map_location=device))
# sbit_model.load_state_dict(torch.load(f"{checkpoint_dir}/sbit_epoch_{LAST_EPOCH}.pt", map_location=device))
# baseline_model.eval(); sbit_model.eval()


optimizer_bs = optim.AdamW(baseline_model.parameters(), lr=5e-5)
optimizer_sbit = optim.AdamW(sbit_model.parameters(), lr=5e-5)

scheduler_bs = optim.lr_scheduler.ReduceLROnPlateau(optimizer_bs, 'min', patience=10, factor=0.5)
scheduler_sbit = optim.lr_scheduler.ReduceLROnPlateau(optimizer_sbit, 'min', patience=10, factor=0.5)

criterion = nn.MSELoss()
train_len = int(0.8 * len(dataset))
val_len = int(0.1 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

def get_features(images):
    with torch.no_grad():
        img_rgb = images.expand(-1, 3, -1, -1)
        img_resized = F.interpolate(img_rgb, size=(224, 224))
        outputs = encoder(**{'pixel_values': img_resized})
        features = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(features, p=2, dim=1) * 7.0

history = []
print(f"Запуск обучения SBIT-S (Small) | Пространство: {HIDDEN_DIM}D")

for epoch in range(1, MAX_EPOCHS + 1):
    baseline_model.train()
    sbit_model.train()
    
    total_loss_bs = 0
    total_loss_sbit = 0

    for i, video in enumerate(train_loader):
        # f0 - кадр t, f1 - кадр t+10
        f0 = (video[:, 0].float() / 255.0).to(device)
        f1 = (video[:, 10].float() / 255.0).to(device)
        
        z0 = get_features(f0) 
        z1 = get_features(f1) 
        
        curr_batch_size = z0.shape[0]
        direction = torch.full((curr_batch_size,), FORWARD, device=device, dtype=torch.long)
        log_sigma = torch.zeros(curr_batch_size, device=device)
        y_label = torch.zeros(curr_batch_size, device=device, dtype=torch.long) 

        # (JEPA)
        optimizer_bs.zero_grad()
        t_zero = torch.zeros(curr_batch_size, device=device) 
        pred_bs = baseline_model(x_t=z0, tau=t_zero, direction=direction, log_sigma=log_sigma, y=y_label, anchor=z0)
        loss_bs = criterion(pred_bs, z1)
        loss_bs.backward()
        optimizer_bs.step()
        total_loss_bs += loss_bs.item()
        
        # sbit
        # optimizer_sbit.zero_grad()
        
        # t = torch.rand(curr_batch_size, device=device)
        # t_view = t.view(-1, 1)
        
        # #затухающее расписание шума моста
        # sigma_bridge = 0.1 
        # std_t = torch.sqrt(sigma_bridge * t * (1 - t)).view(-1, 1)
        
        # noise = torch.randn_like(z0)
        
        # # Промежуточная точка zt (смесь z0, z1 и шума)
        # zt = (1 - t_view) * z0 + t_view * z1 + std_t * noise
        # target_v = z1 - z0
        
        # # Модель предсказывает направление скорости
        # pred_sbit_v = sbit_model(x_t=zt, tau=t, direction=direction, log_sigma=log_sigma, y=y_label, anchor=z0)
        
        # loss_sbit = criterion(pred_sbit_v, target_v)
        
        # loss_sbit.backward()
        # optimizer_sbit.step()
        # total_loss_sbit += loss_sbit.item()

        optimizer_sbit.zero_grad()
        SIGMA = 0.32   # sqrt(0.1): та же дисперсия шума моста, что и раньше (sigma_bridge=0.1)
        loss_sbit = bridge_matching_loss(
            sbit_model, z0, z1, SIGMA,
            bidirectional=False,   # нужен только forward: z0(t) -> z1(t+10)
            use_anchor=True,
        )
        loss_sbit.backward()
        optimizer_sbit.step()
        total_loss_sbit += loss_sbit.item()

    avg_loss_bs = total_loss_bs / len(train_loader)
    avg_loss_sbit = total_loss_sbit / len(train_loader)
    
    scheduler_bs.step(avg_loss_bs)
    scheduler_sbit.step(avg_loss_sbit)

    history.append({
        'epoch': epoch,
        'baseline_mse': avg_loss_bs,
        'sbit_mse': avg_loss_sbit,
        'lr': optimizer_sbit.param_groups[0]['lr']
    })

    print(f"Epoch {epoch:03d} | BS MSE: {avg_loss_bs:.6f} | SBIT Velocity MSE: {avg_loss_sbit:.6f}")

    if epoch % 10 == 0 or epoch == 1:
        torch.save(baseline_model.state_dict(), f"{checkpoint_dir}/jepa_baseline_epoch_{epoch}.pt")
        torch.save(sbit_model.state_dict(), f"{checkpoint_dir}/sbit_epoch_{epoch}.pt")
        pd.DataFrame(history).to_csv(log_path, index=False)

print(f"Обучение завершено. Логи: {log_path}")
