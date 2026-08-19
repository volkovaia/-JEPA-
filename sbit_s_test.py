import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import os
from sbit.models import sbit_coord, FORWARD
from torch.utils.data import DataLoader, random_split
from sbit.bridge import sample_sde

NUM_TEST = 50
SAMPLES_COUNT = 100 
HIDDEN_DIM = 1408  
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = 'models_checkpoints_S' 
output_vis = 'latent_scientific_final_Velocity'
os.makedirs(output_vis, exist_ok=True)

LAST_EPOCH = 130 
baseline_model = sbit_coord("SBIT-S", data_dim=HIDDEN_DIM).to(device)
sbit_model = sbit_coord("SBIT-S", data_dim=HIDDEN_DIM).to(device)

try:
    baseline_model.load_state_dict(torch.load(f"{checkpoint_dir}/jepa_baseline_epoch_{LAST_EPOCH}.pt", map_location=device))
    sbit_model.load_state_dict(torch.load(f"{checkpoint_dir}/sbit_epoch_{LAST_EPOCH}.pt", map_location=device))
    print(f"Чекпоинты SBIT-S (Эпоха {LAST_EPOCH}) успешно загружены")
except Exception as e:
    print(f"Ошибка загрузки: {e}")

baseline_model.eval(); sbit_model.eval()

def get_features(images):
    img_4d = images.reshape(-1, 1, 64, 64)
    img_rgb = img_4d.expand(-1, 3, -1, -1)
    img_resized = F.interpolate(img_rgb, size=(224, 224))
    outputs = model(**{'pixel_values': img_resized.to(device)})
    features = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(features, p=2, dim=1) * 7.0

all_stats = []
print(f"Запуск тестирования")
train_len = int(0.8 * len(dataset))
val_len = int(0.1 * len(dataset))
test_len = len(dataset) - train_len - val_len
train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

with torch.no_grad():
    for i in range(NUM_TEST):
        video = test_ds[i]
        f0 = (video[0:1].float() / 255.0).to(device)
        f_real_img = (video[10:11].float() / 255.0).to(device)
        
        z0 = get_features(f0) 
        z_real = get_features(f_real_img)
        
        # Общие тензоры управления
        dir_b = torch.full((SAMPLES_COUNT,), FORWARD, device=device, dtype=torch.long)
        y_b = torch.zeros(SAMPLES_COUNT, device=device, dtype=torch.long)
        log_s_b = torch.zeros(SAMPLES_COUNT, device=device)

        # 1. JEPA 
        z_jepa = baseline_model(x_t=z0, tau=torch.zeros(1, device=device), 
                                direction=dir_b[:1], log_sigma=log_s_b[:1], 
                                y=y_b[:1], anchor=z0)

        # 2. SBIT 
        SIGMA = 0.32   # то же значение, что и в обучении
        N_STEPS = 30
        z0_batch = z0.repeat(SAMPLES_COUNT, 1)
        z_sbit_final = sample_sde(
            sbit_model, z0_batch, SIGMA,
            direction=FORWARD, n_steps=N_STEPS, use_anchor=True,
        )
        
        
        jepa_err = torch.norm(z_jepa - z_real).item()
        sbit_dists = torch.norm(z_sbit_final - z_real, dim=1)
        oracle_err = sbit_dists.min().item()
        diversity = torch.cdist(z_sbit_final, z_sbit_final).mean().item()

        all_stats.append({
            'jepa_error': jepa_err,
            'sbit_oracle_error': oracle_err,
            'diversity': diversity
        })

        # PCA Визуализация
        sbit_np = z_sbit_final.cpu().numpy()
        real_np = z_real.detach().cpu().numpy()
        jepa_np = z_jepa.detach().cpu().numpy()
        all_pts = np.vstack([sbit_np, real_np, jepa_np])
        pca = PCA(n_components=2)
        pts_2d = pca.fit_transform(all_pts)
        
        fig, ax = plt.subplots(1, 3, figsize=(22, 6), gridspec_kw={'width_ratios': [1, 1, 3]})
        ax[0].imshow(f0.cpu().squeeze(), cmap='gray'); ax[0].set_title("Input (t=0)"); ax[0].axis('off')
        ax[1].imshow(f_real_img.cpu().squeeze(), cmap='gray'); ax[1].set_title("Target (t=10)"); ax[1].axis('off')
        
        # Облако SBIT
        ax[2].scatter(pts_2d[:SAMPLES_COUNT, 0], pts_2d[:SAMPLES_COUNT, 1], 
                    c='royalblue', alpha=0.3, label=f'SBIT Cloud (Div: {diversity:.2f})')
        # Истина
        ax[2].scatter(pts_2d[SAMPLES_COUNT, 0], pts_2d[SAMPLES_COUNT, 1], 
                    c='lime', marker='*', s=500, label='Truth', edgecolors='black', zorder=10)
        # JEPA
        ax[2].scatter(pts_2d[SAMPLES_COUNT+1, 0], pts_2d[SAMPLES_COUNT+1, 1], 
                    c='red', marker='X', s=300, label=f'JEPA (Err: {jepa_err:.2f})', edgecolors='black', zorder=11)
        
        ax[2].set_title(f"Video {i} | Velocity-based Prediction")
        ax[2].legend(); ax[2].grid(True, alpha=0.1)
        
        plt.tight_layout()
        plt.savefig(f"{output_vis}/plot_{i}.png")
        if i % 10 == 0: plt.show()
        else: plt.close()

df = pd.DataFrame(all_stats)
print("\n" + "="*50)
print(f"Итоговые результаты:")
print(f"Average JEPA Error:   {df['jepa_error'].mean():.4f}")
print(f"Average SBIT Oracle:  {df['sbit_oracle_error'].mean():.4f}")
print(f"SBIT Win Rate:        {(df['sbit_oracle_error'] < df['jepa_error']).mean()*100:.1f}%")
print(f"Average Diversity:    {df['diversity'].mean():.4f}")
print("="*50)
