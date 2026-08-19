import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from sbit.models import sbit_coord, FORWARD
from sbit.bridge import sample_sde


def sample_sde_chunked(model, z0, sigma, *, direction, n_steps, use_anchor,
                       samples_count, chunk_size, device):
    """Обёртка над sample_sde, которая считает SAMPLES_COUNT сэмплов чанками,
    а не одним большим батчем
    """
    outs = []
    remaining = samples_count
    while remaining > 0:
        cur = min(chunk_size, remaining)
        z0_chunk = z0.repeat(cur, 1) if z0.shape[0] == 1 else z0[:cur]
        z_out = sample_sde(model, z0_chunk, sigma, direction=direction,
                           n_steps=n_steps, use_anchor=use_anchor)
        outs.append(z_out)
        remaining -= cur
    return torch.cat(outs, dim=0)

HIDDEN_DIM = 1408
LAST_EPOCH = 130
checkpoint_dir = 'models_checkpoints_S'
output_dir = 'diagnostics_after_fix'
os.makedirs(output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

NUM_TEST = 50
SAMPLES_COUNT = 100
CHUNK_SIZE = 10    
                
N_STEPS = 30
TRAIN_SIGMA = 0.32          # sigma, использованная при обучении (для теста C по умолчанию)
EPS_SIGMA = 1e-3            # "почти без шума": sample_sde делает math.log(sigma) внутри и
                             # не принимает ровно 0.0 (log(0) не определён)
SIGMA_SWEEP = [EPS_SIGMA, 0.05, 0.1, 0.15, 0.2, 0.32, 0.5, 0.8]


encoder = model.to(device).eval()
baseline_model = sbit_coord("SBIT-S", data_dim=HIDDEN_DIM).to(device)
sbit_model = sbit_coord("SBIT-S", data_dim=HIDDEN_DIM).to(device)
baseline_model.load_state_dict(torch.load(f"{checkpoint_dir}/jepa_baseline_epoch_{LAST_EPOCH}.pt", map_location=device))
sbit_model.load_state_dict(torch.load(f"{checkpoint_dir}/sbit_epoch_{LAST_EPOCH}.pt", map_location=device))
baseline_model.eval(); sbit_model.eval()
print(f"Чекпоинты SBIT-S (эпоха {LAST_EPOCH}) успешно загружены")


def get_features(images):
    img_4d = images.reshape(-1, 1, 64, 64)
    img_rgb = img_4d.expand(-1, 3, -1, -1)
    img_resized = F.interpolate(img_rgb, size=(224, 224))
    with torch.no_grad():
        outputs = encoder(**{'pixel_values': img_resized.to(device)})
    features = outputs.last_hidden_state.mean(dim=1)
    return F.normalize(features, p=2, dim=1) * 7.0


train_len = int(0.8 * len(dataset))
val_len = int(0.1 * len(dataset))
test_len = len(dataset) - train_len - val_len
_, _, test_ds = random_split(dataset, [train_len, val_len, test_len])


def load_pair(i):
    video = test_ds[i]
    f0 = (video[0:1].float() / 255.0).to(device)
    f1 = (video[10:11].float() / 255.0).to(device)
    return get_features(f0), get_features(f1)


def test_sigma_sweep():
    print("\n" + "=" * 70)
    print("ТЕСТ A/B: sigma-sweep на инференсе (включая sigma≈0 -> bias без шума)")
    print("=" * 70)

    rows = []
    with torch.no_grad():
        for i in range(NUM_TEST):
            z0, z1 = load_pair(i)
            dir_b = torch.full((1,), FORWARD, device=device, dtype=torch.long)
            y_b = torch.zeros(1, device=device, dtype=torch.long)
            log_s_b = torch.zeros(1, device=device)
            z_jepa = baseline_model(x_t=z0, tau=torch.zeros(1, device=device),
                                     direction=dir_b, log_sigma=log_s_b, y=y_b, anchor=z0)
            jepa_err = torch.norm(z_jepa - z1).item()

            for sigma_test in SIGMA_SWEEP:
                z_sbit = sample_sde_chunked(sbit_model, z0, sigma_test,
                                            direction=FORWARD, n_steps=N_STEPS, use_anchor=True,
                                            samples_count=SAMPLES_COUNT, chunk_size=CHUNK_SIZE, device="cuda")
                dists = torch.norm(z_sbit - z1, dim=1)
                diversity = torch.cdist(z_sbit, z_sbit).mean().item()
                cloud_mean = z_sbit.mean(dim=0, keepdim=True)
                bias_vs_truth = torch.norm(cloud_mean - z1).item()
                bias_vs_jepa = torch.norm(cloud_mean - z_jepa).item()

                rows.append({
                    'video': i,
                    'sigma_test': sigma_test,
                    'jepa_error': jepa_err,
                    'sbit_oracle_error': dists.min().item(),
                    'sbit_mean_error': dists.mean().item(),
                    'diversity': diversity,
                    'cloud_mean_bias_vs_truth': bias_vs_truth,
                    'cloud_mean_bias_vs_jepa': bias_vs_jepa,
                    'win': float(dists.min().item() < jepa_err),
                })
            if device == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'sigma_sweep_raw.csv'), index=False)
    summary = df.groupby('sigma_test').agg(
        jepa_error=('jepa_error', 'mean'),
        sbit_oracle_error=('sbit_oracle_error', 'mean'),
        sbit_mean_error=('sbit_mean_error', 'mean'),
        diversity=('diversity', 'mean'),
        cloud_mean_bias_vs_truth=('cloud_mean_bias_vs_truth', 'mean'),
        cloud_mean_bias_vs_jepa=('cloud_mean_bias_vs_jepa', 'mean'),
        win=('win', 'mean'),
    )
    summary = summary.rename(columns={'win': 'win_rate'})
    summary['win_rate'] *= 100
    print(summary.to_string())
    summary.to_csv(os.path.join(output_dir, 'sigma_sweep_summary.csv'))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(summary.index, summary['diversity'], 'o-', color='royalblue')
    axes[0].set_xlabel('sigma (тест)'); axes[0].set_ylabel('Diversity')
    axes[0].set_title('Diversity vs test-time sigma'); axes[0].grid(alpha=0.3)

    axes[1].plot(summary.index, summary['sbit_oracle_error'], 'o-', color='seagreen', label='SBIT Oracle')
    axes[1].plot(summary.index, summary['sbit_mean_error'], 'o-', color='orange', label='SBIT Mean')
    axes[1].axhline(summary['jepa_error'].iloc[0], color='indianred', linestyle='--', label='JEPA (baseline)')
    axes[1].set_xlabel('sigma (тест)'); axes[1].set_ylabel('L2 error')
    axes[1].set_title('Error vs test-time sigma'); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(summary.index, summary['win_rate'], 'o-', color='purple')
    axes[2].axhline(50, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('sigma (тест)'); axes[2].set_ylabel('Win Rate, %')
    axes[2].set_title('Win Rate vs test-time sigma'); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sigma_sweep.png'), dpi=130)
    plt.close()
    print(f"\nГрафик: {output_dir}/sigma_sweep.png")

    print("\nДиагноз")
    best_sigma = summary['sbit_oracle_error'].idxmin()
    print(f"Лучшее (по oracle error) значение test-time sigma: {best_sigma} "
          f"(oracle_err={summary.loc[best_sigma, 'sbit_oracle_error']:.4f}, "
          f"win_rate={summary.loc[best_sigma, 'win_rate']:.1f}%)")
    if EPS_SIGMA in summary.index:
        zero_row = summary.loc[EPS_SIGMA]
        print(f"\nПри sigma≈0 (почти чистый ODE, без шума): error={zero_row['sbit_oracle_error']:.4f} "
              f"vs JEPA={zero_row['jepa_error']:.4f}, diversity={zero_row['diversity']:.4f}")
        if zero_row['sbit_oracle_error'] > 1.2 * zero_row['jepa_error']:
            print("-> Даже без шума SBIT хуже JEPA — есть систематический bias в самой")
            print("   выученной drift-функции (не только проблема шума/диверсности).")
            print("   Стоит потренировать дольше / проверить LR-расписание / попробовать Base.")
        else:
            print("-> Без шума SBIT сопоставим с JEPA — bias под контролем, всё дело в")
            print("   калибровке масштаба шума. Используйте sigma из строки выше на инференсе")
            print("   вместо TRAIN_SIGMA=0.32 — переобучение не требуется.")
    return df, summary


def test_stratify_by_difficulty(sigma_for_test=None):
    sigma_for_test = TRAIN_SIGMA if sigma_for_test is None else sigma_for_test
    print("\n" + "=" * 70)
    print(f"ТЕСТ C: где именно SBIT выигрывает — простые или сложные случаи? (sigma={sigma_for_test})")
    print("=" * 70)

    rows = []
    with torch.no_grad():
        for i in range(NUM_TEST):
            z0, z1 = load_pair(i)
            dir_b = torch.full((1,), FORWARD, device=device, dtype=torch.long)
            y_b = torch.zeros(1, device=device, dtype=torch.long)
            log_s_b = torch.zeros(1, device=device)
            z_jepa = baseline_model(x_t=z0, tau=torch.zeros(1, device=device),
                                     direction=dir_b, log_sigma=log_s_b, y=y_b, anchor=z0)
            jepa_err = torch.norm(z_jepa - z1).item()

            z_sbit = sample_sde_chunked(sbit_model, z0, sigma_for_test,
                                        direction=FORWARD, n_steps=N_STEPS, use_anchor=True,
                                        samples_count=SAMPLES_COUNT, chunk_size=CHUNK_SIZE, device="cuda")
            oracle_err = torch.norm(z_sbit - z1, dim=1).min().item()

            rows.append({'video': i, 'jepa_error': jepa_err, 'sbit_oracle_error': oracle_err,
                         'win': float(oracle_err < jepa_err)})
            if device == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df['difficulty_tertile'] = pd.qcut(df['jepa_error'], 3, labels=['easy', 'medium', 'hard'])
    summary = df.groupby('difficulty_tertile', observed=True).agg(
        jepa_error=('jepa_error', 'mean'),
        sbit_oracle_error=('sbit_oracle_error', 'mean'),
        win=('win', 'mean'),
        n=('video', 'count'),
    )
    summary = summary.rename(columns={'win': 'win_rate'})
    summary['win_rate'] *= 100
    print(summary.to_string())
    df.to_csv(os.path.join(output_dir, 'stratify_by_difficulty_raw.csv'), index=False)
    summary.to_csv(os.path.join(output_dir, 'stratify_by_difficulty_summary.csv'))

    print("\nДиагноз")
    if summary.loc['hard', 'win_rate'] > summary.loc['easy', 'win_rate']:
        print("Win Rate выше на 'сложных' случаях (где JEPA ошибается больше) — это хороший")
        print("знак: подтверждает исходную гипотезу — SBIT специально помогает там, где")
        print("реально есть неопределённость (после столкновений/бифуркаций движения),")
        print("а не на лёгких почти-детерминированных случаях.")
    else:
        print("Win Rate не выше на сложных случаях — SBIT пока не показывает ожидаемого")
        print("паттерна 'выигрываю именно там, где неопределённость выше'. Возможно,")
        print("облако плохо охватывает именно те случаи, где реально нужна многомодальность —")
        print("посмотрите на конкретные 'hard' видео глазами (PCA-плоты) для этой sigma.")
    return df, summary


if __name__ == "__main__":
    df_sweep, summary_sweep = test_sigma_sweep()
    df_strat, summary_strat = test_stratify_by_difficulty()
    print(f"\nВсе результаты сохранены в {output_dir}/")
