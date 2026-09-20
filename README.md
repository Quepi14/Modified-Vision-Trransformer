# Adaptive Patch Size ViT + Structured Magnitude-Based Pruning

Implementasi kode untuk skripsi:

> **Modifikasi Arsitektur Vision Transformer Menggunakan Adaptive Patch Size yang Dioptimalkan melalui Structured Magnitude-Based Pruning untuk Klasifikasi Patologi Daun pada Kondisi Pencahayaan Dinamis**
> Nurul Wachdan Alaudin — NIM 231080200106 — Informatika, Universitas Muhammadiyah Sidoarjo (UMSIDA)

Repo ini berisi dua kontribusi utama yang diuji pada tugas klasifikasi penyakit daun (Plant Village Dataset):

1. **Adaptive Patch Size** — Vision Transformer dengan ukuran patch yang tidak seragam. Setiap region gambar dinilai kompleksitasnya lewat *Information Complexity Score* (ICS = varians piksel lokal + magnitude gradien Sobel), lalu diberi ukuran patch 8×8 (region kompleks/detail), 16×16 (sedang), atau 32×32 (region homogen) — bukan satu ukuran patch tetap seperti ViT standar.
2. **Structured Magnitude-Based Pruning** — pemangkasan seluruh neuron (bukan bobot individual) pada blok MLP Transformer berdasarkan norma-L1 bobot masuknya, dicari secara iteratif sambil fine-tuning, untuk menekan ukuran/komputasi model tanpa menjatuhkan akurasi lebih dari toleransi yang ditentukan.

Ketahanan model terhadap **kondisi pencahayaan dinamis** (perubahan brightness/contrast dan gamma) juga disimulasikan dan dievaluasi terpisah dari akurasi pada kondisi normal.

## Struktur proyek

```
adaptive_vit/                  # library generik, tidak terikat ke dataset tertentu
├── config.py                  # dataclass konfigurasi (DataConfig, ICSConfig, ModelConfig, TrainConfig, PruningConfig)
├── data.py                    # scan dataset, validasi (corrupt/duplikat/near-duplikat), split, Dataset class
├── preprocessing.py           # transform (resize, normalisasi, simulasi pencahayaan)
├── ics.py                     # Information Complexity Score & penentuan patch_size_map
├── patch_embedding.py         # ekstraksi patch adaptif + collate_fn untuk DataLoader
├── vit_model.py                # ModifiedVisionTransformer (adaptive) & BaselineVisionTransformer (fixed patch)
├── train.py                    # training loop generik
├── pruning.py                  # structured magnitude-based pruning + iterative search
├── evaluate.py                  # metrik klasifikasi, efisiensi (FLOPs), perbandingan baseline vs. optimized
└── visualize.py                 # semua figure (kurva training, hasil pruning, perbandingan model, peta patch adaptif)

examples/
├── plantvillage_taxonomy.py            # adapter khusus dataset Plant Village (spesies + kondisi -> 29 kelas)
├── plantvillage_pipeline_example.py    # skrip referensi urutan pemanggilan, end-to-end (dokumentasi, bukan untuk dijalankan mentah-mentah)
└── main.ipynb                          # notebook Colab siap-jalan, cell demi cell, memanggil seluruh pipeline di atas
```

`adaptive_vit/` murni generik: kalau dataset kamu sudah dalam bentuk `root/<nama_kelas>/*.jpg`, kamu bisa lewati `plantvillage_taxonomy.py` sama sekali dan langsung pakai `adaptive_vit.data.scan_imagefolder(root)`.

## Instalasi

```bash
git clone <url-repo-kamu>
cd <nama-repo>
pip install -r requirements.txt
```

(Opsional) supaya `import adaptive_vit` bisa dipanggil dari mana saja tanpa utak-atik `sys.path`, install sebagai package lokal:

```bash
pip install -e .
```

## Dataset

Studi ini memakai **Plant Village Dataset (Updated)** oleh `tushar5harma` di Kaggle — 9 spesies tanaman, 29 kelas total (kombinasi spesies × kondisi/penyakit).

```bash
# via Kaggle CLI (butuh kaggle.json terlebih dahulu)
kaggle datasets download -d tushar5harma/plant-village-dataset-updated
unzip plant-village-dataset-updated.zip -d plant-village-dataset-updated
```

Setelah diekstrak, `DataConfig.dataset_root` harus menunjuk ke folder yang di dalamnya langsung berisi folder per spesies.

## Cara pakai

### Opsi 1 — Google Colab (paling gampang)

1. Upload folder `adaptive_vit/` dan `examples/` ke Google Drive (atau clone repo ini langsung di Colab lewat `git clone`).
2. Buka `examples/main.ipynb` di Colab.
3. Jalankan cell 0a (mount Drive + `sys.path`) dan cell 0b (siapkan dataset), sesuaikan path-nya.
4. Jalankan sisanya berurutan dari atas ke bawah — setiap cell sudah diberi judul markdown (Cell 1 s.d. Cell 10) yang mengikuti alur: scan & validasi dataset → split 80:10:10 → hitung mean/std → fit parameter ICS → bangun DataLoader → training Modified ViT → training Baseline ViT (pembanding) → pruning + fine-tuning → evaluasi test set → generate figure untuk BAB IV.

### Opsi 2 — Python biasa (lokal / server)

Contoh minimal memakai library ini langsung (lihat `examples/plantvillage_pipeline_example.py` untuk versi lengkapnya, termasuk fit ICS dan evaluasi):

```python
import functools
import torch
from torch.utils.data import DataLoader

from adaptive_vit import (
    DataConfig, ICSConfig, ModelConfig, TrainConfig,
    ModifiedVisionTransformer, adaptive_collate_fn, train_model,
)
from adaptive_vit import data as avdata, preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_cfg = DataConfig(dataset_root="/path/ke/plant-village-dataset-updated")
ics_cfg = ICSConfig()
train_cfg = TrainConfig()

# PENTING: region_size collate_fn HARUS diikat ke ics_cfg.region_size lewat
# functools.partial. DataLoader hanya memanggil collate_fn dengan satu
# argumen (batch), jadi default bawaan adaptive_collate_fn (region_size=32)
# TIDAK otomatis mengikuti ICSConfig -- kalau kamu mengubah region_size di
# config tapi lupa mengikatnya di sini, ekstraksi patch akan korup diam-diam.
collate_fn = functools.partial(adaptive_collate_fn, region_size=ics_cfg.region_size)

samples, class_names = avdata.scan_imagefolder(data_cfg.dataset_root)
valid_samples, report = avdata.validate_dataset(
    samples, class_names, min_resolution=data_cfg.image_size,
    imbalance_tolerance=data_cfg.class_imbalance_tolerance,
    check_near_duplicates=True,  # opsional: deteksi gambar duplikat termasuk versi flip
)

model_cfg = ModelConfig(num_classes=len(class_names))
model = ModifiedVisionTransformer(model_cfg)
# ... bangun Dataset/DataLoader (lihat examples/plantvillage_pipeline_example.py
#     Cell 3-5 untuk fit mean/std dan alpha/beta/p30/p70 sebelum sampai di sini) ...
model, history = train_model(model, train_loader, val_loader, train_cfg, device)
```

Untuk dataset PlantVillage secara spesifik, ganti `avdata.scan_imagefolder(...)` dengan `plantvillage_taxonomy.scan_plantvillage_dataset(...)` — sisanya identik.

### Alur lengkap (10 tahap)

1. **Scan & validasi dataset** — deteksi file corrupt (dipisah dari file resolusi rendah), duplikat persis (MD5), duplikat mendekati/ter-flip (perceptual hash, opsional), lalu cek keseimbangan kelas.
2. **Stratified split** 80:10:10 (train/val/test), per kelas.
3. **Hitung mean/std** aktual dari subset train (bukan angka ImageNet default) untuk normalisasi.
4. **Fit parameter ICS** (`alpha`, `beta`, `p30`, `p70`) dari sampel subset validasi — parameter inilah yang menentukan ambang "kompleks vs. homogen" per region gambar.
5. **Bangun Dataset & DataLoader** — `ImageListDataset` mengembalikan `(image_tensor, patch_size_map, label)`; `adaptive_collate_fn` menyusun batch dengan token-token berukuran variabel.
6. **Latih Modified ViT** (adaptive patch size) — kontribusi utama.
7. **Latih Baseline ViT** (patch 16×16 tetap) — sebagai pembanding di tabel hasil.
8. **Structured Magnitude-Based Pruning** — pencarian rasio pruning secara iteratif + fine-tuning, berhenti begitu penurunan akurasi melewati `PruningConfig.accuracy_drop_tolerance`.
9. **Evaluasi di test set** — akurasi, efisiensi (estimasi FLOPs), dan akurasi di bawah simulasi pencahayaan dinamis, untuk baseline maupun model yang sudah di-pruning.
10. **Visualisasi** — kurva training, grafik pencarian rasio pruning, perbandingan baseline vs. model akhir, dan peta ukuran patch adaptif pada satu sampel gambar (menunjukkan mekanisme ICS secara visual).

## Konfigurasi penting

Semua parameter ada di `adaptive_vit/config.py` sebagai dataclass, jadi tidak ada angka ajaib yang tersembunyi di tengah kode:

| Config | Parameter kunci | Default |
|---|---|---|
| `DataConfig` | `image_size`, `split_ratios`, `region_size` | 224, (0.8, 0.1, 0.1), 32 |
| `ICSConfig` | `patch_size` (kecil/sedang/besar), `p_low`/`p_high` | (8, 16, 32), 30/70 (persentil) |
| `ModelConfig` | `num_classes` (**wajib diisi manual**), `embed_dim`, `num_layers` | — , 384, 12 |
| `TrainConfig` | `batch_size`, `learning_rate`, `num_epochs` | 32, 3e-4, 100 |
| `PruningConfig` | `initial_prune_ratio`, `accuracy_drop_tolerance` | 0.30, 0.02 (2%) |

## Catatan implementasi

- Validasi dataset (`data.validate_dataset`) memaksa `img.load()` (bukan cuma `Image.open()`) supaya file yang terpotong/rusak tertangkap saat validasi, bukan mendadak crash di tengah training.
- `ImageListDataset.__getitem__` punya fallback: kalau ada satu file yang tetap gagal dibaca saat training (misal rusak setelah validasi), sample tersebut diganti sample lain secara acak, bukan menghentikan seluruh training run.
- Deteksi near-duplicate/flip (`check_near_duplicates=True`) memakai perceptual hash (dHash) buatan sendiri (hanya PIL + numpy, tanpa dependency tambahan), dicek pada orientasi normal maupun hasil `mirror` horizontal — berguna karena dataset dari Kaggle sering punya sumber campuran yang berisiko menyisipkan gambar yang sama (atau flip-nya) ke split train dan test sekaligus.

## Sitasi dataset

Plant Village Dataset (Updated), oleh tushar5harma, Kaggle: https://www.kaggle.com/datasets/tushar5harma/plant-village-dataset-updated
