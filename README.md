# Fuzzy Set-Based Autonomous Navigation – Reproduction and Extension

Bulanık Mantık dersi dönem projesi. Referans makaledeki bulanık küme tabanlı rota
planlama yöntemi (FPPA) Python ile yeniden gerçekleştirilmiş, makaledeki Case Study 1
yeniden üretilmiş ve gürültülü engel algılama altında çalışan **belirsizliğe duyarlı
üyelik fonksiyonu** önerilip test edilmiştir.

## Report and Presentation

- Teknik rapor (Türkçe): [`report/report.pdf`](report/report.pdf)
- Sunum: [`presentation/presentation.pdf`](presentation/presentation.pdf)

## Student Information

- **Ad Soyad:** Abdullah Taha Üstünsoy
- **Ders:** Bulanık Mantık
- **Üniversite:** İstanbul Üniversitesi-Cerrahpaşa, Bilgisayar Mühendisliği

## Original Paper

E. A. Rastkhiz, H. Schwartz, and I. Lambadaris, "A fuzzy set-based methodology for
autonomous navigation," *Fuzzy Sets and Systems*, vol. 518, Art. no. 109485, 2025,
doi: [10.1016/j.fss.2025.109485](https://doi.org/10.1016/j.fss.2025.109485).

Makalenin yazarları kod yayımlamamıştır; bu repodaki tüm kod proje kapsamında yazılmıştır.

## Problem Description

Önceden planlanmış bir global rotayı takip eden bir robotun, yol üzerinde karşılaştığı
yerel engellerden rotadan çok sapmadan kaçması. Klasik Yapay Potansiyel Alan (APF)
yöntemi içbükey (U şeklinde) engellerde yerel minimuma takılır. Makale engelleri ve
rotayı bulanık kümelerle tek bir ortam haritasında birleştirir ve robotu bu haritanın
kontur çizgileri üzerinde ilerletir.

## Method

1. **Bulanık engel** (Denklem 2, 3, 6): Her engel küçük dairelere (atomik kümeler) bölünür.
   Her dairenin üyelik fonksiyonu `μ(χ) = tanh((χᵀχ)^-b)`; yarıçap = nesne boyutu λ +
   güvenlik mesafesi r. Hareketli engeller hız yönünde uzatılır.
2. **Bulanık rota** (Denklem 8–10): Ardışık waypoint'lerin arasına Gauss üyelik
   fonksiyonları yerleştirilir; rota boyunca bir "sırt" oluşur.
3. **Birleşim** (Denklem 7, 14): Tüm atomik engeller ve waypoint'ler tek bir Yager
   t-conorm ile birleştirilir → ortam haritası `μ_E(x)`.
4. **Kontur takibi** (Denklem 17, 18, 22): Robotun engel üyeliği μ_O'ya göre hedef kontur
   `μ_th = α(1−μ_O)^d_o` ve hız sınırı `v_max = v_m(1−μ_O)^d_v` belirlenir. Her adım bir
   normal (hedef kontura it/çek) ve bir teğet (kontur boyunca ilerle) bileşenden oluşur.

## Fuzzy System Architecture

```
 Engel ölçümleri ──► 1. Bulanık engel  μ_O ──┐
                                            ├──► 3. Yager birleşimi μ_E ──► 4. Kontur takibi ──► Δx
 Global rota ──────► 2. Bulanık rota   μ_P ──┘                                  ▲
                                                                               │
                     μ_O ──► hedef kontur μ_th, hız sınırı v_max ──────────────┘
```

| Dosya | İçerik |
|---|---|
| `src/fuzzy_env.py` | Üyelik fonksiyonları, Yager t-conorm, ortam haritası ve gradyanı |
| `src/fppa.py` | FPPA planlayıcısı (Denklem 17, 18, 22) |
| `src/apf.py` | Karşılaştırma için klasik APF (aynı `run()` arayüzü) |
| `src/scenarios.py` | Case Study 1 senaryosu |
| `src/metrics.py` | Performans ölçütleri |
| `src/noise.py` | Gürültülü algılama simülasyonu ve önerilen yöntem |

## Dataset

Gerçek bir veri seti kullanılmamaktadır. Makaledeki gibi tüm deneyler simülasyondur;
senaryo tanımları için bkz. [`data/README.md`](data/README.md).

## Installation

```bash
git clone https://github.com/AbdullahTahaUstunsoy/fuzzy-logic.git
cd fuzzy-logic
pip install -r requirements.txt
```

## Requirements

- Python 3.10+ (geliştirme: Python 3.13)
- numpy ≥ 1.24, matplotlib ≥ 3.7

## How to Run

```bash
python experiments/baseline/show_environment.py      # bulanık ortam haritaları
python experiments/baseline/run_fppa.py              # FPPA tek koşu (--alpha 0.99 ile makale değeri)
python experiments/baseline/case1.py                 # Case Study 1: FPPA vs klasik APF  (~10 s)
python experiments/proposed/noise_experiment.py      # önerilen yöntem deneyi            (~1-2 dk)
python experiments/proposed/noise_experiment.py --quick   # hızlı deneme (~10 s)
```

Tüm grafikler `results/figures/`, tablolar `results/tables/` altına yazılır.

## Reproduction Experiments

**Case Study 1** (makale Bölüm 4.1): düz rota [0,20] → [200,20], merkezi [70,0] ve
yarıçapı 30 m olan yay şeklinde içbükey engel (97 atomik küme).

| Yöntem | Hedefe ulaştı | Yol (m) | Engel yüzeyine min. mesafe (m) | RMS yaw rate (rad/s) | Adım süresi (ms) |
|---|---|---|---|---|---|
| FPPA, α = 0.90 | Evet (89 s) | 375 | 2.35 | 0.42 | 0.25 |
| FPPA, α = 0.99 (makale) | Hayır (döngü) | – | 2.27 | 1.14 | 0.21 |
| Klasik APF | Hayır (yerel minimum) | – | 3.19 | 0.04 | 0.02 |

- Makaledeki gibi FPPA içbükey engelden çıkıp rotaya dönüyor, klasik APF takılıyor
  (`results/figures/case1_paths.png`).
- Güvenlik mesafesi (1.25 m) korunuyor; adım başına hesaplama süresi makaledeki
  < 0.3 ms ile tutarlı.
- **Makaleden sapma:** α = 0.99 ile robot döngüye giriyor; tanh doygunluğu nedeniyle
  rota sırtının en yüksek üyeliği ≈ 0.955 olduğundan 0.99 konturu oluşmuyor. α = 0.90
  kullanıldı (`results/figures/case1_alpha.png`).

## Proposed Extension

**Motivasyon:** Makale bulanık temsilin algılama belirsizliğini modellediğini söyler,
ancak deneylerde engel konumları kusursuz bilinir.

**Gürültü modeli:** Her 50 ms'de her atomik engel merkezine bağımsız
`N(0, σ²)` gürültü eklenir; robot yalnızca bu ölçümleri görür.

**Önerilen yöntem – belirsizliğe duyarlı üyelik fonksiyonu:** bulanık dairenin
yarıçapı ölçüm belirsizliğiyle genişletilir:

```
makale :  λ + r
önerilen: λ + r + k·σ        (k = 0 → baseline)
```

**Deney:** σ ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0} m, k ∈ {0, 1, 2}, her koşul için 10 tekrar.
Baseline ve önerilen yöntem aynı rastgele tohumlarla (birebir aynı gürültüyle) çalışır;
ölçütler gerçek engele göre hesaplanır.

## Experimental Results

Tüm koşullarda başarı oranı %100'dür (hedefe ulaşma, çarpışma yok, güvenlik mesafesi
korunuyor). Yönteme göre değişen, sürüş kalitesi ve rotadan sapmadır
(`results/figures/noise_metrics.png`, `results/tables/noise_summary.csv`).

| Ölçüt (σ = 1.0 m) | Baseline (k=0) | Önerilen (k=2) | Değişim |
|---|---|---|---|
| RMS yaw rate (rad/s) | 4.55 ± 0.23 | 0.85 ± 0.03 | −81 % |
| Engel yüzeyine min. mesafe (m) | 2.47 | 7.98 | +5.5 m |
| Yol uzunluğu (m) | 379 | 347 | −8 % |
| Adım süresi (ms) | 0.21 | 0.22 | ≈ aynı |

| Ölçüt (σ = 2.0 m) | Baseline (k=0) | Önerilen (k=2) | Değişim |
|---|---|---|---|
| RMS yaw rate (rad/s) | 9.63 ± 0.64 | 0.70 ± 0.01 | −93 % |
| Engel yüzeyine min. mesafe (m) | 2.43 | 18.09 | +15.7 m |
| Yol uzunluğu (m) | 390 | 312 | −20 % |

- Baseline'da gürültü arttıkça direksiyon titremesi (yaw rate) neredeyse doğrusal artıyor.
- Önerilen yöntemde titreme σ'dan büyük ölçüde bağımsız kalıyor.
- Bedeli: robot engelden çok daha uzaktan geçiyor. Büyük σ'da genişleyen bulanık bölge
  çanağın ağzını kapatıyor; robot çanağa girmeden dışından dolaşıyor, bu yüzden yol kısalıyor
  (`results/figures/noise_paths.png`).

## Limitations

- Yalnızca Case Study 1 yeniden üretildi; Case Study 2–3'ün rota koordinatları makalede verilmiyor.
- Makaledeki karşılaştırma yöntemi IAPF yerine klasik APF kullanıldı.
- α parametresi makaledeki değerden farklı (0.90); nedeni Reproduction Experiments'ta açıklandı.
- Robot nokta kabul edildi; tek tekerlekli kinematik model ve araç dinamikleri simüle edilmedi.
- Gürültü modeli bağımsız Gauss; gerçek sensörlerde hata zamanla ilişkili olabilir.
  Önerilen yöntem σ'nın bilindiğini varsayar.
- Büyük k·σ değerleri temsili aşırı temkinli yapar; dar geçitlerde robot geçebileceği
  bir yolu kapalı görebilir.

## References

Bkz. [`references/README.md`](references/README.md).
