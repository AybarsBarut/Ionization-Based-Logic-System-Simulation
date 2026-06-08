# İyonlaşma Tabanlı Mantık Sistemi

## 1. Kısa model özeti

Bu çalışma, seyrek iyonlaşmış bir gazın darbeli elektrik alan altındaki ortalama
tepkisini eğitim amacıyla simüle eder. Model; elektron ve iyon yoğunluğu,
uyarılmış durum nüfusu, iletkenlik, optik emisyon ve gürültülü sensör çıktısını
zaman içinde üretir. Gelişmiş katman ayrıca negatif iyon, metastabil nüfus,
ortalama elektron enerjisi ve bir boyutlu uzaysal yayılımı çözer. Sensör çıktısı
eşiklenerek 0/1 değerine çevrilir.

Bu bir nükleer transmutasyon modeli değildir. Çekirdek tepkimesi, atom numarası
değişimi, yüksek gerilim donanımı veya laboratuvar kontrolü içermez.

### Mimari kapsam

Bu projedeki AND, OR ve XOR adları seri/paralel bağlı iki gaz tüpü devresini
ifade etmez. Ayrı tüp dalları, ortak balast direnci veya Kirchhoff düğüm
denklemleri yoktur. Fiziksel girişlerin toplam sürüşe katkısı sıfır, bir, iki
veya üç aktif giriş için farklı analog plazma/sensör seviyeleri üretir; mantık
adları bu seviyelerin kalibre edilmiş eşiklerle elde edilen doğruluk tablosuna
verilir.

Dolayısıyla seri AND devresinde voltaj bölüşümü veya paralel OR devresinde ortak
direnç geribeslemesi bu modelin hatalı uygulanmış bir parçası değil, henüz
tanımlanmamış ayrı bir devre-modelleme katmanıdır. Böyle bir genişletme için her
tüpün iç plazma durumu ile düğüm voltajları ve dal akımlarının birlikte
çözülmesi gerekir.

Modelde fiziksel bellek tamamen yok değildir. Elektron, iyon, uyarılma ve
metastabil durumları darbe bittiğinde sonlu rekombinasyon, duvar kaybı ve
de-uyarılma hızlarıyla söner. Ayrıca iyonlaşma etkinliği keskin bir `if`
geçişi değil lojistik fonksiyondur. Schmitt histerezisi ise fiziksel tüpün
maintenance-voltage eğrisi değil, sensör karar katmanındaki ek histerezistir.

### Deney sorusu

Belirli basınç, sıcaklık, indirgenmiş elektrik alan, darbe ve akış koşullarında
ortalama plazma tepkisi kararlı bir eşik çıktısı üretir mi; bu çıktı AND, OR,
NOT, NAND, NOR, XOR, yarım toplayıcı, tam toplayıcı ve durumlu mantık
davranışına dönüştürülebilir mi?

### Hipotez

1. `E/N` etkin iyonlaşma eşiğinin altında kalırsa elektron yoğunluğu ve sensör
   çıktısı düşük kalır.
2. Eşik civarında küçük alan/basınç değişimleri büyük çıktı değişimi üretir.
3. Eşiğin yeterince üstünde periyodik ve çevrimden çevrime kararlı bir yanıt
   oluşur.
4. Sıfır, bir ve iki aktif fiziksel giriş için ayrışan sensör seviyeleri, uygun
   eşikler kullanıldığında mantık tablolarına dönüştürülebilir.

### Değişkenler ve ölçüm planı

- Bağımsız değişkenler: gaz, basınç, sıcaklık, alan, voltaj, darbe frekansı,
  darbe genişliği, akış, elektrot aralığı, başlangıç iyonlaşması ve gürültü.
- Temel durum değişkenleri: normalize elektron yoğunluğu `x_e`, normalize iyon
  yoğunluğu `x_i`, uyarılmış durum kesri `x_*` ve algılayıcı durumu `y`.
- Gelişmiş durum değişkenleri: pozitif iyon `x_+`, negatif iyon `x_-`,
  metastabil nüfus `x_m` ve ortalama elektron enerjisi `epsilon_e`.
- Ölçümler: elektron/iyon yoğunluğu, iyonlaşma kesri, emisyon, iletkenlik,
  sensör seviyesi, eşik çıktısı, gecikme, kararlılık ve gürültü dayanıklılığı.
- Karşılaştırma: üç senaryo, eşik taraması, alan-basınç taraması, tek
  parametreli yüzde 10 duyarlılık analizi, 60 örnekli Monte Carlo analizi,
  BDF-RK45 karşılaştırması ve 1B reaksiyon-difüzyon çözümü.

## 2. Varsayımlar

1. Gaz uzaysal olarak homojendir; model sıfır boyutlu ortalama-alan modelidir.
2. Nötr yoğunluk ideal gaz bağıntısıyla hesaplanır:

   `N = p / (k_B T)`

3. Elektrik alan ile voltajdan hesaplanan `V/d` alanı, kullanıcı tarafından
   ayarlanabilen `b` katsayısıyla harmanlanır:

   `E_0 = (1-b) E_girdi + b (V/d)`

   Varsayılan `b=0.5` iki girdinin aynı fiziksel büyüklüğü iki kez toplamasını
   önler.

4. İndirgenmiş alan:

   `(E/N)_Td = E / (N * 10^-21)`

5. Elektron ve iyon yoğunlukları, `n_s = f_max N` ölçeğine bölünür.
   Varsayılan `f_max=2e-7`, seyrek iyonlaşmış rejim için sayısal bir üst
   ölçektir; gerçek bir doygunluk sabiti değildir.
6. Gelişmiş model ortalama elektron enerjisi, negatif iyon ve tek metastabil
   havuzu içerir; tam elektron enerji dağılımı ve ayrıntılı tür kimyası yine
   çözülmez.
7. Emisyon mutlak radyometrik birim yerine bağıl birimdir.
8. Temel model hızlı Euler çözümünü korur. Gelişmiş model adaptif BDF, Radau,
   RK45 veya LSODA çözücüsü kullanır.
9. 1B model öz-tutarlı elektrik alan çözmez; önceden tanımlı yumuşak bir alan
   profili ve sıfır akılı sınır koşulu kullanır.

## 3. Alt modeller ve denklemler

### 3.1 İyonlaşma

Alan etkinliği lojistik eşik fonksiyonudur:

`A(E/N) = 1 / (1 + exp(-((E/N)-E_th)/w))`

- `E_th`: gazın etkin iyonlaşma eşiği, Td.
- `w`: eşik geçiş genişliği, Td.
- `k_ion`: en büyük etkin iyonlaşma hızı, 1/s.
- `x_seed`: arka plan/başlangıç elektron kaynağı.
- `x_cap`: normalize yoğunluk tavanı.

İyonlaşma kaynağı:

`R_ion = k_ion A (x_e + x_seed) (1 - x_e/x_cap)`

Bu ifade Townsend çığının ayrıntılı kesit hesabı değildir; eşik üstünde
çoğalmayı ve sonlu taşıyıcı kapasitesini temsil eder.

### 3.2 Rekombinasyon ve kayıplar

`R_rec = k_rec x_e x_i`

`dx_e/dt = R_ion - R_rec - k_e,loss x_e`

`dx_i/dt = R_ion - R_rec - k_i,loss x_i`

- `k_rec`: etkin iki-cisim rekombinasyon katsayısının normalize biçimi, 1/s.
- `k_e,loss`: duvar, akış ve hava için elektron bağlanma kayıpları.
- `k_i,loss`: iyon duvar/akış kaybı.
- Akış katkısı: `k_flow = c_flow Q`; `Q` sccm cinsinden yalnızca bağıl
  yenilenme/kayıp göstergesidir. Geometri verilmediği için gerçek kalış süresi
  hesabı yapılmaz.

### 3.3 Uyarılma ve de-uyarılma

`dx_*/dt = k_exc A [x_e/(0.12+x_e)] (1-x_*) - k_deexc,eff x_*`

`k_deexc,eff = k_deexc [1 + q (p/p_atm)]`

- `k_exc`: elektron etkili uyarılma hızı.
- `k_deexc`: radyatif ve radyatif olmayan toplam de-uyarılma hızı.
- `q`: basınca bağlı söndürme kuvveti.

### 3.4 Optik emisyon

`I_em = f_rad k_deexc,eff x_* / 1000`

- `f_rad`: de-uyarmanın ışık üreten etkin kesri.
- Sonuç bağıl şiddet birimindedir (`a.u.`).

### 3.5 İletkenlik

`n_e = x_e f_max N`

`sigma = e n_e mu_e`

Elektron hareketliliği kaba olarak `mu_e ~ (p_atm/p) sqrt(T/300)` ile ölçeklenir
ve sayısal aşırılığı önlemek için üstten sınırlandırılır.

### 3.6 Sensör, eşik ve gürültü

İletkenlik ve emisyon doygun yanıtları:

`C = sigma/(sigma+sigma_ref)`

`L = I_em/(I_em+I_ref)`

Birleşik sensör:

`s = (1-w_opt) C + w_opt L`

Algılayıcının sonlu bant genişliği:

`dy/dt = (s-y)/tau_sensor`

Ölçüm:

`y_m = clip(y + Normal(0, noise_std), 0, 1)`

İkili çıktı:

`B = 1, y_m >= theta; aksi halde 0`

### 3.7 Kararlılık ve gürültü dayanıklılığı

Kararlılık, darbe içindeki doğal salınımı cezalandırmak yerine son sekiz
çevrimin ortalamalarını karşılaştırır:

`S = exp(-4 CV_cycle - 2 drift_cycle)`

`S` 0 ile 1 arasındadır. Rejim etiketi çevrim geçmişine göre `off`,
`transition`, `stable` veya `unstable` olur.

Gürültü dayanıklılığı, son yüzde 25'lik bölümde gürültülü ve gürültüsüz eşik
kararlarının uyuşma oranıdır.

### 3.8 Negatif iyon ve metastabil alt modeli

Elektron bağlanması ve alan destekli ayrılma:

`R_att = k_att (1 - 0.65 A_E) x_e`

`R_det = k_det A_E x_-`

İyon-iyon nötrleşmesi:

`R_ii = k_ii x_+ x_-`

Metastabil üretim ve kayıp:

`dx_m/dt = k_m A x_e/(0.10+x_e) (1-x_m) - k_m,loss x_m - R_step`

Basamaklı iyonlaşma:

`R_step = k_step x_m (x_e+x_seed) (1-(x_e+x_-)/x_cap)`

Bu alt model özellikle hava senaryosunda elektronların bir bölümünün negatif
iyon havuzuna aktarılmasını ve metastabillerin gecikmeli iyonlaşma kaynağı
olmasını temsil eder.

### 3.9 Ortalama elektron enerjisi

Tam elektron enerji dağılımı yerine tek ortalama enerji değişkeni kullanılır:

`d epsilon_e/dt = k_E (epsilon_target(E/N)-epsilon_e) - L_inelastic`

`epsilon_target = min(epsilon_max, epsilon_bg + g_E (E/N))`

Enerji etkinliği ikinci bir lojistik fonksiyondur. Toplam iyonlaşma etkinliği
alan ve enerji etkinliklerinin geometrik ortalamasıdır:

`A = sqrt(A_E A_epsilon)`

Bu yaklaşım iki sıcaklıklı modelin sade bir vekilidir; gaz sıcaklığı kullanıcı
girdisi, elektron enerjisi ise dinamik değişkendir.

### 3.10 Adaptif ODE çözümü

`simulate_advanced_plasma` SciPy `solve_ivp` üzerinden BDF, Radau, RK45 veya
LSODA kullanır. Varsayılan BDF, hızlı enerji gevşemesi ile daha yavaş iyon ve
metastabil kayıplarının oluşturduğu zaman ölçeği ayrımı için seçilmiştir.
Darbe kenarlarını çözebilmek için en büyük iç adım `dt_s` ile sınırlandırılır.

### 3.11 Belirsizlik ve katsayı kestirimi

Monte Carlo katmanı basınç, sıcaklık, alan, voltaj, darbe genişliği, akış ve
başlangıç iyonlaşmasına bağımsız log-normal belirsizlik uygular. Çıktı
ortalaması, standart sapma, yüzde 2.5/50/97.5 nicelikleri ve `P(B=1)` hesaplanır.

Katsayı kestirimi, `drive_scale`, `sensor_observed` ve isteğe bağlı `weight`
sütunlarını taşıyan CSV uyumlu gözlemlere karşı pozitif gaz katsayılarını
logaritmik uzayda `least_squares` ile uydurur. Mevcut örnek sentetik veriyle
yazılım doğrulamasıdır; deneysel kalibrasyon iddiası değildir.

### 3.12 Bir boyutlu reaksiyon-difüzyon modeli

Her tür için genel biçim:

`partial x_j/partial t = R_j(x,E/N) + D_j partial^2 x_j/partial z^2`

İkinci uzaysal türev merkezî sonlu farkla, sınırlar sıfır akı koşuluyla çözülür.
Açık adım kararlılığı `D_max dt/dx^2 <= 0.48` koşuluyla denetlenir. Model
elektron, pozitif iyon, negatif iyon ve metastabil difüzyonunu ayrı etkin
katsayılarla temsil eder.

## 4. Parametrelerin anlamı

| Parametre | Birim | Anlam |
|---|---:|---|
| `gas` | - | Argon, neon, helyum, hava veya özel gaz |
| `pressure_pa` | Pa | Nötr yoğunluğu ve çarpışma/söndürmeyi belirler |
| `temperature_k` | K | İdeal gaz yoğunluğunu ve hareketlilik ölçeğini etkiler |
| `electric_field_v_m` | V/m | Kullanıcı alan girdisi |
| `applied_voltage_v` | V | `V/d` alan bileşenini oluşturur |
| `pulse_frequency_hz` | Hz | Darbelerin tekrar hızı |
| `pulse_width_s` | s | Her darbenin açık kalma süresi |
| `gas_flow_sccm` | sccm | Etkin parçacık kayıp/yenilenme göstergesi |
| `electrode_gap_m` | m | Voltajdan alan hesabındaki mesafe |
| `measurement_noise_std` | 0-1 | Sensör tam ölçeğine göre Gauss gürültüsü |
| `initial_ionization_fraction` | - | Başlangıç elektron/nötr oranı |
| `binary_threshold` | 0-1 | Dijitalleştirme eşiği |
| `solver_method` | - | Gelişmiş modelde BDF, Radau, RK45 veya LSODA |
| `electron_energy_ev` | eV | Dinamik ortalama elektron enerji durumu |

Gaz katsayıları `plasma_model.py` içindeki `GAS_LIBRARY` tablosundadır.
Değerler gerçek bir cihazın kalibrasyonu değil, gazlar arasında beklenen nitel
farkları oluşturan etkin katsayılardır.

### Gazlara ait etkin katsayılar

| Gaz | `E_th` (Td) | `w` (Td) | `k_ion` (1/s) | `k_rec` (1/s) | `k_exc` (1/s) | `k_deexc` (1/s) | `f_rad` | `mu_ref` (m²/Vs) | `k_attach` (1/s) | `q` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Argon | 115 | 14 | 4500 | 360 | 3200 | 1900 | 0.72 | 0.085 | 0 | 0.80 |
| Neon | 150 | 18 | 4100 | 310 | 3800 | 2400 | 0.82 | 0.110 | 0 | 0.65 |
| Helyum | 185 | 22 | 3700 | 250 | 3500 | 2900 | 0.58 | 0.160 | 0 | 0.45 |
| Hava | 135 | 20 | 3700 | 690 | 2500 | 2100 | 0.38 | 0.055 | 260 | 1.50 |

### Ortak sayısal/model katsayıları

| Katsayı | Varsayılan | Fiziksel/model anlamı |
|---|---:|---|
| `field_voltage_blend` | 0.50 | Girilen alan ile `V/d` alanının harman oranı |
| `afterglow_field_fraction` | 0.025 | Darbeler arasındaki artık alan kesri |
| `max_ionization_fraction` | `2e-7` | Normalize yoğunluğun fiziksel ölçeği |
| `electron_wall_loss_rate_s` | 300 1/s | Etkin elektron duvar/difüzyon kaybı |
| `ion_wall_loss_rate_s` | 85 1/s | Etkin iyon duvar kaybı |
| `flow_loss_per_sccm_s` | 0.55 1/(s·sccm) | Akışın etkin kayıp katkısı |
| `seed_floor_normalized` | `1e-4` | Sayısal arka plan elektron kaynağı |
| `density_ceiling_normalized` | 1.5 | Lojistik yoğunluk tavanı |
| `conductivity_reference_s_m` | 0.020 S/m | İletkenlik sensörü yarı-doyum değeri |
| `emission_reference_au` | 0.30 a.u. | Optik sensör yarı-doyum değeri |
| `detector_time_constant_s` | 0.00075 s | Sensör düşük geçiren zaman sabiti |
| `optical_sensor_weight` | 0.48 | Birleşik sensörde optik kanal ağırlığı |

Gelişmiş modelin varsayılan enerji gevşeme hızı `5500 1/s`, arka plan elektron
enerjisi `0.45 eV`, alan-enerji kazancı `0.019 eV/Td` ve enerji üst sınırı
`10 eV` değerindedir.

Özel gaz için kullanıcı yeni bir `GasProperties` nesnesi oluşturup
`simulate_plasma(config, custom_gas=...)` çağrısına verebilir. Özel gaz
katsayılarının da yukarıdaki etkin katsayılarla aynı anlamda olması gerekir.

## 5. Python kodu ve çalıştırma

Ana dosyalar:

- `plasma_model.py`: denklemler, gaz kütüphanesi, mantık kalibrasyonu ve
  duyarlılık işlevleri.
- `advanced_models.py`: adaptif ODE, elektron enerjisi, negatif iyon,
  metastabil, Monte Carlo, katsayı kestirimi ve 1B reaksiyon-difüzyon.
- `run_experiments.py`: üç senaryo, taramalar, CSV ve grafik üretimi.
- `test_plasma_model.py`: sayısal ve mantıksal regresyon testleri.

Çalıştırma:

```powershell
python run_experiments.py --output-dir outputs
python -m unittest -v
```

CSV zaman serisi sütunları arasında `time_s`, `reduced_field_td`,
`ionization_fraction`, `electron_density_m3`, `ion_density_m3`,
`excitation_fraction`, `conductivity_s_m`, `emission_intensity_au`,
`sensor_measured`, `binary_output` ve `regime` bulunur.

Gelişmiş CSV ayrıca `negative_ion_density_m3`, `metastable_fraction`,
`electron_energy_ev`, `attachment_source_normalized_s` ve
`stepwise_ionization_source_normalized_s` sütunlarını içerir.

## 6. Örnek senaryolar ve beklenen yorum

### Giriş setleri

| Senaryo | Gaz | Basınç (Pa) | Sıcaklık (K) | Alan (V/m) | Voltaj (V) | Frekans (Hz) | Genişlik (s) | Akış (sccm) | Gürültü |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | Argon | 4000 | 300 | 80000 | 400 | 2000 | 0.00018 | 60 | 0.020 |
| S2 | Neon | 3500 | 310 | 220000 | 1100 | 2000 | 0.00022 | 40 | 0.020 |
| S3 | Hava | 5000 | 320 | 220000 | 1100 | 1500 | 0.00040 | 90 | 0.085 |

### S1: Eşik altı argon

Alan düşük tutulur. Hipoteze göre elektron çoğalması kayıpları yenemez; sensör
ortalaması ve ikili çıktı düşük kalmalıdır.

### S2: Kararlı neon

İndirgenmiş alan neon eşiğinin üstündedir ve görev çevrimi yüksektir. Elektron
yoğunluğu, emisyon ve iletkenlik yükselmeli; çevrim ortalamaları yerleşerek
kararlı bir yüksek çıktı vermelidir.

### S3: Gürültülü hava/eşik

Elektron bağlanması, daha yüksek rekombinasyon ve yüksek ölçüm gürültüsü
birlikte kullanılır. Darbe genişliği sensör yanıtını dijital eşiğin yakınına
getirecek şekilde seçilmiştir; ikili kararın gürültü dayanıklılığı S2'den düşük
olmalıdır.

Kesin sayısal sonuçlar `outputs/scenario_summary.csv` ve `outputs/summary.txt`
dosyalarına çalıştırma sırasında yazılır.

### Gerçek çalıştırma sonuçları

| Senaryo | Kuyruk sensör ort. | Dijital çıktı | Gecikme (ms) | Kararlılık | Gürültü dayanıklılığı | Son rejim |
|---|---:|---:|---:|---:|---:|---|
| S1 eşik altı argon | 0.000234 | 0 | yok | 0.9996 | 1.0000 | off |
| S2 kararlı neon | 0.747291 | 1 | 4.34 | 0.9993 | 1.0000 | stable |
| S3 gürültülü hava/eşik | 0.445599 | 0 | 5.74 | 0.9931 | 0.5399 | stable |

S3'ün kuyruk ortalaması 0.45 eşiğinin biraz altındadır. Başlangıç geçişinde
eşiği kısa süre aştığı için bir gecikme değeri vardır, ancak yerleşik ortalama
çıktısı 0'dır. Fiziksel çevrimi kararlı olmasına rağmen eşik marjı küçük olduğu
için ölçüm kararı gürültüye hassastır. Bu, fiziksel kararlılık ile dijital karar
kararlılığının aynı kavram olmadığını gösterir.

## 7. Mantık kapısı benzetimi

Fiziksel girişler aktif giriş sayısına göre darbe sürüşünü değiştirir:

`drive = drive_base + n_active drive_step`

Sıfır, bir ve iki aktif giriş için sensör yanıtları kalibre edilir.

- OR eşiği: sıfır ve bir giriş yanıtının ortası.
- AND eşiği: bir ve iki giriş yanıtının ortası.
- NAND/NOR: ilgili eşik kararının terslenmesi.
- XOR: OR eşiğinin üstünde ve AND eşiğinin altında kalan pencere.
- NOT: girişin sürüşü bastırdığı tamamlayıcı fiziksel giriş düzeni.
- Yarım toplayıcı: `SUM=XOR`, `CARRY=AND`.

Tam toplayıcı dört analog yanıt seviyesini sıfır, bir, iki ve üç aktif giriş
olarak sınıflandırır. `SUM` sınıflandırılan aktif giriş sayısının paritesi,
`CARRY_OUT` ise sayının iki veya daha büyük olmasıdır.

Çalıştırmada elde edilen kalibrasyon seviyeleri:

- Sıfır aktif giriş: `0.000008`
- Bir aktif giriş: `0.438535`
- İki aktif giriş: `0.675391`
- OR alt eşiği: `0.219272`
- AND üst eşiği: `0.556963`

| A | B | AND | OR | NAND | NOR | XOR | Yarım toplayıcı toplam | Taşıma |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| 1 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 |
| 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |

### Tam toplayıcı sonucu

| A | B | Carry in | Sensör | Sum | Carry out |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.001729 | 0 | 0 |
| 0 | 0 | 1 | 0.438535 | 1 | 0 |
| 0 | 1 | 0 | 0.438535 | 1 | 0 |
| 0 | 1 | 1 | 0.589095 | 0 | 1 |
| 1 | 0 | 0 | 0.438535 | 1 | 0 |
| 1 | 0 | 1 | 0.589095 | 0 | 1 |
| 1 | 1 | 0 | 0.589095 | 0 | 1 |
| 1 | 1 | 1 | 0.657447 | 1 | 1 |

Schmitt-benzeri eşikleyici `0.40` alt ve `0.50` üst eşiği arasında önceki
durumu korur. Gürültülü hava senaryosunda tek eşikli çıktının 594 geçişini
316'ya düşürerek 278 gereksiz geçişi bastırdı. SR latch, SET ve RESET
uyaranları arasında `Q` durumunu korur; eşzamanlı SET/RESET geçersiz olarak
işaretlenir.

## 8. Hipotez değerlendirmesi

1. S1'in `0.000234`, S2'nin `0.747291` sensör ortalaması üretmesi alan-eşik
   hipotezini destekledi.
2. Alan-basınç taraması, sabit alan için basınç artınca `E/N` ve yanıtın
   azaldığını; alan artınca yanıtın yükseldiğini gösterdi.
3. S2'nin `0.9993` kararlılık metriği periyodik kararlı rejim hipotezini
   destekledi.
4. AND, OR, NOT, NAND, NOR, XOR ve yarım toplayıcı tabloları ideal doğruluk
   tablolarıyla eşleşti.
5. Temel mantık koşulunda eşik 0.65'ten 0.70'e çıkarıldığında kuyruk çıktısı
   1'den 0'a geçti. Bu, dijitalleştirmenin eşik seçimine duyarlı olduğunu
   doğruladı.
6. Yüzde 10 tek-parametre taramasında en yüksek normalize duyarlılıklar darbe
   frekansı (`0.2748`) ve darbe genişliğinde (`0.2603`) görüldü. Bu sonuç,
   seçilen çalışma noktasında bir darbedeki enerji/zaman birikiminin alan
   genliğindeki küçük değişimden daha belirleyici olduğunu gösterir.
7. Gelişmiş hava modelinde tepe elektron yoğunluğu `9.41e16 m^-3`, tepe negatif
   iyon yoğunluğu `2.94e16 m^-3` ve tepe metastabil kesri `0.2675` oldu.
8. BDF ve RK45 çözümlerinin sensör ortalamaları `0.629519` ve `0.629511`
   çıktı; bağıl fark yaklaşık `-1.3e-5` oldu.
9. Yüzde 8 giriş belirsizliğine sahip 60 Monte Carlo örneğinde sensör için
   yüzde 95 merkezi aralık `0.3198-0.5266`, `P(B=1)=0.5333` bulundu.
10. Sentetik katsayı kestiriminde eşik `122.34 Td`, iyonlaşma hızı
    `4124.14 1/s`, rekombinasyon hızı `425.90 1/s` ve RMSE `0.000916` oldu.
11. 1B hava çözümünde tepe yoğunluk merkezde oluştu; merkez/kenar oranı
    `1.2649`, uzaysal düzgünlük metriği `0.9238` oldu.

Sonuç olarak hipotezler bu sade modelin kendi varsayımları içinde
desteklenmiştir. Bu destek deneysel doğrulama veya gerçek cihaz performansı
anlamına gelmez.

## 9. Sınırlamalar

- Katsayılar deneysel veriyle kalibre edilmemiştir; nicel tahmin yapılamaz.
- Ayrık gaz tüplerinden oluşan seri/paralel devre, ortak balast direnci ve
  Kirchhoff düğüm çözümü yoktur.
- Paschen yasasına dayalı kırılma voltajı ve ayrı ateşleme/söndürme voltajları
  uygulanmamıştır. Basınç ve sıcaklık etkisi bu sürümde `E/N`, ideal gaz
  yoğunluğu ve etkin gaz katsayıları üzerinden temsil edilir.
- 1B model yalnızca reaksiyon-difüzyon gösterimidir; elektrot kılıfı, filament,
  ark ve plazma dalgalarını çözmez.
- Negatif iyon tek toplu türdür; ayrıntılı moleküler hava kimyası yoktur.
- Alan ile plazma arasındaki öz-tutarlı Poisson/Maxwell geri beslemesi yoktur.
- Emisyon spektrumu yerine tek bağıl toplam şiddet kullanılır.
- Mantık eşikleri kalibrasyona bağlıdır ve farklı koşullarda yeniden
  belirlenmelidir.
- Çıktılar laboratuvar tasarımı, güvenlik sınırı veya donanım sürme komutu
  olarak kullanılmamalıdır.

## 10. Uygulanan geliştirmeler ve sonraki adımlar

Uygulananlar:

1. Adaptif BDF/Radau/RK45/LSODA çözümü.
2. Ortalama elektron enerji denklemi.
3. Negatif iyon ve metastabil durumları.
4. CSV uyumlu katsayı kestirim altyapısı ve sentetik doğrulama.
5. Monte Carlo belirsizlik yayılımı.
6. Tam toplayıcı, SR latch ve Schmitt-benzeri histerezis.
7. Bir boyutlu reaksiyon-difüzyon modeli.

Bilimsel olarak anlamlı sonraki adımlar:

1. Güvenli, yayımlanmış deney verisini ayrı bir veri kaynağı olarak ekleyip
   katsayıları gaz ve basınç aralığına göre yeniden kestirmek.
2. Tek ortalama enerji yerine ayrık elektron enerji dağılımı kullanmak.
3. Poisson denklemiyle öz-tutarlı alan ve uzay yükü eklemek.
4. Birden çok metastabil, negatif iyon ve moleküler kanal tanımlamak.
5. Parametreler arası korelasyonları ve Bayesçi art dağılımları modellemek.
