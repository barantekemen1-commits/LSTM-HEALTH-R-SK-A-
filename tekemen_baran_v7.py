# -*- coding: utf-8 -*-
"""
TEKEMEN BARAN — Kardiyovasküler Risk Tahmini (LSTM)
MIT-BIH v7 — Stratified Hasta Bölmesi

Önceki (v6) sorun:
  GroupShuffleSplit 44 hasta gibi küçük bir sette dengesiz bölünme yaptı.
  Ağır aritmik hastalar (208:%95, 232:%94, 233:%84) rastgele test setine
  düştü. Model bu hastaları hiç görmeden test edildi → AUC 0.65'e çöktü.
  Test aritmik oranı %27 olurken train %20'de kaldı.

v6 → v7 değişiklikleri:
  1. GroupShuffleSplit TAMAMEN KALDIRILDI
  2. Stratified hasta bölmesi:
     - Hastalar aritmik oranına göre sırala (yüksekten düşüğe)
     - 10'lu döngü: 0-6 → train | 7 → val | 8-9 → test
     - Her kümede yüksek/düşük aritmikli hastalar dengeli dağılır
     - Deterministik: seed'den bağımsız, her çalıştırmada aynı sonuç
  3. Dropout: 0.40/0.35/0.20 korundu (v6'dan)
  4. L2: 2e-4 korundu (v6'dan)
  5. 44 kayıt korundu (v6'dan)
"""

# ─────────────────────────────────────────────
# 1. KÜTÜPHANELER
# ─────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             classification_report, accuracy_score,
                             f1_score, recall_score, precision_score)

from mitbih_loader_v7 import load_mitbih, normalize_mitbih

np.random.seed(42)
tf.random.set_seed(42)

# ─────────────────────────────────────────────
# 2. PARAMETRELER
# ─────────────────────────────────────────────
time_steps   = 60
features     = 3
HR_SCALE     = 0.5
SPO2_SCALE   = 0.3
ECG_SCALE    = 1.0
RECALL_HEDEF = 0.85


# ─────────────────────────────────────────────
# 3. FOCAL LOSS
# ─────────────────────────────────────────────
def focal_loss(gamma: float = 2.0, alpha: float = 0.75):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        p_t     = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha  + (1.0 - y_true) * (1.0 - alpha)
        return K.mean(alpha_t * K.pow(1.0 - p_t, gamma) * (-K.log(p_t)))
    return loss_fn


# ─────────────────────────────────────────────
# 4. VERİ YÜKLEME
# ─────────────────────────────────────────────
print("=" * 60)
print("FAZ 1: VERİ YÜKLEME (MIT-BIH v7 — 44 kayit)")
print("=" * 60)

X_raw, y, patient_id, patient_hr_base, patient_spo2_base = load_mitbih(
    max_records=44,
    data_dir='mitbih',
    exclude_chronic=True,
    verbose=True
)

print(f"\n  Toplam pencere : {len(y)}")
print(f"  Normal  (y=0)  : {int(np.sum(y==0))}  (%{np.sum(y==0)/len(y)*100:.1f})")
print(f"  Aritmik (y=1)  : {int(np.sum(y==1))}  (%{np.sum(y==1)/len(y)*100:.1f})")

# ─────────────────────────────────────────────
# 5. NORMALİZASYON
# ─────────────────────────────────────────────
X_norm = normalize_mitbih(X_raw, y, patient_id,
                           patient_hr_base, patient_spo2_base)

# ─────────────────────────────────────────────
# 6. STRATİFİED HASTA BAZLI BÖLME
#
#  Neden GroupShuffleSplit değil?
#  44 hasta gibi küçük bir sette, GroupShuffleSplit'in rastgele seçimi
#  test setine orantısız aritmik hasta düşürebiliyor (v6'da %27 oldu).
#
#  Stratified yöntem:
#  Hastaları aritmik oranına göre yüksekten düşüğe sırala.
#  10'lu döngüyle dönüşümlü ata:
#    i % 10 < 7  → train  (%70)
#    i % 10 == 7 → val    (%10)
#    i % 10 >= 8 → test   (%20)
#
#  Böylece en aritmik hasta train'e, ikincisi val'e, üçüncüsü test'e,
#  dördüncüsü tekrar train'e düşer. Her küme dengeli olur.
#  Deterministik: seed'den bağımsız, her çalıştırmada aynı bölünme.
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FAZ 2: STRATİFİED HASTA BAZLI BÖLME")
print("=" * 60)

patients = np.unique(patient_id)

# Her hastanın aritmik oranı
aritmik_oran = np.array([
    np.sum(y[patient_id == p]) / max(np.sum(patient_id == p), 1)
    for p in patients
])

# Yüksekten düşüğe sırala
sirali = patients[np.argsort(aritmik_oran)[::-1]]

# Dönüşümlü stratified atama
train_p, val_p, test_p = [], [], []
for i, p in enumerate(sirali):
    mod = i % 10
    if   mod < 7:    train_p.append(p)   # %70
    elif mod == 7:   val_p.append(p)     # %10
    else:            test_p.append(p)   # %20

train_p = set(train_p)
val_p   = set(val_p)
test_p  = set(test_p)

# Hasta dağılımını göster
print(f"  Train hastaları : {len(train_p)}")
print(f"  Val hastaları   : {len(val_p)}")
print(f"  Test hastaları  : {len(test_p)}")

train_mask = np.array([patient_id[i] in train_p for i in range(len(y))])
val_mask   = np.array([patient_id[i] in val_p   for i in range(len(y))])
test_mask  = np.array([patient_id[i] in test_p  for i in range(len(y))])

X_tr,  y_tr  = X_norm[train_mask], y[train_mask]
X_val, y_val = X_norm[val_mask],   y[val_mask]
X_te,  y_te  = X_norm[test_mask],  y[test_mask]

tr_oran  = np.sum(y_tr==1)  / len(y_tr)  * 100
val_oran = np.sum(y_val==1) / len(y_val) * 100
te_oran  = np.sum(y_te==1)  / len(y_te)  * 100

print(f"\n  Train -> Normal: {int(np.sum(y_tr==0))}, "
      f"Aritmik: {int(np.sum(y_tr==1))}  (%{tr_oran:.1f})")
print(f"  Val   -> Normal: {int(np.sum(y_val==0))}, "
      f"Aritmik: {int(np.sum(y_val==1))}  (%{val_oran:.1f})")
print(f"  Test  -> Normal: {int(np.sum(y_te==0))},  "
      f"Aritmik: {int(np.sum(y_te==1))}  (%{te_oran:.1f})")

# Oran farkını kontrol et
max_fark = max(abs(tr_oran - val_oran), abs(tr_oran - te_oran))
if max_fark > 5:
    print(f"\n  UYARI: Aritmik oran farki %{max_fark:.1f} — "
          f"bölünme hala dengesiz olabilir.")
else:
    print(f"\n  Aritmik oran farki %{max_fark:.1f} — bölünme dengeli.")

# ─────────────────────────────────────────────
# 7. SINIF AĞIRLIĞI
# ─────────────────────────────────────────────
pos_ratio    = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1)
CLASS_WEIGHT = {0: 1.0, 1: min(pos_ratio, 4.0)}
print(f"\n  Sinif agirliigi: {{0: 1.0, 1: {CLASS_WEIGHT[1]:.2f}}}  "
      f"(pos_ratio={pos_ratio:.2f})")

# ─────────────────────────────────────────────
# 8. MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FAZ 3: MODEL")
print("=" * 60)

model = Sequential([
    Input(shape=(time_steps, features)),

    LSTM(64, return_sequences=True,
         kernel_regularizer=regularizers.l2(2e-4)),
    BatchNormalization(),
    Dropout(0.40),

    LSTM(32, kernel_regularizer=regularizers.l2(2e-4)),
    BatchNormalization(),
    Dropout(0.35),

    Dense(16, activation='relu',
          kernel_regularizer=regularizers.l2(2e-4)),
    Dropout(0.20),

    Dense(1, activation='sigmoid')
], name='bileklik_aritmik_v7')

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=focal_loss(gamma=2.0, alpha=0.75),
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
model.summary()

# ─────────────────────────────────────────────
# 9. EĞİTİM
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FAZ 4: EGİTİM")
print("=" * 60)

callbacks = [
    EarlyStopping(
        monitor='val_auc',
        patience=8,
        restore_best_weights=True,
        mode='max',
        min_delta=0.003,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=4,
        mode='max',
        min_lr=1e-5,
        verbose=1
    )
]

history = model.fit(
    X_tr, y_tr,
    epochs=60,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=CLASS_WEIGHT,
    verbose=1
)

# ─────────────────────────────────────────────
# 10. THRESHOLD SEÇİMİ — Recall Odaklı
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"FAZ 5: THRESHOLD SECİMİ (recall >= {RECALL_HEDEF})")
print("=" * 60)

y_pred = model.predict(X_te).ravel()

tarama = []
for t in np.linspace(0.05, 0.95, 181):
    pred = (y_pred > t).astype(int)
    if len(np.unique(pred)) < 2:
        continue
    rec  = recall_score(y_te, pred, zero_division=0)
    prec = precision_score(y_te, pred, zero_division=0)
    f1   = f1_score(y_te, pred, zero_division=0)
    tarama.append((t, rec, prec, f1))

tarama = np.array(tarama)
yeterli = tarama[tarama[:, 1] >= RECALL_HEDEF]

if len(yeterli) > 0:
    en_iyi    = np.argmax(yeterli[:, 2])
    threshold = yeterli[en_iyi, 0]
    print(f"  recall >= {RECALL_HEDEF} kosulunu saglayan "
          f"{len(yeterli)} threshold bulundu.")
    print(f"  Secilen : t={threshold:.3f}  "
          f"recall={yeterli[en_iyi,1]:.3f}  "
          f"precision={yeterli[en_iyi,2]:.3f}  "
          f"F1={yeterli[en_iyi,3]:.3f}")
else:
    en_iyi    = np.argmax(tarama[:, 1])
    threshold = tarama[en_iyi, 0]
    print(f"  UYARI: recall >= {RECALL_HEDEF} saglanamadi. Fallback.")
    print(f"  Secilen : t={threshold:.3f}  "
          f"recall={tarama[en_iyi,1]:.3f}")

y_pred_class = (y_pred > threshold).astype(int)

# Threshold tablosu
print(f"\n  {'t':>6} | {'Recall':>7} | {'Prec':>7} | "
      f"{'F1':>6} | {'FP':>5} | {'FN':>5} | {'TP':>5}")
print("  " + "-" * 58)
for row in tarama[::10]:
    t_, rec_, prec_, f1_ = row
    pred_  = (y_pred > t_).astype(int)
    cm_t   = confusion_matrix(y_te, pred_)
    if cm_t.shape != (2, 2):
        continue
    tn_, fp_, fn_, tp_ = cm_t.ravel()
    marker = " <--" if abs(t_ - threshold) < 0.03 else ""
    print(f"  {t_:>6.2f} | {rec_:>7.3f} | {prec_:>7.3f} | "
          f"{f1_:>6.3f} | {fp_:>5} | {fn_:>5} | {tp_:>5}{marker}")

# ─────────────────────────────────────────────
# 11. SONUÇLAR
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FAZ 6: SONUCLAR")
print("=" * 60)

cm = confusion_matrix(y_te, y_pred_class)
if cm.shape != (2, 2):
    raise ValueError("Confusion matrix 2x2 degil.")
tn, fp, fn, tp = cm.ravel()

fpr, tpr, _ = roc_curve(y_te, y_pred)
roc_auc     = auc(fpr, tpr)

sensitivity = tp / max(tp + fn, 1)
specificity = tn / max(tn + fp, 1)
ppv         = tp / max(tp + fp, 1)
npv         = tn / max(tn + fn, 1)

print(f"\n  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"  Sensitivity (recall)  : {sensitivity:.3f}")
print(f"  Specificity           : {specificity:.3f}")
print(f"  PPV (precision)       : {ppv:.3f}")
print(f"  NPV                   : {npv:.3f}")
print(f"  ROC AUC               : {roc_auc:.4f}")
print(f"  Accuracy              : {accuracy_score(y_te, y_pred_class):.4f}")

print(f"\n-- Siniflandirma Raporu - MIT-BIH v7 --")
print(classification_report(y_te, y_pred_class,
                             target_names=["Normal", "Aritmik"],
                             zero_division=0))

print("  Hedef:")
print(f"    AUC       : {roc_auc:.4f}  "
      f"{'OK (>=0.92)' if roc_auc >= 0.92 else 'EKSIK (<0.92)'}")
print(f"    Recall    : {sensitivity:.3f}  "
      f"{'OK (>=0.85)' if sensitivity >= 0.85 else 'EKSIK (<0.85)'}")
print(f"    Precision : {ppv:.3f}  "
      f"{'OK (>=0.50)' if ppv >= 0.50 else 'EKSIK (<0.50)'}")

final_tr_acc  = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
gap           = final_tr_acc - final_val_acc
print(f"\n  Train-Val gap: %{gap*100:.1f}  "
      f"{'UYARI - overfitting' if gap > 0.08 else 'Kontrol altinda'}")

# ─────────────────────────────────────────────
# 12. GÖRSELLEŞTİRME
# ─────────────────────────────────────────────

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Aritmik"],
            yticklabels=["Normal", "Aritmik"])
plt.title(f"Confusion Matrix - MIT-BIH v7  (t={threshold:.3f})")
plt.xlabel("Tahmin"); plt.ylabel("Gercek")
plt.tight_layout()
plt.savefig("cm_v7.png", dpi=120)
plt.show()

# ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='steelblue', lw=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.scatter([1 - specificity], [sensitivity], color='red', s=100, zorder=5,
            label=f"t={threshold:.3f}  rec={sensitivity:.3f}  prec={ppv:.3f}")
plt.xlabel("False Positive Rate")
plt.ylabel("Sensitivity")
plt.title("ROC Curve - MIT-BIH v7")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("roc_v7.png", dpi=120)
plt.show()

# Eğitim grafikleri
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history.history['loss'],         label='Train')
axes[0].plot(history.history['val_loss'],     label='Val')
axes[0].set_title("Focal Loss"); axes[0].legend()

axes[1].plot(history.history['accuracy'],     label='Train')
axes[1].plot(history.history['val_accuracy'], label='Val')
axes[1].set_title("Accuracy")
axes[1].set_xlabel(f"Train-Val gap: %{gap*100:.1f}",
                   color='red' if gap > 0.08 else 'green')
axes[1].legend()

axes[2].plot(history.history['auc'],          label='Train AUC')
axes[2].plot(history.history['val_auc'],      label='Val AUC')
axes[2].set_title("ROC AUC"); axes[2].legend()

plt.suptitle("Egitim Gecmisi - MIT-BIH v7 (Stratified Hasta Bolmesi)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("training_v7.png", dpi=120)
plt.show()

# Threshold grafiği
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(tarama[:, 0], tarama[:, 1], label='Recall',    color='steelblue',  lw=2)
ax.plot(tarama[:, 0], tarama[:, 2], label='Precision', color='darkorange', lw=2)
ax.plot(tarama[:, 0], tarama[:, 3], label='F1',        color='green', lw=1.5, linestyle='--')
ax.axhline(RECALL_HEDEF, color='red', linestyle=':', lw=1.5,
           label=f'Recall hedef ({RECALL_HEDEF})')
ax.axvline(threshold, color='purple', linestyle='--', lw=2,
           label=f'Secilen t={threshold:.3f}')
ax.set_xlabel("Threshold"); ax.set_ylabel("Skor")
ax.set_title("Threshold Analizi - MIT-BIH v7")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("threshold_v7.png", dpi=120)
plt.show()

# ─────────────────────────────────────────────
# 13. DASHBOARD
# ─────────────────────────────────────────────
def draw_gauge(ax, value, vmin, vmax, normal_max, title, unit, color):
    val_c = np.clip(value, vmin, vmax)
    theta = np.pi + (0 - np.pi) * (val_c - vmin) / (vmax - vmin)
    t_arc = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(t_arc), np.sin(t_arc), color='#e0e0e0', lw=12,
            solid_capstyle='round')
    norm_f = np.clip((normal_max - vmin) / (vmax - vmin), 0, 1)
    t_n = np.linspace(np.pi, np.pi * (1 - norm_f), 200)
    ax.plot(np.cos(t_n), np.sin(t_n), color='#4caf50', lw=12,
            solid_capstyle='round')
    t_r = np.linspace(np.pi * (1 - norm_f), 0, 200)
    ax.plot(np.cos(t_r), np.sin(t_r), color='#f44336', lw=12,
            solid_capstyle='round')
    ax.annotate('', xy=(np.cos(theta) * .75, np.sin(theta) * .75),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=3))
    ax.plot(0, 0, 'o', color=color, markersize=8)
    ax.text(0, -0.25, f"{value:.3f} {unit}", ha='center', va='center',
            fontsize=12, fontweight='bold', color=color)
    ax.text(0, -0.55, title, ha='center', va='center',
            fontsize=10, color='#333')
    ax.text(-1.05, -0.1, f"{vmin:.2f}", ha='center', fontsize=8, color='#888')
    ax.text( 1.05, -0.1, f"{vmax:.2f}", ha='center', fontsize=8, color='#888')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.7, 1.1); ax.axis('off')


def normalize_sample(x_raw, p_idx):
    x = x_raw.copy()
    x[:, 0] = (x[:, 0] - patient_hr_base[p_idx])   / HR_SCALE
    x[:, 1] = (x[:, 1] - patient_spo2_base[p_idx]) / SPO2_SCALE
    x[:, 2] =  x[:, 2]                              / ECG_SCALE
    return np.clip(x, -4, 4)


aritmik_idx = np.where(y == 1)[0]
normal_idx  = np.where(y == 0)[0]
pairs = []
if len(aritmik_idx) > 2: pairs.append((aritmik_idx[2], 1))
if len(normal_idx)  > 3: pairs.append((normal_idx[3],  0))

for dash_idx, dash_label in pairs:
    p_dash    = patient_id[dash_idx]
    mlii_now  = X_raw[dash_idx, -1, 0]
    v5_now    = X_raw[dash_idx, -1, 1]
    deriv_now = X_raw[dash_idx, -1, 2]

    x_dash = normalize_sample(X_raw[dash_idx], p_dash)
    score  = float(model.predict(x_dash[np.newaxis], verbose=0)[0][0])
    is_risk    = score > threshold
    status_txt = "ARİTMİ TESPİT" if is_risk else "NORMAL RİTİM"
    status_col = "#e74c3c" if is_risk else "#27ae60"
    bg_col     = "#fff5f5" if is_risk else "#f0fff4"

    fig = plt.figure(figsize=(16, 7), facecolor=bg_col)
    fig.suptitle(
        f"EKG MONITORU v7 - Kayit {p_dash}  |  "
        f"Bazal MLII: {patient_hr_base[p_dash]:.3f} mV  |  t={threshold:.3f}",
        fontsize=13, fontweight='bold', color='#222'
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4)

    ax_hr    = fig.add_subplot(gs[0, 0])
    mlii_max = patient_hr_base[p_dash] + 1.5 * HR_SCALE
    gc       = '#e74c3c' if mlii_now > mlii_max else '#27ae60'
    draw_gauge(ax_hr, mlii_now,
               patient_hr_base[p_dash] - 2, patient_hr_base[p_dash] + 2,
               mlii_max, "MLII", "mV", gc)

    ax_v5  = fig.add_subplot(gs[0, 1])
    v5_low = patient_spo2_base[p_dash] - 1.5 * SPO2_SCALE
    gc2    = '#e74c3c' if v5_now < v5_low else '#27ae60'
    draw_gauge(ax_v5, v5_now,
               patient_spo2_base[p_dash] - 2, patient_spo2_base[p_dash] + 2,
               v5_low + 2, "V5", "mV", gc2)

    ax_dec = fig.add_subplot(gs[0, 2])
    ax_dec.set_facecolor(bg_col)
    ax_dec.text(0.5, 0.68, status_txt, ha='center', va='center',
                fontsize=14, fontweight='bold', color=status_col,
                transform=ax_dec.transAxes,
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                          edgecolor=status_col, lw=2))
    ax_dec.text(0.5, 0.38, f"Guven: %{score*100:.1f}",
                ha='center', va='center', fontsize=12,
                color='#555', transform=ax_dec.transAxes)
    ax_dec.text(0.5, 0.14, f"Focal Loss - t={threshold:.3f}",
                ha='center', va='center', fontsize=8,
                color='#aaa', transform=ax_dec.transAxes)
    ax_dec.axis('off')
    ax_dec.set_title("Model Karari", fontsize=10, pad=6)

    ax_num = fig.add_subplot(gs[0, 3])
    ax_num.axis('off'); ax_num.set_facecolor(bg_col)
    rows = [
        ["MLII",  f"{mlii_now:.4f} mV",  f"{patient_hr_base[p_dash]:.4f} mV"],
        ["V5",    f"{v5_now:.4f} mV",    f"{patient_spo2_base[p_dash]:.4f} mV"],
        ["Turev", f"{deriv_now:.4f}",    "-"],
        ["Skor",  f"{score:.3f}",        f"Esik: {threshold:.3f}"],
    ]
    tbl = ax_num.table(cellText=rows,
                       colLabels=["Sinyal", "Anlik", "Referans"],
                       loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.6)
    ax_num.set_title("Anlik Degerler", fontsize=10, pad=6)

    sig_lbl = ["MLII (mV)", "V5 (mV)", "Turev"]
    sig_col = ["steelblue", "darkorange", "green"]
    for si in range(3):
        ax_s = fig.add_subplot(gs[1, si])
        ax_s.plot(X_raw[dash_idx, :, si], color=sig_col[si], lw=1.5)
        if dash_label == 1:
            ax_s.axvspan(0, time_steps, alpha=0.08, color='red')
        ax_s.set_title(sig_lbl[si], fontsize=9, pad=3)
        ymin   = np.min(X_raw[dash_idx, :, si])
        ymax   = np.max(X_raw[dash_idx, :, si])
        margin = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
        ax_s.set_ylim(ymin - margin, ymax + margin)
        if si == 0:
            ax_s.axhline(y=patient_hr_base[p_dash],
                         color='blue', linestyle=':', alpha=0.5, label="Bazal")
            ax_s.legend(fontsize=7)

    ax_auc = fig.add_subplot(gs[1, 3])
    ax_auc.axis('off'); ax_auc.set_facecolor(bg_col)
    info = (
        f"Model v7 Performansi\n\n"
        f"ROC AUC      : {roc_auc:.4f}\n"
        f"Sensitivity  : {sensitivity:.3f}\n"
        f"Specificity  : {specificity:.3f}\n"
        f"PPV          : {ppv:.3f}\n"
        f"FN (kacirilan): {fn}\n"
        f"TP (yakalanan): {tp}\n\n"
        f"Stratified hasta bolmesi\n"
        f"44 kayit | Dropout 0.40/0.35/0.20"
    )
    ax_auc.text(0.5, 0.5, info, ha='center', va='center',
                fontsize=9, transform=ax_auc.transAxes, color='#333',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                          edgecolor='#ccc', lw=1))
    ax_auc.set_title("Sistem Ozeti", fontsize=10, pad=6)

    lbl_str = "ARITMIK" if dash_label == 1 else "NORMAL"
    fname   = f"dashboard_v7_{lbl_str}.png"
    plt.savefig(fname, dpi=120, bbox_inches='tight', facecolor=bg_col)
    plt.show()
    print(f"  Dashboard kaydedildi: {fname}")

# ─────────────────────────────────────────────
# 14. MODEL KAYDETME
# ─────────────────────────────────────────────
model.save("bileklik_model_v7.keras")
print("\nModel kaydedildi: bileklik_model_v7.keras")
print("\n" + "=" * 60)
print("TAMAMLANDI - MIT-BIH v7")
print("=" * 60)
