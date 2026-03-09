import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.sparse.linalg import cg, LinearOperator

# ============================================================
# Robust Flash Deblurring : estimation du noyau K sur un patch
# B : patch flou
# F : patch flash
# ============================================================

#  fonctions   

def gray01(path):
    # Charge une image, la convertit en niveaux de gris
    # et normalise les intensités entre 0 et 1
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    if x is None:
        raise FileNotFoundError(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return x

def sobelx(x):
    # Gradient horizontal
    return cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3) / 8.0

def sobely(x):
    # Gradient vertical
    return cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3) / 8.0

def gradmag(x):
    # Norme du gradient
    return np.sqrt(sobelx(x)**2 + sobely(x)**2 + 1e-12)

def conv_same(x, k):
    # Convolution 2D en gardant la même taille de sortie
    return fftconvolve(x, k, mode="same")

def flipk(k):
    # Retourne le noyau pour les convolutions adjointes
    return k[::-1, ::-1]

def normalize_kernel(k):
    # Force le noyau à être positif et de somme 1
    k = np.maximum(k, 0)
    s = k.sum()
    if s < 1e-12:
        k[:] = 0
        k[k.shape[0] // 2, k.shape[1] // 2] = 1.0
    else:
        k /= s
    return k

def upsample_kernel(k, new_size):
    # Agrandit le noyau quand on passe à une échelle plus fine
    k2 = cv2.resize(k, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
    k2 = np.maximum(k2, 0) * (new_size * new_size) / (k.shape[0] * k.shape[1])
    return normalize_kernel(k2)

def build_pyramid(x, levels):
    # Construit une pyramide d'images du plus petit au plus grand niveau
    pyr = [x]
    for _ in range(1, levels):
        x = cv2.pyrDown(x)
        pyr.append(x)
    return pyr[::-1]

def phase_align(F, B):
    # Aligne localement F sur B par corrélation de phase
    shift, _ = cv2.phaseCorrelate(np.float32(F), np.float32(B))
    dx, dy = shift
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    Fa = cv2.warpAffine(
        F, M, (F.shape[1], F.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return Fa

def align_images(F, B):
    # Aligne globalement l'image flash sur l'image floue
    shift, response = cv2.phaseCorrelate(
        np.float32(F),
        np.float32(B)
    )

    dx, dy = shift
    print("global shift:", dx, dy)

    M = np.array([
        [1, 0, dx],
        [0, 1, dy]
    ], dtype=np.float32)

    F_aligned = cv2.warpAffine(
        F, M,
        (F.shape[1], F.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    return F_aligned

#  poids IRLS 

def lorentz_weights(I, F, eps):
    # Poids robustes pour contraindre grad(I) à suivre grad(F)
    dx = sobelx(I) - sobelx(F)
    dy = sobely(I) - sobely(F)
    d2 = dx * dx + dy * dy
    # Eq. du papier : w_i = 2 / (2 eps^2 + ||gradI-gradF||^2)
    return 2.0 / (2.0 * eps * eps + d2)

def kernel_irls_weights(K, alpha, delta=1e-6):
    # Poids IRLS pour la régularisation du noyau |K|^alpha
    return alpha / np.power(np.abs(K) + delta, 2.0 - alpha)

#  estimation principale du noyau 

def estimate_kernel(B, F,
                    kernel_schedule=(3, 7, 15, 31),
                    outer_iters=6,
                    lam_f=0.08,
                    lam_k=0.02,
                    alpha=0.8,
                    eps=0.05):

    # Nombre de niveaux de pyramide
    n_levels = len(kernel_schedule)

    # Construction des pyramides
    Bp = build_pyramid(B, n_levels)
    Fp = build_pyramid(F, n_levels)

    # Initialisation du noyau par un delta
    K = np.zeros((kernel_schedule[0], kernel_schedule[0]), np.float32)
    K[kernel_schedule[0] // 2, kernel_schedule[0] // 2] = 1.0

    I = None

    # Parcours des échelles du plus grossier au plus fin
    for s, (Bs, Fs, ks) in enumerate(zip(Bp, Fp, kernel_schedule)):
        print(f"\n--- Scale {s+1}/{n_levels} | kernel {ks}x{ks} ---")

        # Recalage local du flash sur le blur
        Fs = phase_align(Fs, Bs)

        # Mise à la bonne taille du noyau
        if K.shape[0] != ks:
            K = upsample_kernel(K, ks)

        # Initialisation de l'image latente à cette échelle
        if I is None:
            I = Fs.copy()
        else:
            I = cv2.resize(I, (Bs.shape[1], Bs.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Masque de confiance basé sur les gradients forts de F
        G = gradmag(Fs)
        thresh = np.percentile(G, 70)
        M = (G >= thresh).astype(np.float32)

        # Alternance : mise à jour de I puis de K
        for it in range(outer_iters):
            I = update_I_masked(Bs, Fs, K, M, lam_f=lam_f, eps=eps, iters=2)

            K_old = K.copy()
            K = update_K_masked(Bs, I, K, M, alpha=alpha, lam_k=lam_k, iters=3)

            rel = np.linalg.norm(K - K_old) / (np.linalg.norm(K_old) + 1e-8)
            print(f"iter {it+1}/{outer_iters}, rel={rel:.4e}, K.max={K.max():.4f}")

            # Arrêt si le noyau change peu
            if rel < 5e-3:
                break

    return I, K

def update_I_masked(B, F, K, M, lam_f=0.02, eps=0.01, iters=3):
    # Met à jour l'image latente I en gardant K fixé
    I = F.copy()
    kf = flipk(K)

    for _ in range(iters):
        W = lorentz_weights(I, F, eps) * M

        def Aop(xvec):
            x = xvec.reshape(B.shape)
            data = conv_same(conv_same(x, K), kf)

            gx = sobelx(x)
            gy = sobely(x)

            reg = -cv2.Sobel(W * gx, cv2.CV_32F, 1, 0, ksize=3) / 8.0 \
                  -cv2.Sobel(W * gy, cv2.CV_32F, 0, 1, ksize=3) / 8.0

            return (data + lam_f * reg).ravel()

        rhs = conv_same(B, kf)
        rhs += lam_f * (
            -cv2.Sobel(W * sobelx(F), cv2.CV_32F, 1, 0, ksize=3) / 8.0
            -cv2.Sobel(W * sobely(F), cv2.CV_32F, 0, 1, ksize=3) / 8.0
        )

        A = LinearOperator((B.size, B.size), matvec=Aop, dtype=np.float32)
        sol, _ = cg(A, rhs.ravel(), x0=I.ravel(), maxiter=100)
        I = sol.reshape(B.shape).astype(np.float32)
        I = np.clip(I, 0.0, 1.0)

    return I

#  extraction du patch 

def extract_matched_patches(F, B, x, y, ps, search_radius=40):
    """
    F : image flash en niveaux de gris [0,1]
    B : image blur en niveaux de gris [0,1]
    x,y : coin haut-gauche du patch dans F
    ps : taille du patch
    """

    # Patch de référence dans F
    Fp = F[y:y+ps, x:x+ps]

    # Zone de recherche dans B autour de la position attendue
    x0 = max(0, x - search_radius)
    y0 = max(0, y - search_radius)
    x1 = min(B.shape[1], x + ps + search_radius)
    y1 = min(B.shape[0], y + ps + search_radius)

    search = B[y0:y1, x0:x1]

    # Recherche du meilleur patch correspondant
    res = cv2.matchTemplate(
        search.astype(np.float32),
        Fp.astype(np.float32),
        method=cv2.TM_CCOEFF_NORMED
    )

    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    bx = x0 + max_loc[0]
    by = y0 + max_loc[1]

    Bp = B[by:by+ps, bx:bx+ps]

    return Fp, Bp, (x, y), (bx, by), max_val

#  mise à jour du noyau 

def update_K_masked(B, I, K, M, alpha=0.8, lam_k=0.02, iters=3):
    # Met à jour K en gardant I fixé
    Fx, Fy = sobelx(I), sobely(I)
    Bx, By = sobelx(B), sobely(B)

    # On ne garde que les régions fiables du masque
    Fx *= M
    Fy *= M
    Bx *= M
    By *= M

    ks = K.shape[0]
    yy, xx = np.mgrid[0:ks, 0:ks]

    for _ in range(iters):
        WK = kernel_irls_weights(K, alpha)

        predx = conv_same(Fx, K)
        predy = conv_same(Fy, K)
        rx, ry = predx - Bx, predy - By

        g = conv_same(rx, flipk(Fx)) + conv_same(ry, flipk(Fy))
        cy, cx = g.shape[0] // 2, g.shape[1] // 2
        g = g[cy - ks // 2: cy + ks // 2 + 1, cx - ks // 2: cx + ks // 2 + 1]
        g += lam_k * WK * K

        # Descente de gradient stabilisée
        step = 0.15 / (np.max(np.abs(g)) + 1e-6)
        K = K - step * g

        K = np.maximum(K, 0)

        # Suppression des faibles valeurs
        K[K < 0.08 * K.max()] = 0

        # Petit lissage
        K = cv2.GaussianBlur(K, (3, 3), 0)

        # Normalisation
        s = K.sum()
        if s > 1e-12:
            K /= s
        else:
            K[:] = 0
            K[ks // 2, ks // 2] = 1.0

        # Recentrage du noyau
        cy_mass = (yy * K).sum()
        cx_mass = (xx * K).sum()

        Mshift = np.array([
            [1, 0, (ks - 1) / 2 - cx_mass],
            [0, 1, (ks - 1) / 2 - cy_mass]
        ], dtype=np.float32)

        K = cv2.warpAffine(
            K, Mshift, (ks, ks),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Nettoyage final à chaque itération
        K = normalize_kernel(K)
        K[K < 0.12 * K.max()] = 0
        K = cv2.GaussianBlur(K, (3, 3), 0)
        K /= K.sum()

    return K

#  chargement des images INPUT

F = gray01("F_crop.png")
B = gray01("B_crop.png")

# Mise à la même taille
B = cv2.resize(B, (F.shape[1], F.shape[0]), interpolation=cv2.INTER_LINEAR)

# Alignement global
F = align_images(F, B)

# Affichage des images alignées
plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
plt.imshow(F, cmap='gray')
plt.title("Flash aligné")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(B, cmap='gray')
plt.title("Blur")
plt.axis("off")

plt.tight_layout()
plt.show()

# ---------------------- choix du patch ----------------------

x, y, ps = 1310, 1399, 225

Fp, Bp, flash_pos, blur_pos, score = extract_matched_patches(F, B, x, y, ps, search_radius=60)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.imshow(Fp, cmap='gray')
plt.title("Patch flash recalé")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(Bp, cmap='gray')
plt.title("Patch blur correspondant")
plt.axis("off")

plt.tight_layout()
plt.show()

def final_kernel_cleanup(K):
    # Nettoyage final du noyau pour enlever les parasites
    thresh = 0.42 * K.max()
    mask = (K > thresh)

    K = K * mask
    K = cv2.GaussianBlur(K, (3, 3), 0)

    K = np.maximum(K, 0)
    K /= K.sum() + 1e-8

    return K

print("Patch flash :", flash_pos)
print("Patch blur  :", blur_pos)
print("Score match :", score)

# ---------------------- estimation finale du noyau ----------------------

I_est, K_est = estimate_kernel(
    Bp, Fp,
    kernel_schedule=(3, 7, 15, 27),
    outer_iters=6,
    lam_f=0.1,
    lam_k=0.02,
    alpha=0.8,
    eps=0.05,
)

# Nettoyage final
K_est = final_kernel_cleanup(K_est)

print("kernel sum:", K_est.sum())
print("kernel size:", K_est.shape)
print("kernel max:", K_est.max())

# Affichage final
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(Fp, cmap="gray")
plt.title("Patch flash")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(Bp, cmap="gray")
plt.title("Patch blur")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(K_est, cmap="gray")
plt.title("Kernel estimé")
plt.axis("off")

plt.tight_layout()
plt.show()

# Sauvegarde du noyau estimé
np.save("K_est_crop.npy", K_est)
print("Kernel sauvegardé dans K_est.npy")