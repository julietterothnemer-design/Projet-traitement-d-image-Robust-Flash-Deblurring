import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.sparse.linalg import cg, LinearOperator

# ============================================================
# Partie 2 - Robust flash deblurring : reconstruction sur l'image entière
# ============================================================
# Entrées :
#   B : image couleur floue
#   F : image couleur flash
#   K_est : noyau estimé
#
# Sortie :
#   I_rgb : image reconstruite finale
#
# L'énergie = optimisation :
#   ||I*K - B||^2 + lambda_f * M o rho(grad(I)-grad(F)) + lambda_i * |grad(I)|^alpha
# avec :
#   - un masque M local
#   - une résolution IRLS
#   - une reconstruction sur toute l'image
# ============================================================



# UTILITAIRES DE BASE


def load_rgb01(path):
    # Charge une image couleur et la convertit en RGB float32 entre 0 et 1
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    if x is None:
        raise FileNotFoundError(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return x


def rgb_to_gray01(x):
    # Convertit une image RGB [0,1] en niveau de gris [0,1]
    g = cv2.cvtColor((np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return g.astype(np.float32) / 255.0


def sobelx(x):
    # Gradient horizontal
    return cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3) / 8.0


def sobely(x):
    # Gradient vertical
    return cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3) / 8.0


def grad_mag(x):
    # Norme du gradient
    gx = sobelx(x)
    gy = sobely(x)
    return np.sqrt(gx * gx + gy * gy + 1e-12)


def conv_same(x, k):
    # Convolution 2D avec sortie de même taille
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
    return k.astype(np.float32)



#  MASQUE M (approximation pratique)


def build_mask_M(B_gray, F_gray, K, sat_thresh=0.95, grad_ratio=2.5, blur_sigma=5.0):
    """
    Construit un masque M inspiré du papier :
      - 20 pour les zones sur-saturées dans B
      - 0 pour les zones où F est peu fiable
      - 1 ailleurs
    puis lisse ce masque.

    Ici, les artefacts flash sont détectés quand le gradient de F
    est fort alors que celui de B est faible.
    """
    # Zones saturées dans l'image floue
    over_sat_B = (B_gray >= sat_thresh).astype(np.float32)

    # Zones saturées dans l'image flash
    over_sat_F = (F_gray >= sat_thresh).astype(np.float32)

    # Gradients des deux images
    gB = grad_mag(B_gray)
    gF = grad_mag(F_gray)

    # Détection simple des artefacts flash
    flash_art = ((gF > grad_ratio * (gB + 1e-4)) & (gF > np.percentile(gF, 75))).astype(np.float32)

    # Construction du masque
    M = np.ones_like(B_gray, dtype=np.float32)
    M[flash_art > 0] = 0.0
    M[over_sat_B > 0] = 20.0

    # Si le flash est saturé localement, on réduit sa confiance
    M[over_sat_F > 0] = np.minimum(M[over_sat_F > 0], 0.5)

    # Lissage du masque
    M = cv2.GaussianBlur(M, (0, 0), blur_sigma)

    # Bornage de sécurité
    M = np.clip(M, 0.0, 20.0)
    return M.astype(np.float32), flash_art, over_sat_B



# POIDS IRLS


def lorentz_weights(I, F, eps):
    """
    Poids robustes du terme rho(grad(I)-grad(F)).
    Forme utilisée dans le papier :
        w_i = 2 / (2 eps^2 + ||gradI-gradF||^2)
    """
    dx = sobelx(I) - sobelx(F)
    dy = sobely(I) - sobely(F)
    d2 = dx * dx + dy * dy
    W = 2.0 / (2.0 * eps * eps + d2)
    return W.astype(np.float32)


def sparse_grad_weights(I, alpha=0.8, delta=1e-4):
    """
    Poids IRLS pour approximer la pénalisation |grad(I)|^alpha.
    """
    gx = sobelx(I)
    gy = sobely(I)
    g = np.sqrt(gx * gx + gy * gy + delta * delta)
    W = alpha / np.power(g, 2.0 - alpha)
    return W.astype(np.float32)



#  MISE A JOUR D'UN CANAL


def update_channel_IRLS(Bc, Fc, K, M, lam_f=0.03, lam_i=0.002, alpha=0.8, eps=0.01,
                        outer_irls=6, cg_maxiter=120):
    """
    Reconstruit un canal Ic à partir de :
      Bc : canal flou
      Fc : canal flash
      K  : noyau estimé
      M  : masque local

    Optimise :
      ||I*K - B||^2 + lam_f * M o rho(grad(I)-grad(F)) + lam_i * |grad(I)|^alpha
    via IRLS.
    """
    # Initialisation du canal reconstruit
    I = Fc.copy().astype(np.float32)
    Kf = flipk(K)

    h, w = Bc.shape
    n = h * w

    for it in range(outer_irls):
        # Poids robustes recalculés à chaque itération
        Wf = lorentz_weights(I, Fc, eps) * M
        Wi = sparse_grad_weights(I, alpha=alpha)

        def Aop(xvec):
            # Opérateur linéaire appliqué au vecteur image
            x = xvec.reshape(h, w).astype(np.float32)

            # Terme de fidélité : K^T K x
            data = conv_same(conv_same(x, K), Kf)

            # Gradient de l'image courante
            gx = sobelx(x)
            gy = sobely(x)

            # Régularisation liée à l'image flash
            reg_f = (
                -cv2.Sobel(Wf * gx, cv2.CV_32F, 1, 0, ksize=3) / 8.0
                -cv2.Sobel(Wf * gy, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            )

            # Régularisation sparse sur les gradients
            reg_i = (
                -cv2.Sobel(Wi * gx, cv2.CV_32F, 1, 0, ksize=3) / 8.0
                -cv2.Sobel(Wi * gy, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            )

            out = data + lam_f * reg_f + lam_i * reg_i
            return out.ravel()

        # Second membre du système
        rhs = conv_same(Bc, Kf)

        rhs += lam_f * (
            -cv2.Sobel(Wf * sobelx(Fc), cv2.CV_32F, 1, 0, ksize=3) / 8.0
            -cv2.Sobel(Wf * sobely(Fc), cv2.CV_32F, 0, 1, ksize=3) / 8.0
        )

        # Résolution par gradient conjugué
        A = LinearOperator((n, n), matvec=Aop, dtype=np.float32)

        sol, info = cg(A, rhs.ravel(), x0=I.ravel(), maxiter=cg_maxiter)
        if info != 0:
            print(f"[WARN] CG info={info} à l'itération IRLS {it+1}")

        I = sol.reshape(h, w).astype(np.float32)

        # Ici on a choisi de ne pas réancrer explicitement I vers F
        # I = 0.98 * I + 0.02 * Fc

        I = np.clip(I, 0.0, 1.0)

        # Suivi de l'erreur de fidélité
        pred = conv_same(I, K)
        data_loss = np.mean((pred - Bc) ** 2)
        print(f"  IRLS {it+1}/{outer_irls} | data_loss={data_loss:.6f}")

    return I



# RECONSTRUCTION COULEUR COMPLETE


def reconstruct_full_image(B_rgb, F_rgb, K_est,
                           lam_f=0.03, lam_i=0.002, alpha=0.8, eps=0.01,
                           outer_irls=6, cg_maxiter=120):
    """
    Reconstruction couleur complète.
    """
    # Nettoyage / normalisation du noyau
    K_est = normalize_kernel(K_est.astype(np.float32))

    # Conversion en gris pour construire le masque
    B_gray = rgb_to_gray01(B_rgb)
    F_gray = rgb_to_gray01(F_rgb)

    # Construction du masque M
    M, flash_art, over_sat = build_mask_M(B_gray, F_gray, K_est)

    print("Mask M stats:")
    print("  min =", float(M.min()))
    print("  max =", float(M.max()))
    print("  mean =", float(M.mean()))

    # Reconstruction séparée des canaux R, G, B
    channels = []
    for c, name in enumerate(["R", "G", "B"]):
        print(f"\n=== Reconstruction canal {name} ===")
        Ic = update_channel_IRLS(
            B_rgb[:, :, c],
            F_rgb[:, :, c],
            K_est,
            M,
            lam_f=lam_f,
            lam_i=lam_i,
            alpha=alpha,
            eps=eps,
            outer_irls=outer_irls,
            cg_maxiter=cg_maxiter,
        )
        channels.append(Ic)

    # Empilement des trois canaux reconstruits
    I_rgb = np.stack(channels, axis=2)
    I_rgb = np.clip(I_rgb, 0.0, 1.0)
    return I_rgb, M, flash_art, over_sat



#  VISUALISATION


def show_results(B_rgb, F_rgb, I_rgb, K_est, M, flash_art, over_sat):
    # Affiche les résultats principaux du pipeline
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(B_rgb)
    plt.title("Blur B")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(F_rgb)
    plt.title("Flash F")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(I_rgb)
    plt.title("Reconstruction finale I")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(K_est, cmap="gray")
    plt.title(f"Noyau K ({K_est.shape[0]}x{K_est.shape[1]})")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(M, cmap="viridis")
    plt.title("Masque M")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    overlay = np.dstack([
        np.clip(over_sat, 0, 1),        # rouge = saturation
        np.clip(flash_art, 0, 1),       # vert = artefact flash
        np.zeros_like(over_sat)
    ])
    plt.imshow(overlay)
    plt.title("Rouge=over-sat, Vert=flash-art")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



# SAUVEGARDE DU RESULTAT


def save_rgb01(path, img):
    # Sauvegarde une image RGB [0,1] au format PNG/JPG
    out = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, out)




if __name__ == "__main__":
    # Chargement des images recadrées / alignées
    B_rgb = load_rgb01("B_crop.png")
    F_rgb = load_rgb01("F_crop.png")

    # Vérification des tailles
    if F_rgb.shape != B_rgb.shape:
        F_rgb = cv2.resize(F_rgb, (B_rgb.shape[1], B_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Versions en niveaux de gris pour vérification
    Fg = cv2.cvtColor((F_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    Bg = cv2.cvtColor((B_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    print("Shape Fg:", Fg.shape)
    print("Shape Bg:", Bg.shape)

    # Chargement du noyau estimé à l'étape précédente
    K_est = np.load("K_est_crop.npy").astype(np.float32)

    print("kernel sum:", float(K_est.sum()))
    print("kernel size:", K_est.shape)
    print("kernel max:", float(K_est.max()))

    # Superposition visuelle possible si besoin
    # fig, ax = plt.subplots()
    # ax.imshow(F_rgb)
    # ax.imshow(B_rgb, alpha=0.7)
    # ax.axis("off")
    # plt.show()

    # Reconstruction finale
    I_rgb, M, flash_art, over_sat = reconstruct_full_image(
        B_rgb, F_rgb, K_est,
        lam_f=0.0008,
        lam_i=0.00015,
        alpha=0.8,
        eps=0.02,
        outer_irls=2,
        cg_maxiter=20
    )

    # Affichage des résultats
    show_results(B_rgb, F_rgb, I_rgb, K_est, M, flash_art, over_sat)

    # Sauvegarde de l'image reconstruite
    save_rgb01("reconstruction_finale.png", I_rgb)
    print("\nImage sauvegardée : reconstruction_finale.png")