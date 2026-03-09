import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_rgb01(path):
    # Charge une image couleur, la convertit en RGB
    # puis normalise les intensités entre 0 et 1
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    if x is None:
        raise FileNotFoundError(path)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return x


def rgb_to_gray01(img):
    # Convertit une image RGB [0,1] en niveaux de gris [0,1]
    g = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return g.astype(np.float32) / 255.0


def align_ecc_and_crop(B_rgb, F_rgb, motion_model="euclidean", number_of_iterations=5000, termination_eps=1e-7):
    """
    Aligne F sur B avec ECC, puis recadre la zone commune valide.

    motion_model:
        "translation" -> cv2.MOTION_TRANSLATION
        "euclidean"   -> cv2.MOTION_EUCLIDEAN
        "affine"      -> cv2.MOTION_AFFINE
    """


    #  Mettre F à la même taille que B

    if F_rgb.shape != B_rgb.shape:
        F_rgb = cv2.resize(F_rgb, (B_rgb.shape[1], B_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Conversion en niveaux de gris pour le recalage
    B_gray = rgb_to_gray01(B_rgb)
    F_gray = rgb_to_gray01(F_rgb)

    h, w = B_gray.shape


    #  Choix du modèle de mouvement

    if motion_model == "translation":
        warp_mode = cv2.MOTION_TRANSLATION
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif motion_model == "euclidean":
        # translation + rotation
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    elif motion_model == "affine":
        # translation + rotation + échelle/cisaillement léger
        warp_mode = cv2.MOTION_AFFINE
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    else:
        raise ValueError("motion_model doit être 'translation', 'euclidean' ou 'affine'.")

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        number_of_iterations,
        termination_eps
    )


    # Estimation du recalage ECC
    #    On cherche la transformation qui aligne F sur B
  
    cc, warp_matrix = cv2.findTransformECC(
        B_gray,
        F_gray,
        warp_matrix,
        warp_mode,
        criteria,
        inputMask=None,
        gaussFiltSize=5
    )

    print("ECC correlation coefficient:", cc)
    print("Warp matrix:\n", warp_matrix)

  
    # Appliquer la transformation à F
   
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        raise NotImplementedError("Pas utilisé ici.")
    else:
        F_aligned = cv2.warpAffine(
            F_rgb,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        # Masque indiquant quelles zones restent valides après warp
        valid_mask = cv2.warpAffine(
            np.ones((h, w), dtype=np.uint8),
            warp_matrix,
            (w, h),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )


    # Déterminer la plus grande zone commune valide

    ys, xs = np.where(valid_mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("Le recalage ECC a produit un masque vide.")

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Petite marge pour éviter les bords noirs/interpolés
    margin = 5
    x_min = min(max(x_min + margin, 0), w - 1)
    x_max = max(min(x_max - margin, w - 1), x_min + 1)
    y_min = min(max(y_min + margin, 0), h - 1)
    y_max = max(min(y_max - margin, h - 1), y_min + 1)

    # Recadrage des deux images sur la zone commune
    B_crop = B_rgb[y_min:y_max, x_min:x_max]
    F_crop = F_aligned[y_min:y_max, x_min:x_max]
    mask_crop = valid_mask[y_min:y_max, x_min:x_max]

    print("Crop valid area:")
    print("x:", x_min, x_max, " | y:", y_min, y_max)
    print("Final shape:", B_crop.shape)

    return B_crop, F_crop, warp_matrix, mask_crop




if __name__ == "__main__":
    # Chargement des deux images d'origine
    B_rgb = load_rgb01("Images/flou_originale.png")
    F_rgb = load_rgb01("Images/flash_originale.png")

    # Essai possible avec un modèle euclidien
    # B_crop, F_crop, warp_matrix, mask_crop = align_ecc_and_crop(
    #     B_rgb, F_rgb,
    #     motion_model="euclidean",
    #     number_of_iterations=5000,
    #     termination_eps=1e-7
    # )

    # Ici on utilise le modèle affine, plus souple
    B_crop, F_crop, warp_matrix, mask_crop = align_ecc_and_crop(
        B_rgb, F_rgb,
        motion_model="affine",
        number_of_iterations=5000,
        termination_eps=1e-7
    )

    
    # Visualisation du recalage
  
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(B_crop)
    plt.title("B recadrée")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(F_crop)
    plt.title("F alignée + recadrée")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    # Différence en niveaux de gris entre les deux images recalées
    diff = np.abs(rgb_to_gray01(B_crop) - rgb_to_gray01(F_crop))
    plt.imshow(diff, cmap="hot")
    plt.title("Différence |B-F|")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    # Superposition simple pour vérifier visuellement l'alignement
    overlay = 0.5 * B_crop + 0.5 * F_crop
    plt.imshow(np.clip(overlay, 0, 1))
    plt.title("Superposition")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Sauvegarde des images recadrées pour les réutiliser dans les autres scripts
    cv2.imwrite("B_crop.png", (B_crop * 255).astype(np.uint8)[:, :, ::-1])
    cv2.imwrite("F_crop.png", (F_crop * 255).astype(np.uint8)[:, :, ::-1])