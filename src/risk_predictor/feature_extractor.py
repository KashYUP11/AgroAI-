import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import warnings

warnings.filterwarnings('ignore')


class AdvancedFeatureExtractor:
    """
    Extract advanced ML features for disease risk prediction
    Combines: CNN features, texture analysis, color analysis, morphological features
    """

    def __init__(self):
        self.feature_names = []

    # ========== TEXTURE FEATURES ==========
    def extract_glcm_features(self, image):
        """
        Gray Level Co-occurrence Matrix (GLCM)
        Detects patterns in pixel intensity transitions
        Highly sensitive to early pathogenic colonization
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize for consistency
        gray = cv2.resize(gray, (256, 256))

        # Calculate GLCM using NEW API
        glcm = graycomatrix(
            gray,
            distances=[1, 5],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True
        )

        # Extract GLCM properties
        glcm_features = []
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

        for prop in props:
            # Use NEW graycoprops API (returns [distances, angles] shaped array)
            if prop == 'ASM':
                # ASM is the square of energy
                vals = graycoprops(glcm, 'energy')
                vals = vals ** 2
            else:
                vals = graycoprops(glcm, prop)

            glcm_features.extend(vals.flatten().tolist())

        return np.array(glcm_features)

    def extract_lbp_features(self, image):
        """
        Local Binary Patterns (LBP)
        Captures local texture patterns at multiple scales
        Excellent for detecting microscopic lesions
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = cv2.resize(gray, (256, 256))

        # Multi-scale LBP
        lbp_features = []

        for radius in [1, 3, 5]:
            for n_points in [8, 16]:
                lbp = local_binary_pattern(
                    gray,
                    n_points,
                    radius,
                    method='uniform'
                )

                # Histogram of LBP
                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=n_points + 2,
                    range=(0, n_points + 2)
                )

                # Normalize histogram
                hist = hist.astype('float') / hist.sum()
                lbp_features.extend(hist)

        return np.array(lbp_features)

    def extract_color_features(self, image):
        """
        Advanced color space analysis
        Disease changes leaf color in specific ways
        HSV better captures disease-induced discoloration than RGB
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        color_features = []

        # HSV histogram
        for i, channel in enumerate(cv2.split(hsv)):
            hist = cv2.calcHist([channel], [0], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            color_features.extend(hist)

        # Color moments (mean, std, skewness per channel)
        for channel in cv2.split(hsv):
            mean = np.mean(channel)
            std = np.std(channel)
            skewness = (np.mean((channel - mean) ** 3)) / (std ** 3 + 1e-6)

            color_features.extend([mean, std, skewness])

        return np.array(color_features)

    def extract_morphological_features(self, image):
        """
        Morphological features
        Detects lesion-like patterns and abnormal structures
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = cv2.resize(gray, (256, 256))

        # Threshold to find abnormal regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        morph_features = []

        # Opening and closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Contours in original and processed
        contours_orig, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_opened, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        morph_features.append(len(contours_orig))  # Number of contours
        morph_features.append(len(contours_opened))  # Contours after opening

        # Contour areas
        if contours_orig:
            areas = [cv2.contourArea(c) for c in contours_orig]
            morph_features.append(np.mean(areas))  # Mean area
            morph_features.append(np.std(areas))  # Std dev of areas
            morph_features.append(np.max(areas))  # Max area
            morph_features.append(np.min(areas))  # Min area
        else:
            morph_features.extend([0, 0, 0, 0])

        return np.array(morph_features)

    def extract_edge_features(self, image):
        """
        Edge detection features
        Disease causes irregular edge patterns
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        gray = cv2.resize(gray, (256, 256))

        edge_features = []

        # Canny edges
        edges_canny = cv2.Canny(gray, 50, 150)
        edge_features.append(np.sum(edges_canny) / 255)  # Edge density

        # Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        edge_features.append(np.mean(magnitude))
        edge_features.append(np.std(magnitude))
        edge_features.append(np.max(magnitude))

        return np.array(edge_features)

    def extract_all_features(self, image_path):
        """
        Master function: Extract ALL features
        Returns concatenated feature vector (~200+ features)
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Extract all feature types
        print(f"Extracting features from: {image_path}")

        glcm_feats = self.extract_glcm_features(image)
        print(f"  ✓ GLCM: {len(glcm_feats)} features")

        lbp_feats = self.extract_lbp_features(image)
        print(f"  ✓ LBP: {len(lbp_feats)} features")

        color_feats = self.extract_color_features(image)
        print(f"  ✓ Color: {len(color_feats)} features")

        morph_feats = self.extract_morphological_features(image)
        print(f"  ✓ Morphological: {len(morph_feats)} features")

        edge_feats = self.extract_edge_features(image)
        print(f"  ✓ Edge: {len(edge_feats)} features")

        # Concatenate all
        all_features = np.concatenate([
            glcm_feats,
            lbp_feats,
            color_feats,
            morph_feats,
            edge_feats
        ])

        print(f"  ✅ Total features: {len(all_features)}\n")

        return all_features


# Test the extractor
if __name__ == "__main__":
    extractor = AdvancedFeatureExtractor()

    # Test with an image from your dataset
    test_image = r"C:\Users\Asus\PycharmProjects\LeafDisease-CNN\data\PlantVillage\Tomato_healthy\image_0001.jpg"

    features = extractor.extract_all_features(test_image)
    print(f"Feature vector shape: {features.shape}")
