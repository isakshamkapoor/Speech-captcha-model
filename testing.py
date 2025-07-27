import os
import json
import numpy as np
import scipy.io.wavfile
import joblib
from sklearn.decomposition import PCA

from getPotentialSpeakLocation import getPotentialSpeakLocation
from FeatureExtraction import Mfcc  # or use Raw or Rasta depending on training

def load_and_infer(wav_file, txt_file, model_path, pca_path, left=2500, right=2500, applyPca=True):
    # Load trained model
    mlModel = joblib.load(model_path)

    # Load saved PCA
    if applyPca:
        pca = joblib.load(pca_path)

    # Load test .wav file
    rate, data = scipy.io.wavfile.read(wav_file)
    data = np.asarray([0] * left + list(data) + [0] * right)

    # Load expected output for comparison
    output = json.load(open(txt_file))
    expectedLocs = list(map(int, output["offsets"][1:-1].split(',')))
    expectedLocs = [(x + left) for x in expectedLocs]

    print("Expected spoken locs = " + str(expectedLocs))

    # Get potential spoken locations
    locs = getPotentialSpeakLocation(data, rate, left, right, 4)
    print("Detected spoken locs = " + str(locs))

    # Extract 4 segments around locs
    signals = []
    for loc in locs:
        sta = loc - left
        fin = loc + right
        signals.append(data[sta:fin])
    signals = np.array(signals)

    # Feature extraction (must match what you used during training!)
    featureExtraction = Mfcc(flatten=True)
    signals = featureExtraction(signals, rate)

    # Apply saved PCA
    if applyPca:
        signals = pca.transform(signals)

    # Predict
    predictedVals = mlModel.predict(signals)

    # Decode predicted numbers
    captchas = ""
    for c in predictedVals:
        captchas += str(c) if c < 10 else chr(ord('a') + c - 10)

    print("Predicted Captcha =", captchas)
    print("Expected Captcha =", output["code"])

# Example usage
if __name__ == "__main__":
    wav_file = "D:\\Speech Processing\\project\\securimage_all\\test\\3df81dfb.wav"
    txt_file = "D:\\Speech Processing\\project\\securimage_all\\test\\3df81dfb.txt"
    model_path = "svm_model.pkl"
    pca_path = "pca_transform.pkl"

    load_and_infer(wav_file, txt_file, model_path, pca_path, left=2500, right=2500, applyPca=True)
