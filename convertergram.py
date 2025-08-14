import os
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
import librosa
import soundfile as sf

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_FFT_SIZE = 1024
DEFAULT_HOP_DIVISOR = 4
DEFAULT_WINDOW = "hann"
DEFAULT_DB_MIN = -60.0
DEFAULT_DB_MAX = 0.0
DEFAULT_GRIFFINLIM_ITERS = 60
DEFAULT_NORMALIZE_LEVEL = 0.98


def image_to_audio(
    img_path: str,
    output_wav: Optional[str] = None,
    *,
    width: int = 800,
    fft_size: int = DEFAULT_FFT_SIZE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    db_min: float = DEFAULT_DB_MIN,
    db_max: float = DEFAULT_DB_MAX,
    griffinlim_iters: int = DEFAULT_GRIFFINLIM_ITERS,
    window: str = DEFAULT_WINDOW,
    normalize_peak: float = DEFAULT_NORMALIZE_LEVEL,
    create_spectrogram: bool = False,
    spectrogram_path: Optional[str] = None,
) -> Union[str, Tuple[str, str]]:
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    hop_length = fft_size // DEFAULT_HOP_DIVISOR

    img = Image.open(img_path).convert("L")
    target_height = fft_size // 2 + 1
    img = img.resize((width, target_height), resample=Image.BICUBIC)
    img_data = np.flipud(np.array(img).astype(np.float32))

    mag_db = db_min + (db_max - db_min) * (img_data / 255.0)
    magnitude = 10 ** (mag_db / 20.0)

    audio = librosa.griffinlim(
        S=magnitude,
        n_iter=griffinlim_iters,
        hop_length=hop_length,
        win_length=fft_size,
        window=window,
        center=True,
    )

    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * normalize_peak

    if output_wav is None:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_wav = base_name + ".wav"

    sf.write(output_wav, audio, sample_rate)

    spectrogram_out = None
    if create_spectrogram:
        if plt is None:
            raise RuntimeError("matplotlib is required for spectrogram generation")

        if spectrogram_path is None:
            base_name = os.path.splitext(output_wav)[0]
            spectrogram_path = base_name + "_spectrogram.png"

        duration = magnitude.shape[1] * hop_length / sample_rate

        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        im = ax.imshow(
            mag_db,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=(0, duration, 0, sample_rate / 2),
            vmin=db_min,
            vmax=db_max,
        )
        ax.set_title("Spectrogram (dB)")
        cbar = fig.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label("dB")
        fig.savefig(spectrogram_path, dpi=150)
        plt.close(fig)
        spectrogram_out = spectrogram_path

    if create_spectrogram:
        return output_wav, spectrogram_out

    return output_wav


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python convertgram.py targetimg.png output.wav")
        sys.exit(1)

    input_image = sys.argv[1]
    output_audio = sys.argv[2]
    generate_spec = "--spectrogram" in sys.argv

    if generate_spec:
        audio_path, spec_path = image_to_audio(
            input_image, output_audio, create_spectrogram=True
        )
        print(f"Audio saved to: {audio_path}")
        print(f"Spectrogram saved to: {spec_path}")
    else:
        audio_path = image_to_audio(input_image, output_audio)
        print(f"Audio saved to: {audio_path}")
