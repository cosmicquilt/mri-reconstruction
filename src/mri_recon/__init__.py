"""mri_recon physics-informed reconstruction of undersampled mri

submodules imported lazily by callers not here on purpose
core physics (fft masking metrics qc data.synthetic models.zero_filled) is pure
numpy importable with no deep-learning stack
learned models and training loop pull in torch only when used
"""

__version__ = "0.1.0"
