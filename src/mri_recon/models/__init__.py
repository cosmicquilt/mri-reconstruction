"""reconstruction models selectable by name

zero-filled baseline is pure numpy and re-exported here learned models (unet
unrolled dc-cnn) need torch so build_model imports them lazily importing this
package costs nothing until you ask for a trainable model
"""

from .zero_filled import zero_filled_reconstruction

__all__ = ["zero_filled_reconstruction", "build_model"]


def build_model(name: str, **kwargs):
    """factory unet or unrolled (both torch nn.module)

    zero_filled is parameter-free use zero_filled_reconstruction directly rather
    than constructing a module
    """
    name = name.lower()
    if name == "unet":
        from .unet import UnetModel

        return UnetModel(**kwargs)
    if name in ("unrolled", "dccnn", "varnet"):
        from .unrolled import UnrolledRecon

        return UnrolledRecon(**kwargs)
    if name in ("zero_filled", "zerofilled", "zf"):
        raise ValueError(
            "zero_filled has no trainable module; call "
            "models.zero_filled.zero_filled_reconstruction(masked_kspace)."
        )
    raise ValueError(f"Unknown model {name!r} (use 'unet' or 'unrolled').")
