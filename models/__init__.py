from models.dpcgans import DPCGAN

__all__ = "DPCGANSynthesizer"


def get_all_synthesizers():
    return {name: globals()[name] for name in __all__}
