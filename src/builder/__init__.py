# nanocrystal_builder/__init__.py
__all__ = [
    "main",
    "generate_nucleation_map",
    "generate_nucleation_result",
    "load_nucleation_spec",
    "write_nucleation_bundle",
    "registry_to_dict",
    "write_nucleation_json",
]


def __getattr__(name):
    if name in {
        "generate_nucleation_map",
        "generate_nucleation_result",
        "load_nucleation_spec",
        "write_nucleation_bundle",
        "registry_to_dict",
        "write_nucleation_json",
    }:
        from . import nucleation
        return getattr(nucleation, name)
    raise AttributeError(name)
