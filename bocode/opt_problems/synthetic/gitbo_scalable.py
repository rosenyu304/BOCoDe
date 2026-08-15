"""High-dimensional scalable variants of the synthetic test functions used by
GIT-BO (Figure 17): each of nine scalable functions instantiated at dims
{100, 200, 300, 400, 500}.

These are thin ``dim``-fixed subclasses of the existing synthetic wrappers, named
``<Func>_<D>D`` (e.g. ``Ackley_100D``, ``Rosenbrock_500D``). They exist so the
campaign can target a fixed high dimension by name; the base classes stay
unchanged.

Sources:
BoTorch synthetic test functions: https://github.com/meta-pytorch/botorch/blob/main/botorch/test_functions/synthetic.py
"""

from __future__ import annotations

from .Ackley import Ackley
from .DixonPrice import DixonPrice
from .Griewank import Griewank
from .Levy import Levy
from .Michalewicz import Michalewicz
from .Powell import Powell
from .Rastrigin import Rastrigin
from .Rosenbrock import Rosenbrock
from .StyblinskiTang import StyblinskiTang

_BASES = (
    Ackley,
    DixonPrice,
    Griewank,
    Levy,
    Michalewicz,
    Powell,
    Rastrigin,
    Rosenbrock,
    StyblinskiTang,
)
_DIMS = (100, 200, 300, 400, 500)

#: name -> class, for registration in the synthetic package's SYNTHETIC_PROBLEMS.
GITBO_SCALABLE_PROBLEMS: dict[str, type] = {}

for _base in _BASES:
    for _d in _DIMS:
        _name = f"{_base.__name__}_{_d}D"
        _cls = type(
            _name,
            (_base,),
            {
                "botorch_kwargs": {"dim": _d},
                "available_dimensions": (2, 500),
                "__doc__": f"{_base.__name__} fixed to {_d} dimensions (GIT-BO scalable variant).",
            },
        )
        globals()[_name] = _cls
        GITBO_SCALABLE_PROBLEMS[_name] = _cls

del _base, _d, _name, _cls

__all__ = [*GITBO_SCALABLE_PROBLEMS.keys(), "GITBO_SCALABLE_PROBLEMS"]
