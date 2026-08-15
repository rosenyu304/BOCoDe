"""EPFL AIG logic-synthesis sequence optimization — 20 circuits, 56-D mixed each.

Berkeley ABC compiles a logic circuit by applying a *sequence* of synthesis operators;
which operators, in which order, and with which flags, decides how big and how deep the
resulting LUT netlist is. Choosing that sequence is the ``aig_optimization_hyp`` task of
MCBO / BOiLS — one of the reference *combinatorial-plus-continuous* real-world benchmarks
for mixed-variable optimization — instantiated here on the 20 circuits of the EPFL
combinational benchmark suite (10 arithmetic, 10 control).

Search space (56-D, identical for every circuit; the static description lives in
``_epfl_base.py`` and is cross-checked against MCBO on first use)::

    dims 0..19  : the operator sequence, pattern 'basic_w_post_map'
                  ``([pre]*7 + [map] + [post, post]) * 2``, i.e. 14 slots with 10
                  pre-mapping operators to choose from, 2 slots with 2 mapping operators
                  (``if`` / ``if -a``) and 4 slots with 2 post-mapping operators
                  (``speedup_if`` / ``mfs2_lutpack``). The level of a slot is the
                  operator's index in MCBO's ``all_operators``.
    dims 20..55 : the operator hyperparameters ('boils_hyp_op_space'): 22 integers
                  (e.g. ``if C`` in [4, 1024], ``mfs2_lutpack mfs2 M`` in [300, 1000])
                  and 14 booleans (e.g. ``balance -l``).

**Objective.** The sequence is run through Berkeley ABC on the circuit's ``.blif`` file
and scored, as in MCBO with ``objective='both'``, by::

    lut / ref_lut  +  level / ref_level

where ``ref_lut`` / ``ref_level`` come from the ``resyn2`` reference sequence (computed
once per circuit and cached on disk) and the best intermediate ``print_stats`` stage is
kept (MCBO's ``return_best_intermediate=True`` default). MCBO minimizes that ratio sum;
``evaluate()`` returns its negation because BoCoDe maximizes. A sequence that makes ABC
crash scores the reference value ``2.0`` (MCBO's own fallback). Roughly 0.1-3 s per
evaluation depending on circuit size.

**Setup.** These problems drive an external tool chain that is far too large to ship with
BoCoDe, so three environment variables must point at it::

    BOCODE_EPFL_CIRCUITS  # directory of EPFL .blif circuits
                          #   https://github.com/lsils/benchmarks
    BOCODE_ABC_BIN        # a compiled Berkeley ABC binary
                          #   https://github.com/berkeley-abc/abc
    BOCODE_MCBO_ROOT      # an MCBO checkout (its EDA utility modules are imported;
                          #   the MCBO package itself is never imported)
    BOCODE_EPFL_RESULTS   # optional; where ABC's reference cache is written
                          #   (default: <circuits>/../epfl_results)

Until they are set, constructing a problem and reading its bounds works, and evaluating
raises an ``ImportError`` that names the three variables.

Sources:
K. Dreczkowski, A. Grosnit, H. Bou-Ammar. Framework and Benchmarks for Combinatorial and Mixed-variable Bayesian Optimization. Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks), 2023. arXiv:2306.09803. https://github.com/huawei-noah/HEBO/tree/master/MCBO
A. Grosnit, C. Malherbe, R. Tutunov, X. Wan, J. Wang, H. Bou-Ammar. BOiLS: Bayesian Optimisation for Logic Synthesis. Design, Automation and Test in Europe (DATE), 2022.
L. Amaru, P.-E. Gaillardon, G. De Micheli. The EPFL Combinational Benchmark Suite. Proceedings of the 24th International Workshop on Logic and Synthesis (IWLS), 2015. https://github.com/lsils/benchmarks
R. Brayton, A. Mishchenko. ABC: An Academic Industrial-Strength Verification Tool. Computer Aided Verification (CAV), 2010. https://github.com/berkeley-abc/abc
"""

from __future__ import annotations

from ._epfl_base import EPFLSeqOptProblem

_TAGS = {"single_objective", "unconstrained", "56D", "mixed", "logic_synthesis"}


class AIG_adder(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL arithmetic circuit ``adder`` (128-bit adder), 56-D mixed."""

    tags = _TAGS

    circuit = "adder"


class AIG_bar(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL arithmetic circuit ``bar`` (barrel shifter), 56-D mixed."""

    tags = _TAGS

    circuit = "bar"


class AIG_div(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL arithmetic circuit ``div`` (128-bit divisor), 56-D mixed."""

    tags = _TAGS

    circuit = "div"


class AIG_hyp(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL arithmetic circuit ``hyp`` (hypotenuse (the largest EPFL circuit)), 56-D mixed."""

    tags = _TAGS

    circuit = "hyp"


class AIG_log2(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL arithmetic circuit ``log2`` (binary logarithm), 56-D mixed."""

    tags = _TAGS

    circuit = "log2"


class AIG_max(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL arithmetic circuit ``max`` (max of four 128-bit words), 56-D mixed."""

    tags = _TAGS

    circuit = "max"


class AIG_multiplier(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL arithmetic circuit ``multiplier`` (64-bit multiplier), 56-D mixed."""

    tags = _TAGS

    circuit = "multiplier"


class AIG_sin(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL arithmetic circuit ``sin`` (sine), 56-D mixed."""

    tags = _TAGS

    circuit = "sin"


class AIG_sqrt(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL arithmetic circuit ``sqrt`` (square root), 56-D mixed."""

    tags = _TAGS

    circuit = "sqrt"


class AIG_square(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL arithmetic circuit ``square`` (squarer), 56-D mixed."""

    tags = _TAGS

    circuit = "square"


class AIG_arbiter(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL control circuit ``arbiter`` (round-robin arbiter), 56-D mixed."""

    tags = _TAGS

    circuit = "arbiter"


class AIG_cavlc(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL control circuit ``cavlc`` (CAVLC coding block), 56-D mixed."""

    tags = _TAGS

    circuit = "cavlc"


class AIG_ctrl(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL control circuit ``ctrl`` (ALU control unit), 56-D mixed."""

    tags = _TAGS

    circuit = "ctrl"


class AIG_dec(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL control circuit ``dec`` (128-bit decoder), 56-D mixed."""

    tags = _TAGS

    circuit = "dec"


class AIG_i2c(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL control circuit ``i2c`` (I2C controller), 56-D mixed."""

    tags = _TAGS

    circuit = "i2c"


class AIG_int2float(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL control circuit ``int2float`` (integer-to-float converter), 56-D mixed."""

    tags = _TAGS

    circuit = "int2float"


class AIG_mem_ctrl(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL control circuit ``mem_ctrl`` (memory controller), 56-D mixed."""

    tags = _TAGS

    circuit = "mem_ctrl"


class AIG_priority(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL control circuit ``priority`` (priority encoder), 56-D mixed."""

    tags = _TAGS

    circuit = "priority"


class AIG_router(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL control circuit ``router`` (lookahead router), 56-D mixed."""

    tags = _TAGS

    circuit = "router"


class AIG_voter(EPFLSeqOptProblem):
    """ABC synthesis-sequence optimization for the EPFL control circuit ``voter`` (majority voter), 56-D mixed."""

    tags = _TAGS

    circuit = "voter"
