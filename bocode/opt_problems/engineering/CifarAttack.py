"""Casmopolitan sparse adversarial attack on CIFAR-10 — 85-D mixed (43 cat + 42 cont).

A *sparse, targeted, black-box adversarial attack*: the image is perturbed on a coarse
``14 x 14 x 3`` grid in which each of the 14 rows of each of the 3 colour channels may be
perturbed at exactly one column. The design therefore chooses 42 pixel *columns*
(categorical, 14 choices each), one *upsampling method* (categorical, 3 choices) to blow
the ``14 x 14`` grid up to ``32 x 32``, and 42 continuous perturbation *magnitudes* in
``[-1, 1]`` (scaled by ``epsilon``). The attacked CNN is Carlini & Wagner's CIFAR-10
network; the attack construction is the one used by Casmopolitan (Wan et al., ICML 2021),
which inherits it from the BayesOpt-Attack / GenAttack line of work.

Design variables (85-D — exactly Casmopolitan's ``categorical_dims`` /
``continuous_dims`` / ``n_vertices = [14]*42 + [3]``)::

    dims 0..41  : pixel column index, 14 categories (0..13)
    dim  42     : upsampling method, 3 categories (0 = BILI, 1 = NN, 2 = BICU)
    dims 43..84 : perturbation magnitude in [-1, 1] (continuous)

Fixed hyperparameters are the upstream repo's own defaults (``bbattack_wrapped.py``):
``epsilon = 0.3``, ``low_dim = 14*14 = 196``, ``high_dim = 32*32 = 1024``,
``obj_metric = 2``, ``rescale = True``, ``img_offset = 1``, target-label index 0.

**Fixed, seed-stable instance.** Following ``utilities.generate_attack_data_set`` with
``num_sample = 1``, ``img_offset = 1``, ``attack_type = 'targeted'``,
``random_target_class = None``, the CIFAR-10 test set is first filtered to the
correctly-classified images, the image at position 1 of that subset is taken, and the
target classes are ``[0..9]`` minus the true label in ascending order;
``get_data_sample(0)`` then picks the first of those. With this port's CNN (78.0 % test
accuracy) that resolves to CIFAR-10 test index 2, true label 8 (ship), target label 0
(airplane) — also reported by :meth:`CifarAttack85.instance_info`.

**Objective** (Casmopolitan ``obj_metric = 2``, minimized)::

    score = log( sum_{j != target} p_j )  -  log( p_target )

on the softmax output of the CNN for the rescaled adversarial image. ``evaluate()``
returns ``-score`` because BoCoDe maximizes, i.e. it pushes the target-class probability
up. Attack success is available via ``problem.last_success`` and from the sign of the
score (``score < 0`` iff the target class holds more mass than all other classes
combined).

*Documented deviation from the original code*: ``CNN.evaluate`` hard-overrides
``score = -1`` as soon as the attack succeeds (``argmax == target``), because their loop
stops there. That override creates a flat plateau over all successful attacks — and is
actually *worse* than the natural score once the target dominates — which would destroy
the optimization signal for the remainder of a fixed budget. We therefore always return
the continuous margin.

**Setup.** The victim CNN ships as a TF1/Keras HDF5 file in the Casmopolitan repo and the
CIFAR-10 test images as the standard binary batch; neither is bundled with BoCoDe (9.2 MB
+ 30 MB). They are fetched on demand into the BoCoDe cache (see ``bocode/_fetch.py``), or
taken from a local directory pointed at by ``BOCODE_CASMO_ATTACK_DIR``::

    $BOCODE_CASMO_ATTACK_DIR/cifar_keras_weights.h5           # Casmopolitan repo,
    #   mixed_test_func/AdvAttack/models/cifar (MIT license)
    $BOCODE_CASMO_ATTACK_DIR/cifar-10-batches-bin/test_batch.bin
    #   https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

Reading the HDF5 weights additionally needs ``h5py``
(``pip install 'bocode[casmoattack]'``). One CNN forward pass per evaluation; CPU is
plenty.

Sources:
X. Wan, V. Nguyen, H. Ha, B. Ru, C. Lu, M. A. Osborne. Think Global and Act Local: Bayesian Optimisation over High-Dimensional Categorical and Mixed Search Spaces. Proceedings of the 38th International Conference on Machine Learning, PMLR 139:10663-10674, 2021. https://github.com/xingchenwan/Casmopolitan
N. Carlini, D. Wagner. Towards Evaluating the Robustness of Neural Networks. IEEE Symposium on Security and Privacy, 2017 (the attacked CIFAR-10 CNN, nn_robust_attacks).
B. Ru, A. Cobb, A. Blaas, Y. Gal. BayesOpt Adversarial Attack. International Conference on Learning Representations, 2020 (the sparse attack construction).
"""

from __future__ import annotations

import numpy as np
import torch

from ...base import BenchmarkProblem

_EPSILON = 0.3
_LOW_DIM = 14 * 14  # 196 -> n_row = n_column = 14
_HIGH_DIM = 32 * 32  # 1024
_NCHANNEL = 3
_N_ROW = 14
_IMG_OFFSET = 1  # bbattack_wrapped.py default
_TARGET_RANK = 0  # get_data_sample(i=0): the first target class
_UPSAMPLE = ["BILI", "NN", "BICU"]  # CNN.dim_reduction_method_list order


class CifarAttack85(BenchmarkProblem):
    """Casmopolitan 85-D sparse adversarial attack on CIFAR-10 (43 categorical + 42 cont).

    Maximize ``-(log sum_{j != target} p_j - log p_target)``. See the module docstring for
    the encoding, the fixed instance, the one documented deviation from the original code
    and the data setup (``BOCODE_CASMO_ATTACK_DIR`` / on-demand download + ``h5py``).
    """

    available_dimensions = 85
    num_objectives = 1
    num_constraints = 0

    tags = {"single_objective", "unconstrained", "85D", "mixed", "adversarial"}

    def __init__(self, device: str = "cpu") -> None:
        self.variable_types = (
            [[float(i) for i in range(14)]] * 42
            + [[0.0, 1.0, 2.0]]
            + ["continuous"] * 42
        )
        super().__init__(
            dim=85,
            num_objectives=1,
            num_constraints=0,
            bounds=[(0, 13)] * 42 + [(0, 2)] + [(-1.0, 1.0)] * 42,
        )
        self._device = device
        self._state = None
        self.last_success = None

    # -- victim model + fixed instance (lazy: keeps construction cheap) ---------
    def _ensure(self):
        if self._state is None:
            from . import _cifar_cnn_torch as helper

            model = helper.load_cifar_model(device=self._device)
            images, labels = helper.load_cifar_test()
            instance = helper.pick_fixed_attack_image(
                model,
                images,
                labels,
                img_offset=_IMG_OFFSET,
                target_label_id=_TARGET_RANK,
            )
            self._state = (
                helper,
                model,
                instance,
                instance["image"][None, ...].astype(np.float64),
            )
        return self._state

    def instance_info(self) -> dict:
        """The fixed, reproducible attack instance (see the module docstring)."""
        _helper, _model, instance, _x0 = self._ensure()
        return {
            "cifar10_test_index": instance["test_index"],
            "true_label": instance["true_label"],
            "target_label": instance["target_label"],
            "img_offset": _IMG_OFFSET,
            "target_label_id": _TARGET_RANK,
            "epsilon": _EPSILON,
        }

    def _one(self, row: np.ndarray) -> float:
        """One design (natural units) -> Casmopolitan score (to minimize)."""
        helper, model, instance, x_origin = self._ensure()
        target = int(instance["target_label"])
        h = np.clip(np.rint(row[:42]).astype(int), 0, _N_ROW - 1)  # pixel columns
        up = _UPSAMPLE[int(np.clip(np.rint(row[42]), 0, 2))]
        x = np.clip(row[43:], -1.0, 1.0)  # magnitudes

        # Scatter each (channel, row) magnitude into the chosen column: this is
        # bbattack_objective.CNN.np_coca_evaluate verbatim.
        pert = np.zeros((_N_ROW, _N_ROW, _NCHANNEL), dtype=np.float64)
        h_arr = h.reshape(_NCHANNEL, _N_ROW)
        x_arr = x.reshape(_NCHANNEL, _N_ROW)
        for i in range(_NCHANNEL):
            for j in range(_N_ROW):
                pert[j, h_arr[i, j], i] = x_arr[i, j]
        delta_low = pert.reshape(1, -1) * _EPSILON
        # NOTE (inherited quirk, kept for faithfulness): ``pert`` is built HWC and
        # flattened, while upsample_projection ``.view()``s it as CHW, and its CHW output
        # is reshaped back as HWC below -- exactly as the original np_coca_evaluate +
        # CNN.evaluate do. The composite map is still a fixed, deterministic sparse
        # perturbation; we replicate it rather than "fix" it so the benchmark matches the
        # published Casmopolitan objective.
        delta = helper.upsample_projection(
            up, delta_low, _LOW_DIM, _HIGH_DIM, nchannel=_NCHANNEL
        )
        x_adv = x_origin + delta.reshape(-1, 32, 32, _NCHANNEL)
        x_adv = x_adv / (1.0 + 2.0 * _EPSILON)  # rescale=True
        with torch.no_grad():
            p = model(torch.from_numpy(x_adv.astype(np.float32)).to(self._device))
        p = p.detach().cpu().numpy().astype(np.float64).reshape(1, -1)
        p_t = p[:, target]
        p_other = p.sum(axis=1) - p_t
        self.last_success = bool(p.argmax(axis=1)[0] == target)
        return float(np.log(p_other + 1e-30) - np.log(p_t + 1e-30))

    def _evaluate_implementation(self, X: torch.Tensor, scaling: bool = False):
        if scaling:
            X = super().scale(X)
        X = self.enforce_variable_types(X.to(torch.float64))
        x = X.detach().cpu().numpy().astype(np.float64)
        out = np.array([-self._one(row) for row in x], dtype=np.float64)  # maximize
        return None, torch.tensor(out, dtype=torch.float64).reshape(-1, 1)
