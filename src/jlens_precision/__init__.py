"""jlens_precision - representational and causal precision of natural-language lenses.

The package answers, for a lens claim ``L_X = 1`` at a given layer:

* ``P(R_X = 1 | L_X = 1)``            representational precision
* ``P(R_X = 1, U_X = 1 | L_X = 1)``   causal precision

where ``R_X`` is *independent* evidence that the computational variable whose
value is ``X`` is represented at that layer (Stage 2 probes), and ``U_X`` is
*independent* evidence that the representation is causally used (Stage 2
natural-counterfactual interchange interventions).

Neither ``R_X`` nor ``U_X`` ever depends on a lens.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
