"""OptimizationSpec: a serializable description of a topology optimization problem (plan S3).

The spec is what the GUI edits and the project file stores: objective and constraints as expressions over the response
library, the method and its settings, filtering, manufacturing options and named selections. compile_spec() turns it
into the TOParams the drivers already understand, choosing for each part

  * the built-in QOI type when the expression is exactly a built-in quantity (Compliance(), VolumeFraction(), Mass(),
    StressPNorm(), and the stress constraints MaxStress() / StressFailureFactor()), so OC/Pareto/LevelSet stay
    available for classical problems and benchmarks keep their tested behaviour;
  * otherwise a TO_QOI.GFUNCTION built by the expression compiler, differentiated by autograd (MMA).

validate() reports problems before a run as a list of Issue(level, message).
"""
import copy
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from pyto.topopt.common import GRADIENT_METHOD, TOParams, TO_QOI, gradient_method
from pyto.autodiff.qoi.responses import RESPONSES
from pyto.topopt.expressions import ExpressionError, compile_expression

SPEC_VERSION = 1
METHODS = ("MMA", "OC", "PARETO", "LEVELSET")
ACTIVE_METHODS = ("MMA", "OC")   # offered in the GUI and run by the default benchmark sweep; PARETO and LEVELSET
                                 # remain available from code and specs, but are secondary methods now

# Expressions that map exactly onto a built-in QOI (spaces ignored).
_NATIVE_OBJECTIVES = {"Compliance()": TO_QOI.COMPLIANCE, "VolumeFraction()": TO_QOI.VOLUME_FRACTION,
                      "Mass()": TO_QOI.MASS, "StressPNorm()": TO_QOI.PNORM_STRESS}
_NATIVE_CONSTRAINTS_LE = {"Compliance()": TO_QOI.COMPLIANCE, "VolumeFraction()": TO_QOI.VOLUME_FRACTION,
                          "Mass()": TO_QOI.MASS, "StressPNorm()": TO_QOI.PNORM_STRESS,
                          "MaxStress()": TO_QOI.MAX_VONMISES_STRESS,
                          "StressFailureFactor()": TO_QOI.STRESS_FAILURE_FACTOR}
_NATIVE_CONSTRAINTS_GE = {"VolumeFraction()": TO_QOI.VOLUME_FRACTION_MIN}
# Built-in stress constraints with adaptive (stateful) normalization: only usable as a whole constraint.
_CONSTRAINT_ONLY = {"MaxStress()"}


def _key(expression: str) -> str:
    return expression.replace(" ", "")


PYTHON_PREFIX = "python:"


def load_python_function(reference: str):
    """Load fn(sol, x, fe_solver) from 'python:<path to .py>::<function name>' (power-user objectives/constraints).

    This executes the file, so callers must only do it with the user's consent (compile_spec(..., allow_code=True))."""
    import importlib.util
    import os
    body = reference[len(PYTHON_PREFIX):]
    if "::" not in body:
        raise ExpressionError(f"Python reference must look like python:<file.py>::<function>, got {reference!r}.")
    path, name = body.rsplit("::", 1)
    if not os.path.isfile(path):
        raise ExpressionError(f"Python file not found: {path}")
    module_spec = importlib.util.spec_from_file_location(f"pyto_user_{abs(hash(path))}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    fn = getattr(module, name, None)
    if not callable(fn):
        raise ExpressionError(f"{path} defines no function {name!r}.")
    return fn


def _compile_any(expression, selections, allow_code):
    """(fn, depends_on_solution) for an expression or a python: reference."""
    if expression.strip().startswith(PYTHON_PREFIX):
        if not allow_code:
            raise ExpressionError("This formulation calls a Python file; loading it runs code and needs confirmation "
                                  "(allow_code=True).")
        return load_python_function(expression.strip()), True
    e = compile_expression(expression, selections)
    return e.fn, e.depends_on_solution


@dataclass
class ObjectiveSpec:
    expression: str = "Compliance()"
    sense: str = "minimize"                      # minimize | maximize


@dataclass
class ConstraintSpec:
    expression: str = "VolumeFraction()"
    op: str = "<="                               # <= | >=
    bound: float = 0.5
    enabled: bool = True


@dataclass
class MethodSpec:
    name: str = "MMA"
    max_iterations: int = 150
    move_limit: float = 0.2
    gradient: str = "autodiff"                   # "autodiff" or "analytical" (MMA/OC only; old name "manual")


@dataclass
class RegularizationSpec:
    filter_radius: float = 1.5                   # relative to the element size
    heaviside: bool = False


@dataclass
class ManufacturingSpec:
    extrude: str = ""                            # "", "X", "Y", "Z"
    symmetry: list = field(default_factory=list)  # subset of ["X", "Y", "Z"]
    cyclic_z: int = 0                            # number of sectors about Z (0 = off)
    keep_selection: str = ""                     # name of an element selection kept solid


@dataclass
class OptimizationSpec:
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    constraints: list = field(default_factory=lambda: [ConstraintSpec()])
    method: MethodSpec = field(default_factory=MethodSpec)
    regularization: RegularizationSpec = field(default_factory=RegularizationSpec)
    manufacturing: ManufacturingSpec = field(default_factory=ManufacturingSpec)
    version: int = SPEC_VERSION

    # ---------------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizationSpec":
        d = dict(d)
        version = d.pop("version", SPEC_VERSION)
        if version > SPEC_VERSION:
            raise ValueError(f"Optimization spec version {version} is newer than this PyTO ({SPEC_VERSION}).")
        return cls(objective=ObjectiveSpec(**d.get("objective", {})),
                   constraints=[ConstraintSpec(**c) for c in d.get("constraints", [])],
                   method=MethodSpec(**d.get("method", {})),
                   regularization=RegularizationSpec(**d.get("regularization", {})),
                   manufacturing=ManufacturingSpec(**d.get("manufacturing", {})))

    @classmethod
    def from_json(cls, text: str) -> "OptimizationSpec":
        return cls.from_dict(json.loads(text))

    @property
    def active_constraints(self) -> list:
        return [c for c in self.constraints if c.enabled]


@dataclass(frozen=True)
class Issue:
    level: str        # "error" (cannot run) | "warning" (runs, but check)
    message: str

    def __str__(self):
        return f"{self.level.upper()}: {self.message}"


def _compile_part(expression, selections):
    """Compiled expression, or None for constraint-only built-ins (checked separately)."""
    if _key(expression) in _CONSTRAINT_ONLY:
        return None
    return compile_expression(expression, selections)


def validate(spec: OptimizationSpec, physics: str, selections: dict = None, allow_code: bool = False) -> list:
    """Problems with the spec for a given physics ('structural', 'thermal', 'thermo-structural')."""
    issues = []
    err = lambda m: issues.append(Issue("error", m))
    warn = lambda m: issues.append(Issue("warning", m))
    compiled = {}

    if spec.objective.sense not in ("minimize", "maximize"):
        err(f"Objective sense must be minimize or maximize, not {spec.objective.sense!r}.")
    parts = [("objective", spec.objective.expression)] + \
            [(f"constraint {i + 1}", c.expression) for i, c in enumerate(spec.constraints) if c.enabled]
    for where, text in parts:
        if where == "objective" and _key(text) in _CONSTRAINT_ONLY:
            err(f"{text} can only be used as a constraint (it normalizes itself between iterations).")
            continue
        if "MaxStress(" in _key(text) and _key(text) not in _CONSTRAINT_ONLY:
            err(f"{where}: MaxStress() can only be used on its own as a constraint; use StressPNorm() in formulas.")
            continue
        if text.strip().startswith(PYTHON_PREFIX):
            if allow_code:
                try:
                    load_python_function(text.strip())
                except ExpressionError as ex:
                    err(f"{where}: {ex}")
            else:
                warn(f"{where}: calls a Python file; it runs only after you confirm (it executes code).")
            continue
        try:
            e = _compile_part(text, selections)
        except ExpressionError as ex:
            err(f"{where}: {ex}")
            continue
        compiled[where] = e
        if e is not None and physics not in e.physics:
            missing = sorted(r for r in e.responses if physics not in RESPONSES[r].physics)
            err(f"{where}: {', '.join(missing)} not available for {physics} problems.")
        if e is not None and not e.responses:
            err(f"{where}: {text!r} does not depend on the design.")

    for i, c in enumerate(spec.constraints):
        if not c.enabled:
            continue
        if c.op not in ("<=", ">="):
            err(f"constraint {i + 1}: operator must be <= or >=, not {c.op!r}.")
        if not np.isfinite(c.bound):
            err(f"constraint {i + 1}: bound must be a finite number.")
        elif c.bound <= 0:
            if c.op == ">=":
                err(f"constraint {i + 1}: a >= bound must be > 0 (it is normalized as 1 - value/bound).")
            else:
                warn(f"constraint {i + 1}: bound {c.bound} <= 0 cannot be normalized; the constraint is used as "
                     f"value - bound <= 0, which MMA may scale poorly.")
    if spec.method.name not in METHODS:
        err(f"Unknown method {spec.method.name!r} (choose from {', '.join(METHODS)}).")
    try:
        gradient_method(spec.method.gradient)
    except ValueError:
        err(f"Unknown gradient {spec.method.gradient!r} (choose autodiff or analytical).")
    if spec.method.max_iterations < 1:
        err("Maximum iterations must be at least 1.")
    if not 0 < spec.method.move_limit <= 1:
        err("Move limit must be in (0, 1].")
    if spec.regularization.filter_radius <= 0:
        err("Filter radius must be > 0.")
    if spec.manufacturing.keep_selection and spec.manufacturing.keep_selection not in (selections or {}):
        err(f"Keep-solid selection {spec.manufacturing.keep_selection!r} is not defined.")

    # well-posedness
    obj = _key(spec.objective.expression)
    limits_material = any(_key(c.expression) in ("VolumeFraction()", "Mass()") and c.op == "<="
                          for c in spec.active_constraints)
    if spec.objective.sense == "minimize" and obj in ("Compliance()", "StrainEnergy()") and not limits_material:
        warn("Minimizing compliance without a volume or mass limit: the optimum is the full design.")
    if spec.objective.sense == "minimize" and obj in ("VolumeFraction()", "Mass()") and not spec.active_constraints:
        err("Minimizing volume/mass without constraints: the optimum is an empty design. Add a performance constraint.")
    if spec.objective.sense == "maximize" and obj in ("Compliance()",):
        warn("Maximizing compliance makes the structure as flexible as possible; check that this is intended.")
    seen = set()
    for c in spec.active_constraints:
        k = (_key(c.expression), c.op)
        if k in seen:
            warn(f"Constraint {c.expression} {c.op} appears more than once.")
        seen.add(k)
    return issues


def compile_spec(spec: OptimizationSpec, selections: dict = None, base: TOParams = None,
                 allow_code: bool = False) -> TOParams:
    """TOParams for the drivers. Raises ValueError listing the errors if validate() would report any.

    `base` (optional) supplies settings the spec does not cover (e.g. nDOFDesired, materialModel)."""
    to_params = copy.deepcopy(base) if base is not None else TOParams()
    selections = selections or {}

    obj_key = _key(spec.objective.expression)
    if spec.objective.sense == "minimize" and obj_key in _NATIVE_OBJECTIVES:
        to_params.Objective = (_NATIVE_OBJECTIVES[obj_key], None)
    else:
        fn, _ = _compile_any(spec.objective.expression, selections, allow_code)
        fn = fn if spec.objective.sense == "minimize" else _negated(fn)
        to_params.Objective = (TO_QOI.GFUNCTION, fn)

    constraints = []
    for c in spec.active_constraints:
        k = _key(c.expression)
        if c.op == ">=" and c.bound <= 0:        # 1 - f/b <= 0 would flip the inequality
            raise ValueError(f"{c.expression} >= {c.bound}: a >= bound must be > 0.")
        if c.op == "<=" and k in _NATIVE_CONSTRAINTS_LE and c.bound > 0:
            constraints.append((_NATIVE_CONSTRAINTS_LE[k], None, float(c.bound)))
        elif c.op == ">=" and k in _NATIVE_CONSTRAINTS_GE:
            constraints.append((_NATIVE_CONSTRAINTS_GE[k], None, float(c.bound)))
        else:
            if k in _CONSTRAINT_ONLY:
                raise ValueError(f"{c.expression} is only supported as '<= bound' with bound > 0.")
            fn, _ = _compile_any(c.expression, selections, allow_code)
            if c.op == "<=" and c.bound > 0:
                constraints.append((TO_QOI.GFUNCTION, fn, float(c.bound)))
            elif c.op == "<=":
                constraints.append((TO_QOI.GFUNCTION, _shifted(fn, c.bound), None))       # f - b <= 0
            else:
                constraints.append((TO_QOI.GFUNCTION, _ge_normalized(fn, c.bound), None))  # 1 - f/b <= 0
    to_params.Constraints = constraints
    to_params.ObjectiveLabel = (spec.objective.expression, spec.objective.sense)
    to_params.ConstraintLabels = [(c.expression, c.op, float(c.bound)) for c in spec.active_constraints]

    to_params.MaxIterations = int(spec.method.max_iterations)
    to_params.Gradient = gradient_method(spec.method.gradient)
    to_params.RelativeFilterRadius = float(spec.regularization.filter_radius)
    to_params.HeavisideProjection = bool(spec.regularization.heaviside)
    m = spec.manufacturing
    to_params.ExtrudeX, to_params.ExtrudeY, to_params.ExtrudeZ = (m.extrude == "X", m.extrude == "Y", m.extrude == "Z")
    to_params.XSymmetry, to_params.YSymmetry, to_params.ZSymmetry = ("X" in m.symmetry, "Y" in m.symmetry,
                                                                   "Z" in m.symmetry)
    to_params.ZAxisAngularSymmetry = int(m.cyclic_z)
    if m.keep_selection:
        to_params.ElemsToKeep = list(np.asarray(selections[m.keep_selection]["elements"]))
    return to_params


def _negated(fn):
    g = lambda sol, x, fe: -fn(sol, x, fe)
    g.__name__ = f"maximize[{getattr(fn, '__name__', 'f')}]"
    return g


def _shifted(fn, bound):
    g = lambda sol, x, fe: fn(sol, x, fe) - bound
    g.__name__ = f"{getattr(fn, '__name__', 'f')} - {bound}"
    return g


def _ge_normalized(fn, bound):
    g = lambda sol, x, fe: 1.0 - fn(sol, x, fe) / bound
    g.__name__ = f"1 - {getattr(fn, '__name__', 'f')}/{bound}"
    return g


# ---------------------------------------------------------------------------------------------- benchmark round trip
_REVERSE = {TO_QOI.COMPLIANCE: "Compliance()", TO_QOI.VOLUME_FRACTION: "VolumeFraction()", TO_QOI.MASS: "Mass()",
            TO_QOI.PNORM_STRESS: "StressPNorm()", TO_QOI.MAX_VONMISES_STRESS: "MaxStress()",
            TO_QOI.STRESS_FAILURE_FACTOR: "StressFailureFactor()"}


def spec_from_to_params(to_params: TOParams, method: str = "MMA", num_dofs: int = None):
    """Describe an existing (benchmark) TOParams as a spec. Returns (spec, selections).

    GVECTOR objectives with a 0/1 vector on one displacement component become Displacement(GOutput, <c>, sum);
    other GVECTOR weights and GFUNCTION callables cannot be represented and raise NotImplementedError."""
    selections = {}
    otype, oparam = to_params.Objective[0], to_params.Objective[1]
    if otype in _REVERSE and otype not in (TO_QOI.MAX_VONMISES_STRESS, TO_QOI.STRESS_FAILURE_FACTOR):
        objective = ObjectiveSpec(_REVERSE[otype])
    elif otype == TO_QOI.GVECTOR:
        g = np.asarray(oparam, dtype=float)
        dofs = np.nonzero(g)[0]
        comps = np.unique(dofs % 3)
        if len(dofs) == 0 or not np.allclose(g[dofs], 1.0) or len(comps) != 1:
            raise NotImplementedError("Only 0/1 GVECTORs on a single displacement component can be written as a spec.")
        selections["GOutput"] = {"nodes": dofs // 3}
        objective = ObjectiveSpec(f"Displacement(GOutput, {'xyz'[comps[0]]}, sum)")
    else:
        raise NotImplementedError(f"Objective {otype} cannot be written as a spec.")
    constraints = []
    for ctype, _param, limit in to_params.Constraints:
        if ctype == TO_QOI.VOLUME_FRACTION_MIN:
            constraints.append(ConstraintSpec("VolumeFraction()", ">=", float(limit)))
        elif ctype in _REVERSE:
            constraints.append(ConstraintSpec(_REVERSE[ctype], "<=", float(limit)))
        else:
            raise NotImplementedError(f"Constraint {ctype} cannot be written as a spec.")
    m = ManufacturingSpec(
        extrude="X" if to_params.ExtrudeX else "Y" if to_params.ExtrudeY else "Z" if to_params.ExtrudeZ else "",
        symmetry=[a for a, on in zip("XYZ", (to_params.XSymmetry, to_params.YSymmetry, to_params.ZSymmetry)) if on],
        cyclic_z=int(to_params.ZAxisAngularSymmetry))
    if to_params.ElemsToKeep is not None:
        selections["KeepSolid"] = {"elements": np.asarray(to_params.ElemsToKeep)}
        m.keep_selection = "KeepSolid"
    spec = OptimizationSpec(objective=objective, constraints=constraints,
                            method=MethodSpec(method, int(to_params.MaxIterations),
                                              gradient=gradient_method(to_params.Gradient).value),
                            regularization=RegularizationSpec(float(to_params.RelativeFilterRadius),
                                                              bool(to_params.HeavisideProjection)),
                            manufacturing=m)
    return spec, selections
