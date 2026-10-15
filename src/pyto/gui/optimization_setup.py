"""Optimization formulation for the GUI (plan S5/S6, notes/6_GUI_OPTIMIZATION_FORMULATION_AND_AD_PLAN.md).

Two parts:
  * plain functions (no Qt), unit-tested in tests/test_gui_optimization_setup.py: building expressions from the
    dropdowns, migrating the old "performance" options, preparing a run (validate + compile + method support);
  * FormulationPanel, the Objective / Constraints / Method tabs embedded in the TopOpt Execute window.

Manufacturing options (extrusion, symmetry, cyclic, keep fixed faces) stay in the TopOpt Options window, which has
the 3D previews; they are applied on top of the compiled spec.
"""
from dataclasses import dataclass, field

import numpy as np

from pyto.autodiff.qoi.responses import RESPONSES, available_responses
from pyto.topopt.common import GRADIENT_METHOD, gradient_method
from pyto.topopt.capabilities import (cost_estimate, gradient_source, manual_gradient_support,
                                      method_support)
from pyto.topopt.spec import (METHODS, PYTHON_PREFIX, ConstraintSpec, Issue, MethodSpec, ObjectiveSpec,
                              OptimizationSpec, RegularizationSpec, compile_spec, validate)

# Labels shown in the dropdowns -> response names. Order = display order.
RESPONSE_LABELS = {
    "Compliance": "Compliance", "Volume fraction": "VolumeFraction", "Mass": "Mass",
    "Displacement": "Displacement", "Temperature": "Temperature", "Reaction force": "ReactionForce",
    "Strain energy": "StrainEnergy", "Stress (p-norm)": "StressPNorm", "Stress failure factor": "StressFailureFactor",
    "Mechanical compliance": "MechanicalCompliance", "Thermal compliance": "ThermalCompliance",
}
EXPRESSION_LABEL = "Expression..."
PYTHON_LABEL = "Python function..."
GRADIENT_CHOICES = [("Automatic differentiation", GRADIENT_METHOD.AUTODIFF.value), ("Analytical", GRADIENT_METHOD.ANALYTICAL.value)]
AGGREGATE_LABELS = {"mean": "mean", "p-norm (smooth max)": "pnorm", "sum": "sum"}
UNITS_NOTE = "Bounds are in SI units (m, Pa, K, N, J, kg), independent of the display unit system."


def response_labels(physics: str) -> list:
    """Dropdown labels valid for the physics, in display order."""
    ok = set(available_responses(physics))
    return [label for label, name in RESPONSE_LABELS.items() if name in ok]


def builtin_expression(label: str, selection: str = "All", component: str = "magnitude", aggregate: str = "mean") -> str:
    """Expression text for a dropdown choice, e.g. ('Displacement', 'Load1', 'y', 'pnorm') -> 'Displacement(Load1, y, pnorm)'."""
    name = RESPONSE_LABELS[label]
    if name == "Displacement":
        return f"Displacement({selection}, {component}, {aggregate})"
    if name == "Temperature":
        return f"Temperature({selection}, {aggregate})"
    if name == "ReactionForce":
        return f"ReactionForce({selection}, {component if component in 'xyz' else 'y'})"
    if name in ("VolumeFraction", "StrainEnergy") and selection not in ("", "All"):
        return f"{name}({selection})"
    return f"{name}()"


def default_spec(volume_fraction: float = 0.5) -> OptimizationSpec:
    return OptimizationSpec(objective=ObjectiveSpec("Compliance()", "minimize"),
                            constraints=[ConstraintSpec("VolumeFraction()", "<=", float(volume_fraction))])


def migrate_performance_options(topopt_options) -> list:
    """Old TopOpt Options 'performance' fields (never applied before) -> constraint rows, added DISABLED.

    They used to be ignored, so enabling them silently would change how old projects behave; the user reviews them."""
    perf = (topopt_options or {}).get("performance", {}) if isinstance(topopt_options, dict) else {}
    rows = []
    stress = perf.get("stress_safety", {})
    if isinstance(stress, dict) and stress.get("enabled") and stress.get("value"):
        rows.append(ConstraintSpec("StressFailureFactor()", "<=", 1.0 / float(stress["value"]), enabled=False))
    disp = perf.get("max_displacement", {})
    if isinstance(disp, dict) and disp.get("enabled") and disp.get("value"):
        rows.append(ConstraintSpec("Displacement(All, magnitude, pnorm)", "<=", float(disp["value"]), enabled=False))
    temp = perf.get("max_temperature", {})
    if isinstance(temp, dict) and temp.get("enabled") and temp.get("value"):
        rows.append(ConstraintSpec("Temperature(All, pnorm)", "<=", float(temp["value"]), enabled=False))
    return rows


def manufacturing_from_topopt_options(topopt_options) -> "ManufacturingSpec":
    """TopOpt Options window values -> ManufacturingSpec (extrusion, symmetry planes, cyclic sectors about Z,
    keep fixed faces as the 'FixedFaces' element selection). AM build and draw direction are collected by that window
    but no driver implements them yet; they are not applied (as before)."""
    from pyto.topopt.spec import ManufacturingSpec
    opts = topopt_options or {}
    manu, sym, other = opts.get("manufacturing", {}), opts.get("symmetry", {}), opts.get("other", {})
    m = ManufacturingSpec()
    extrude = manu.get("extrude", {})
    if isinstance(extrude, dict) and extrude.get("enabled") and extrude.get("value"):
        m.extrude = {"XDir": "X", "YDir": "Y", "ZDir": "Z"}.get(extrude["value"], "")
    m.symmetry = [a for a in "XYZ" if sym.get(f"{a.lower()}_symmetry", False)]
    cyclic = manu.get("cyclic_symmetry", {})
    if isinstance(cyclic, dict) and cyclic.get("enabled") and "(" in str(cyclic.get("value", "")):
        try:
            m.cyclic_z = int(str(cyclic["value"]).split("(")[1].split(")")[0])
        except ValueError:
            m.cyclic_z = 0
    if other.get("keep_fixed_faces", False):
        m.keep_selection = "FixedFaces"
    return m


@dataclass
class RunPlan:
    to_params: object = None                 # TOParams, or None when there are errors
    issues: list = field(default_factory=list)
    support: dict = field(default_factory=dict)
    gradient: str = ""
    cost: str = ""

    @property
    def errors(self):
        return [i for i in self.issues if i.level == "error"]


def prepare_run(spec: OptimizationSpec, physics: str, selections: dict, body_force: bool = False,
                base=None, allow_code: bool = False, fe_solver=None) -> RunPlan:
    """Validate, check that the chosen method supports the formulation, and compile. No FE solve.

    With fe_solver, an analytical gradient is also checked against the solver (conductivity law, boundary values)."""
    plan = RunPlan(issues=validate(spec, physics, selections, allow_code=allow_code),
                   support=method_support(spec, physics, body_force))
    method = spec.method.name
    if method in plan.support and not plan.support[method][0]:
        plan.issues.append(Issue("error", plan.support[method][1]))
    manual = gradient_method(spec.method.gradient) is GRADIENT_METHOD.ANALYTICAL
    if manual:
        ok, why = manual_gradient_support(spec, selections, body_force)
        if not ok:
            plan.issues.append(Issue("error", f"Analytical gradient not available: {why} Use automatic differentiation."))
    if method in METHODS:
        plan.gradient = gradient_source(method, spec.method.gradient)
    if not plan.errors:
        try:
            plan.cost = cost_estimate(spec, selections)
            plan.to_params = compile_spec(spec, selections, base=base, allow_code=allow_code)
            if manual and fe_solver is not None:
                from pyto.topopt.manual_sensitivities import manual_support
                ok, why = manual_support(plan.to_params, fe_solver)
                if not ok:
                    raise ValueError(f"Analytical gradient not available: {why} Use automatic differentiation.")
        except Exception as e:                           # compile problems not caught by validate
            plan.issues.append(Issue("error", str(e)))
            plan.to_params = None
    return plan


def display_objective(spec: OptimizationSpec, history_values) -> list:
    """The drivers store the minimized value; for 'maximize' the user expects to see +f, not -f."""
    sign = -1.0 if spec.objective.sense == "maximize" else 1.0
    return [sign * float(np.ravel(v)[0]) for v in history_values]


def final_design_summary(fe_solver, physics: str, threshold: float = 0.5) -> dict:
    """Evaluate the optimized design: stresses/temperatures on the solid region (density > threshold).

    Uses the solver's last solve (the final design), runs postprocess() for the stresses, and returns
    {'solid_fraction', 'max_displacement', 'max_von_mises', 'yield_ratio', 'max_temperature',
     'element_temperature', 'lines'} (entries that do not apply to the physics are None)."""
    import torch
    mesh = fe_solver.mesh
    density = np.asarray(mesh.elemPseudoDensity)
    solid = density > threshold
    if not solid.any():
        solid = np.ones_like(density, dtype=bool)
    solid_nodes = np.unique(mesh.elemArray[solid][:, :8])
    out = {"solid_fraction": float(solid.mean()), "max_displacement": None, "max_von_mises": None,
           "yield_ratio": None, "max_temperature": None, "element_temperature": None}
    lines = [f"Final design: {100 * out['solid_fraction']:.1f} % of the elements are solid (density > {threshold})."]
    if physics in ("structural", "thermo-structural"):
        fe_solver.postprocess()
        disp = fe_solver.deformation.detach().cpu().numpy() if hasattr(fe_solver.deformation, "detach") \
            else np.asarray(fe_solver.deformation)
        out["max_displacement"] = float(disp[solid_nodes].max())
        out["max_von_mises"] = float(np.asarray(fe_solver.vonMisesStress)[solid].max())
        ys = float(getattr(fe_solver.mat_prop, "yield_strength", 0) or 0)
        out["yield_ratio"] = out["max_von_mises"] / ys if ys > 0 else None
        lines.append(f"Max displacement (solid region): {out['max_displacement']:.4g} m")
        lines.append(f"Max von Mises stress (solid region): {out['max_von_mises']:.4g} Pa"
                     + (f" = {out['yield_ratio']:.2f} x yield strength" if out["yield_ratio"] is not None else ""))
    if physics in ("thermal", "thermo-structural"):
        T = fe_solver.temperature if physics == "thermo-structural" else fe_solver.sol
        T = T.detach().cpu().numpy() if isinstance(T, torch.Tensor) else np.asarray(T)
        out["element_temperature"] = T[mesh.elemArray[:, :8]].mean(axis=1)
        out["max_temperature"] = float(T[solid_nodes].max())
        lines.append(f"Max temperature (solid region): {out['max_temperature']:.4g}")
    out["lines"] = lines
    return out


def project_entries(spec, body_force) -> dict:
    """Keys added to a saved project: the formulation (versioned) and the body force (never saved before)."""
    out = {"body_force": dict(body_force) if body_force else None}
    out["optimization"] = spec.to_dict() if spec is not None else None
    return out


def restore_from_project(project_data: dict):
    """(spec or None, body_force or None) from a loaded project.

    New projects carry 'optimization'. Older projects only had 'topopt_options'; their performance fields were never
    applied, so they come back as DISABLED constraint rows next to the default compliance / volume formulation, for
    the user to review. A project without either returns spec None (the window then uses the default)."""
    body_force = project_data.get("body_force") or None
    if project_data.get("optimization"):
        return OptimizationSpec.from_dict(project_data["optimization"]), body_force
    migrated = migrate_performance_options(project_data.get("topopt_options"))
    if migrated:
        spec = default_spec()
        spec.constraints += migrated
        return spec, body_force
    return None, body_force


# ------------------------------------------------------------------------------------------------ Qt panel
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:                                      # the plain functions above work without Qt
    QtWidgets = None

if QtWidgets is not None:

    class FormulationPanel(QtWidgets.QTabWidget):
        """Objective / Constraints / Method tabs. set_context() must be called with the problem's physics and
        selections before use; get_spec() returns the OptimizationSpec the widgets describe."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.physics, self.selections, self.body_force = "structural", {}, False
            self._build_objective_tab()
            self._build_constraints_tab()
            self._build_method_tab()
            self.set_spec(default_spec())

        # -------------------------------------------------------------- objective
        def _build_objective_tab(self):
            w = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(w)
            self.sense_combo = QtWidgets.QComboBox()
            self.sense_combo.addItems(["Minimize", "Maximize"])
            self.type_combo = QtWidgets.QComboBox()
            self.selection_combo = QtWidgets.QComboBox()
            self.component_combo = QtWidgets.QComboBox()
            self.component_combo.addItems(["magnitude", "x", "y", "z"])
            self.aggregate_combo = QtWidgets.QComboBox()
            self.aggregate_combo.addItems(list(AGGREGATE_LABELS))
            self.expression_edit = QtWidgets.QLineEdit()
            self.expression_edit.setPlaceholderText("e.g. Displacement(Load1, y, mean) / 1e-3 + 0.1 * VolumeFraction()")
            self.python_button = QtWidgets.QPushButton("Choose file...")
            self.python_button.clicked.connect(self._choose_python_file)
            form.addRow("Sense", self.sense_combo)
            form.addRow("Quantity", self.type_combo)
            form.addRow("Selection", self.selection_combo)
            form.addRow("Component", self.component_combo)
            form.addRow("Aggregate", self.aggregate_combo)
            form.addRow("Expression", self.expression_edit)
            form.addRow("", self.python_button)
            help_label = QtWidgets.QLabel(
                "Built-in quantities fill in the expression. Choose 'Expression...' to write your own formula from "
                "the quantities above (autograd differentiates it; no gradient needed). Use p-norm for a smooth "
                "maximum.")
            help_label.setWordWrap(True)
            form.addRow(help_label)
            for combo in (self.type_combo, self.selection_combo, self.component_combo, self.aggregate_combo):
                combo.currentIndexChanged.connect(self._objective_widgets_changed)
            self.addTab(w, "Objective")

        def _objective_widgets_changed(self, *_):
            label = self.type_combo.currentText()
            custom = label in (EXPRESSION_LABEL, PYTHON_LABEL)
            name = RESPONSE_LABELS.get(label)
            params = RESPONSES[name].params if name else {}
            self.selection_combo.setEnabled(bool({"nodes", "elements"} & set(params)))
            self.component_combo.setEnabled("component" in params)
            self.aggregate_combo.setEnabled("aggregate_by" in params)
            self.expression_edit.setReadOnly(not custom or label == PYTHON_LABEL)
            self.python_button.setVisible(label == PYTHON_LABEL)
            if not custom and label:
                self.expression_edit.setText(builtin_expression(
                    label, self.selection_combo.currentText() or "All", self.component_combo.currentText(),
                    AGGREGATE_LABELS[self.aggregate_combo.currentText()]))

        def _choose_python_file(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Python file with fn(sol, x, fe_solver)", "",
                                                            "Python (*.py)")
            if path:
                name, ok = QtWidgets.QInputDialog.getText(self, "Function name", "Function in the file:",
                                                          text="objective")
                if ok and name:
                    self.expression_edit.setText(f"{PYTHON_PREFIX}{path}::{name}")

        # -------------------------------------------------------------- constraints
        def _build_constraints_tab(self):
            w = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(w)
            self.constraint_table = QtWidgets.QTableWidget(0, 4)
            self.constraint_table.setHorizontalHeaderLabels(["On", "Quantity / expression", "", "Bound"])
            header = self.constraint_table.horizontalHeader()
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
            v.addWidget(self.constraint_table)
            row = QtWidgets.QHBoxLayout()
            self.add_quantity_combo = QtWidgets.QComboBox()
            add_btn = QtWidgets.QPushButton("Add")
            add_btn.clicked.connect(lambda: self._add_constraint_row(ConstraintSpec(
                self._quantity_for_new_row(), "<=", 1.0)))
            remove_btn = QtWidgets.QPushButton("Remove selected")
            remove_btn.clicked.connect(self._remove_constraint_rows)
            row.addWidget(self.add_quantity_combo, 1)
            row.addWidget(add_btn)
            row.addWidget(remove_btn)
            v.addLayout(row)
            note = QtWidgets.QLabel(UNITS_NOTE + " MaxStress() is also available as a constraint.")
            note.setWordWrap(True)
            v.addWidget(note)
            self.addTab(w, "Constraints")

        def _quantity_for_new_row(self):
            label = self.add_quantity_combo.currentText()
            if label == EXPRESSION_LABEL or not label:
                return "VolumeFraction()"
            return builtin_expression(label, "All", "magnitude", "pnorm")

        def _add_constraint_row(self, c: ConstraintSpec):
            r = self.constraint_table.rowCount()
            self.constraint_table.insertRow(r)
            on = QtWidgets.QTableWidgetItem()
            on.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            on.setCheckState(QtCore.Qt.Checked if c.enabled else QtCore.Qt.Unchecked)
            self.constraint_table.setItem(r, 0, on)
            self.constraint_table.setItem(r, 1, QtWidgets.QTableWidgetItem(c.expression))
            op = QtWidgets.QComboBox()
            op.addItems(["<=", ">="])
            op.setCurrentText(c.op)
            self.constraint_table.setCellWidget(r, 2, op)
            self.constraint_table.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{c.bound:g}"))

        def _remove_constraint_rows(self):
            for r in sorted({i.row() for i in self.constraint_table.selectedIndexes()}, reverse=True):
                self.constraint_table.removeRow(r)

        # -------------------------------------------------------------- method
        def _build_method_tab(self):
            w = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(w)
            self.method_combo = QtWidgets.QComboBox()
            self.method_combo.addItems(list(METHODS))
            self.method_combo.currentIndexChanged.connect(self.refresh_method_info)
            self.gradient_combo = QtWidgets.QComboBox()
            for text, key in GRADIENT_CHOICES:
                self.gradient_combo.addItem(text, key)
            self.gradient_combo.currentIndexChanged.connect(self.refresh_method_info)
            self.gradient_label = QtWidgets.QLabel()
            self.gradient_label.setWordWrap(True)
            self.iterations_spin = QtWidgets.QSpinBox()
            self.iterations_spin.setRange(1, 5000)
            self.move_spin = QtWidgets.QDoubleSpinBox()
            self.move_spin.setRange(0.01, 1.0)
            self.move_spin.setSingleStep(0.05)
            self.filter_spin = QtWidgets.QDoubleSpinBox()
            self.filter_spin.setRange(1.0, 20.0)
            self.filter_spin.setSingleStep(0.5)
            self.heaviside_check = QtWidgets.QCheckBox("Heaviside projection (MMA; sharper 0/1 designs)")
            self.reference_temp_spin = QtWidgets.QDoubleSpinBox()
            self.reference_temp_spin.setRange(-500.0, 5000.0)
            self.reference_temp_spin.setValue(300.0)       # same default as the Analysis window's zero-strain T (K)
            self.reference_temp_spin.setToolTip("Stress-free temperature for thermo-structural problems, in the same "
                                                "unit as the thermal loads (the GUI's temperature unit)")
            self.cost_label = QtWidgets.QLabel()
            self.cost_label.setWordWrap(True)
            self.check_gradient_button = QtWidgets.QPushButton("Check gradient (finite differences)")
            form.addRow("Method", self.method_combo)
            form.addRow("Gradient", self.gradient_combo)
            form.addRow("", self.gradient_label)
            form.addRow("Max iterations", self.iterations_spin)
            form.addRow("Move limit", self.move_spin)
            form.addRow("Filter radius (elements)", self.filter_spin)
            form.addRow("", self.heaviside_check)
            form.addRow("Stress-free temperature", self.reference_temp_spin)
            form.addRow("Cost", self.cost_label)
            form.addRow("", self.check_gradient_button)
            self.addTab(w, "Method")

            # Per-iteration optimizer output (what the drivers print), filled during a run
            self.log_view = QtWidgets.QPlainTextEdit()
            self.log_view.setReadOnly(True)
            self.log_view.setMaximumBlockCount(20000)         # keeps very long runs light
            self.log_view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            self.log_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
            self.addTab(self.log_view, "Iterations")

        def refresh_method_info(self, *_):
            """Enable only methods that support the current formulation; show the gradient source and cost."""
            try:
                spec = self.get_spec()
            except ValueError:
                return
            support = method_support(spec, self.physics, self.body_force)
            model = self.method_combo.model()
            for i, m in enumerate(METHODS):
                ok, reason = support[m]
                item = model.item(i)
                item.setEnabled(ok)
                item.setToolTip(reason or gradient_source(m))
            if not support[self.method_combo.currentText()][0]:
                self.method_combo.setCurrentText("MMA")
            method = self.method_combo.currentText()
            manual_ok, manual_why = manual_gradient_support(spec, self.selections, self.body_force)
            manual_item = self.gradient_combo.model().item(1)
            self.gradient_combo.blockSignals(True)
            manual_item.setEnabled(manual_ok)
            manual_item.setToolTip(manual_why)
            if not manual_ok:
                self.gradient_combo.setCurrentIndex(0)
            self.gradient_combo.setToolTip(manual_why or "Analytical: compliance, p-norm stress and volume fraction; no autograd graph.")
            self.gradient_combo.blockSignals(False)
            self.gradient_label.setText(gradient_source(method, self.gradient_combo.currentData()))
            try:
                self.cost_label.setText(cost_estimate(spec, self.selections))
            except Exception as e:                          # invalid expression: shown by Validate
                self.cost_label.setText(f"(fix the formulation first: {e})")
            self.reference_temp_spin.setEnabled(self.physics == "thermo-structural")

        # -------------------------------------------------------------- context and spec
        def set_context(self, physics: str, selections: dict, body_force: bool = False):
            self.physics, self.selections, self.body_force = physics, dict(selections), bool(body_force)
            current = self.type_combo.currentText()
            labels = response_labels(physics) + [EXPRESSION_LABEL, PYTHON_LABEL]
            for combo in (self.type_combo, self.add_quantity_combo):
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(labels if combo is self.type_combo else response_labels(physics) + [EXPRESSION_LABEL])
                combo.blockSignals(False)
            self.selection_combo.blockSignals(True)
            self.selection_combo.clear()
            self.selection_combo.addItems(["All"] + sorted(selections))
            self.selection_combo.blockSignals(False)
            if current in labels:
                self.type_combo.setCurrentText(current)
            self._objective_widgets_changed()
            self.refresh_method_info()

        def set_spec(self, spec: OptimizationSpec):
            self.sense_combo.setCurrentText(spec.objective.sense.capitalize())
            label = next((l for l, n in RESPONSE_LABELS.items() if f"{n}()" == spec.objective.expression.replace(" ", "")),
                         PYTHON_LABEL if spec.objective.expression.startswith(PYTHON_PREFIX) else EXPRESSION_LABEL)
            if self.type_combo.findText(label) < 0:
                self.type_combo.addItem(label)
            self.type_combo.setCurrentText(label)
            self.expression_edit.setText(spec.objective.expression)
            self.constraint_table.setRowCount(0)
            for c in spec.constraints:
                self._add_constraint_row(c)
            self.method_combo.setCurrentText(spec.method.name if spec.method.name in METHODS else "MMA")
            self.gradient_combo.setCurrentIndex(max(0, self.gradient_combo.findData(gradient_method(spec.method.gradient).value)))
            self.iterations_spin.setValue(spec.method.max_iterations)
            self.move_spin.setValue(spec.method.move_limit)
            self.filter_spin.setValue(spec.regularization.filter_radius)
            self.heaviside_check.setChecked(spec.regularization.heaviside)

        def get_spec(self) -> OptimizationSpec:
            constraints = []
            for r in range(self.constraint_table.rowCount()):
                text = self.constraint_table.item(r, 1).text().strip()
                bound_text = self.constraint_table.item(r, 3).text().strip()
                try:
                    bound = float(bound_text)
                except ValueError:
                    raise ValueError(f"Constraint {r + 1}: bound {bound_text!r} is not a number.") from None
                constraints.append(ConstraintSpec(
                    text, self.constraint_table.cellWidget(r, 2).currentText(), bound,
                    self.constraint_table.item(r, 0).checkState() == QtCore.Qt.Checked))
            return OptimizationSpec(
                objective=ObjectiveSpec(self.expression_edit.text().strip(), self.sense_combo.currentText().lower()),
                constraints=constraints,
                method=MethodSpec(self.method_combo.currentText(), self.iterations_spin.value(), self.move_spin.value(),
                                  self.gradient_combo.currentData() or "autodiff"),
                regularization=RegularizationSpec(self.filter_spin.value(), self.heaviside_check.isChecked()))

        @property
        def reference_temperature(self) -> float:
            return self.reference_temp_spin.value()
