import sys
import os
import vtk
import pyvista as pv
import numpy as np
import json
from collections import defaultdict
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from pyvistaqt import QtInteractor
from PyQt5.QtCore import pyqtSignal, QObject
from pyto.io.stl_reader import STLGeom

import pyto.core.bc as bound_cond
import pyto.core.mat_lib as mat_lib
import pyto.solve.numpy_backend as linear_solvers
from pyto.core.hex_mesher import HexMesher
from pyto.core.tet_mesher import TetMesher
import pyto.solve.deflation as deflation
from pyto.physics.thermal.hex_thermal_fea import HexThermalFEA 
import pyto.physics.structural.hex_structural_fea as hex_structural_fea
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pyto.topopt.drivers.mma import topopt_mma
from pyto.topopt.common import *
from pyto.physics.thermoelastic.topopt_thermostructural_sensitivity import ThermoElasticSensitivity
from pyto.physics.thermoelastic.thermostructural_fea import ThermoStructuralFEA
from pyto.gui.optimization_setup import (FormulationPanel, default_spec, display_objective,
                                        manufacturing_from_topopt_options, migrate_performance_options, prepare_run,
                                        project_entries, restore_from_project, final_design_summary)
from pyto.io.topopt_stl_recovery import (DEFAULT_GAUSSIAN_SIGMA, DEFAULT_SMOOTHING, extract_isosurface,
                                         recovered_volume_check, subtract_voids_from_stl)

#This is optional; requires win32com
#from solidworks_interface import SolidWorksInterface
"""


"""
DEFAULT_FONT_SIZE = 32
# Colours of the 3D view (same palette as the paper figures)
MATERIAL_COLOR = "#90EE90"   # STL geometry and recovered (optimized) STL
SUPPORT_COLOR = "#0072B2"    # structural supports
LOAD_COLOR = "#B00000"       # structural force arrows and labels
DENSITY_CMAP = LinearSegmentedColormap.from_list("pyto_density", ["white", "#3E8E3E"])
#---------------------------------------------------------------------------
def heat_flux_total(flux, triangle_areas_m2, length_units_per_m=1.0):
    """Total heat (W) of a heat flux entered per display area (W/unit^2) over STL triangles whose areas are in m^2
    (STL files are read as metres). length_units_per_m: display length units per metre (1 for MKS, 1000 for mmKS)."""
    return float(flux) * length_units_per_m ** 2 * float(np.sum(triangle_areas_m2))


def density_snapshot(mesh, threshold=0.01):
    """pyvista mesh of the current design (elements with density > threshold, cell data 'density'), for the
    per-iteration topology view. Built in the worker thread; drawn on the main thread."""
    density = np.asarray(mesh.elemPseudoDensity).copy()
    n = mesh.elemArray.shape[0]
    cells = np.column_stack((np.full(n, 8), mesh.elemArray[:, :8])).flatten()
    grid = pv.UnstructuredGrid(cells, np.full(n, pv.CellType.HEXAHEDRON, dtype=np.uint8), mesh.node_xyz)
    mask = density > threshold
    grid = grid.extract_cells(mask)
    grid.cell_data['density'] = density[mask]
    return grid


def topopt_load_problems(physics, has_supports, has_forces, has_body_force, has_thermal):
    """What is missing before a topology optimization can run, by physics (empty list = ready). Supports count in
    any form (faces, nodes, boxes)."""
    missing = []
    if physics in ("structural", "thermo-structural") and not has_supports:
        missing.append("supports (Fixed XYZ/X/Y/Z on faces or nodes)")
    if physics == "structural" and not (has_forces or has_body_force):
        missing.append("a load (force, torque or body force)")
    if physics in ("thermal", "thermo-structural") and not has_thermal:
        missing.append("thermal loads")
    return missing


def nearest_surface_nodes(points, boundary_nodes, boundary_points):
    """Nearest surface (boundary) mesh node of each point: (node indices, distances).

    Node loads/supports are stored as coordinates (STL vertices the user clicked), not mesh node numbers, so they
    survive remeshing; they are mapped to the current mesh here."""
    from scipy.spatial import cKDTree
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) == 0:
        return np.array([], dtype=int), np.array([])
    dist, idx = cKDTree(np.asarray(boundary_points)).query(pts)
    return np.asarray(boundary_nodes)[idx].astype(int), dist


def candidate_node_coordinates(main_window):
    """Coordinates the Node selection mode can pick: the surface nodes of the mesh once it exists (cached per
    mesh; get_boundary_nodes loops over all elements), otherwise the STL vertices."""
    mesh = getattr(main_window, "hex_mesh", None)
    if mesh is not None and getattr(mesh, "num_elems", 0):
        cache = getattr(main_window, "_candidate_node_cache", None)
        if cache is None or cache[0] is not mesh:
            cache = (mesh, np.asarray(mesh.node_xyz, dtype=float)[mesh.get_boundary_nodes()])
            main_window._candidate_node_cache = cache
        return cache[1]
    # Before meshing: a dense sample of the STL surface (edges <= 1/60 of the model size), so a click or a box can
    # land anywhere, not only on the few STL corners. Picks are coordinates, mapped to mesh nodes at solve time.
    stl = main_window.stl_geom
    cache = getattr(main_window, "_stl_sample_cache", None)
    if cache is None or cache[0] is not stl:
        surface = pv.PolyData(np.asarray(stl.mesh.vectors, dtype=float).reshape(-1, 3),
                              np.hstack([np.full((len(stl.mesh.vectors), 1), 3),
                                         np.arange(3 * len(stl.mesh.vectors)).reshape(-1, 3)])).clean()
        sample = surface.subdivide_adaptive(max_edge_len=surface.length / 60.0, max_n_passes=40)
        cache = (stl, np.asarray(sample.points, dtype=float))
        main_window._stl_sample_cache = cache
    return cache[1]


def frustum_to_region(frustum):
    """A box selection (vtkPlanes frustum) as plain data {'normals', 'origins'} (6 planes), for project files."""
    normals = frustum.GetNormals()
    origins = frustum.GetPoints()
    n = frustum.GetNumberOfPlanes()
    return {"normals": [list(normals.GetTuple3(i)) for i in range(n)],
            "origins": [list(origins.GetPoint(i)) for i in range(n)]}


def points_in_region(points, region):
    """Boolean mask of the points inside a stored box region: on the inner side of every plane (VTK convention:
    n . (x - o) <= 0 inside), i.e. inside the box through the whole depth."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = np.asarray(region["normals"], dtype=float)
    o = np.asarray(region["origins"], dtype=float)
    return np.all(np.einsum("pk,jk->pj", pts, n) - np.sum(n * o, axis=1)[None, :] <= 1e-12, axis=1)


def snap_to_triangle_vertex(stl_geom, cell_id, picked_point):
    """The vertex of STL triangle cell_id closest to the clicked point."""
    verts = np.asarray(stl_geom.mesh.vectors[cell_id], dtype=float)
    return verts[int(np.argmin(np.linalg.norm(verts - np.asarray(picked_point, dtype=float), axis=1)))]


def gui_elem_body_force(main_window, mesh, mat_prop):
    """Element body force from the Body force window (accelerations in m/s^2), or None if none is set.

    The Body force window used to store and draw the acceleration without it ever reaching a solver.
    """
    bf = getattr(main_window, "body_force", None) or {}
    return bound_cond.elem_body_force_from_acceleration(
        mesh, mat_prop.mass_density, [bf.get("X", 0.0), bf.get("Y", 0.0), bf.get("Z", 0.0)])


TOPOPT_FIELDS = ("Density", "Deformation", "Von Mises stress", "Temperature")


def topopt_fields_available(results):
    """Result fields that exist for the last optimization (depends on its physics)."""
    if not results or results.get("fe_solver") is None:
        return []
    physics = results.get("physics", "structural")
    return ["Density"] + (["Deformation", "Von Mises stress"] if physics != "thermal" else []) \
        + (["Temperature"] if physics != "structural" else [])


def show_topopt_field(main_window, field):
    """Draw a field of the OPTIMIZED design (its own final solve) in the main view, on the solid region
    (density > 0.5). Used by the TopOpt Execute / Postprocess result buttons and Display Options -> 'TopOpt'.
    Returns None, or a message saying why the field cannot be shown."""
    results = getattr(main_window, "topopt_results", None) or {}
    if field not in topopt_fields_available(results):
        return (f"{field} is not available: run a topology optimization first."
                if not results else f"{field} is not available for a {results.get('physics', 'structural')} optimization.")
    fe_solver = results["fe_solver"]
    physics = results.get("physics", "structural")
    if results.get("density") is not None:      # the mesh is shared with the analyses: draw the optimized design
        fe_solver.mesh.elemPseudoDensity = np.asarray(results["density"]).copy()
    plotter = main_window.plotter
    for name in list(plotter.actors.keys()):
        if name != "geometry_info":
            plotter.remove_actor(name, reset_camera=False)
    for name in list(getattr(plotter, "scalar_bars", {}).keys()):
        plotter.remove_scalar_bar(name)
    if field == "Density":
        fe_solver.plot_mesh(plotter=plotter)
    elif field == "Deformation":
        fe_solver.plot_deformation(plotter=plotter)
    elif field == "Von Mises stress":
        if not hasattr(fe_solver, "vonMisesStress"):
            fe_solver.postprocess()
        fe_solver.plot_vonMisesStress(plotter=plotter)
    elif field == "Temperature":
        summary = results.get("design_summary") or {}
        elem_T = summary.get("element_temperature")
        if elem_T is None:
            return "Temperature of the final design was not evaluated."
        thermal = fe_solver.thermal_fea if physics == "thermo-structural" else fe_solver
        thermal.plot_elem_field(elem_T, title="Temperature", plotter=plotter)
    plotter.render()
    return None


RESULT_FIELDS = ("Deformation", "Von Mises stress", "Temperature")


def result_fields_available(main_window, design):
    """Fields that can be shown for 'initial' (the Structural / Thermal Analysis results) or 'optimized'."""
    if design == "optimized":
        return [f for f in topopt_fields_available(getattr(main_window, "topopt_results", None)) if f in RESULT_FIELDS]
    out = []
    fe = getattr(main_window, "fe_solver", None)
    if fe is not None and getattr(fe, "sol", None) is not None and hasattr(fe, "vonMisesStress"):
        out += ["Deformation", "Von Mises stress"]
    th = getattr(main_window, "thermal_fe_solver", None)
    if th is not None and getattr(th, "sol", None) is not None:
        out.append("Temperature")
    return out


def show_result_field(main_window, design, field):
    """Draw a field of the initial design (analysis results, full material) or of the optimized design (its own
    final solve, solid region). The mesh is shared, and its element densities decide which elements are drawn, so
    they are set to match the design first. Returns None or a message saying why the field cannot be shown."""
    if design == "optimized":
        return show_topopt_field(main_window, field)
    if field not in result_fields_available(main_window, "initial"):
        return {"Temperature": "Run Thermal Analysis (or Structural Analysis with Include Thermal Effect) first.",
                }.get(field, "Run Structural Analysis first.")
    plotter = main_window.plotter
    for name in list(plotter.actors.keys()):
        if name != "geometry_info":
            plotter.remove_actor(name, reset_camera=False)
    for name in list(getattr(plotter, "scalar_bars", {}).keys()):
        plotter.remove_scalar_bar(name)
    solver = main_window.thermal_fe_solver if field == "Temperature" else main_window.fe_solver
    solver.mesh.elemPseudoDensity = np.ones(solver.mesh.num_elems)          # initial design: all material
    if field == "Deformation":
        solver.plot_deformation(plotter=plotter)
    elif field == "Von Mises stress":
        solver.plot_vonMisesStress(plotter=plotter)
    else:
        solver.plot_temperature(plotter=plotter)
    plotter.render()
    return None


def add_topopt_field_buttons(dialog, layout):
    """'Optimized design' buttons (one per field) on a dialog; returns {field: button}."""
    group = QtWidgets.QGroupBox("Optimized design: show")
    row = QtWidgets.QGridLayout(group)
    buttons = {}
    for i, field in enumerate(TOPOPT_FIELDS):
        b = QtWidgets.QPushButton(field)
        b.clicked.connect(lambda _checked=False, f=field: _show_field_or_warn(dialog, f))
        row.addWidget(b, i // 2, i % 2)
        buttons[field] = b
    layout.addWidget(group)
    return buttons


def refresh_topopt_field_buttons(main_window, buttons):
    available = topopt_fields_available(getattr(main_window, "topopt_results", None))
    for field, b in buttons.items():
        b.setEnabled(field in available)
        b.setToolTip("" if field in available else "Not available for the last optimization (or none run yet).")


def _show_field_or_warn(dialog, field):
    try:
        msg = show_topopt_field(dialog.parent, field)
    except Exception as e:
        msg = f"Could not show {field.lower()}: {e}"
    if msg:
        QtWidgets.QMessageBox.information(dialog, "Optimized design", msg)


class MainWindow(QtWidgets.QMainWindow):
    WINDOW_SIZE = (1280, 768)
    SIDEBAR_WIDTH = 250
    MESSAGE_HEIGHT = 162
    
    #Workflow configuration
    WORKFLOW_STEPS = {
        'geometry_loaded': {
            'enables': ['Material', 'Structural Loads', 'Thermal Loads'],
            'message': "Geometry loaded. You can now define material properties."
        },
        'material_defined': {
            'enables': [],
            'message': "Material defined. You can now apply structural or thermal loads.",
            'icon': 'check'
        },
        'structural_loads_complete': {
            'enables': ['Analysis', 'TopOpt Options'],
            'message': "Structural loads complete (forces and constraints applied). You can now run analysis or define TopOpt options.",
            'icon': 'check'
        },
        'thermal_loads_complete': {
            'enables': ['Analysis', 'TopOpt Options'],
            'message': "Thermal loads applied. You can now run analysis or define TopOpt options.",
            'icon': 'check'
        },
        'analysis_performed': {
            'message': "Analysis completed.",
            'icon': 'check'
        },
        'topopt_options_defined': {
            'enables': ['TopOpt Execute'],
            'message': "TopOpt options defined. You can now run topology optimization.",
            'icon': 'check'
        }
    }
    
    #Button configuration with workflow requirements
    BUTTON_CONFIG = [
        {"name": "Units", "icon": "arrow", "always_enabled": True, "handler": "open_units_window"},
        {"name": "Geometry", "icon": "arrow", "always_enabled": True, "handler": "open_geometry_window"},
        {"name": "Material", "icon": "cross", "requires": "geometry_loaded", "handler": "open_material_window"},
        {"name": "Structural Loads", "icon": "cross", "requires": "material_defined", "handler": "open_structural_loads_window"},
        {"name": "Thermal Loads", "icon": "cross", "requires": "material_defined", "handler": "open_thermal_loads_window"},
        {"name": "Body force", "icon": "cross", "requires": "material_defined", "handler": "open_body_force_window"},
        {"name": "Display Options", "icon": "arrow", "always_enabled": True, "handler": "open_display_options_window"},
        {"name": "Analysis", "icon": "cross", "requires": "loads_applied", "handler": "open_analysis_window"},
        {"name": "TopOpt Options", "icon": "cross", "requires": "loads_applied", "handler": "open_topopt_options_window"},
        {"name": "TopOpt Execute", "icon": "cross", "requires": "topopt_options_defined", "handler": "open_structural_topopt_window"},
        {"name": "TopOpt Postprocess", "icon": "cross", "requires": "topopt_performed", "handler": "open_topopt_results_window"},
        {"name": "Projects", "icon": "arrow", "always_enabled": True, "handler": "open_projects_window"},
        {"name": "Help", "icon": "arrow", "always_enabled": True, "handler": "show_help"}
    ]

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.init_window()
        self.init_state()
        self.init_ui()
        self.setup_visualization()
        self.connect_events()

    def init_window(self):
        """Initialize basic window properties"""
        self.setWindowTitle("PyTO")
        self.resize(*self.WINDOW_SIZE)

    def init_state(self):
        """Initialize application state and settings"""
        self.settings = Settings()
        self.LivVar = self.create_initial_state()
        
        # Initialize data containers
        self.constrained_triangles = set()
        self.constraint_actors = []
        self.constraint_data = []
        self.force_data = []
        self.force_actors = []
        self.material_data = None
        self.applied_material = None
        self.highlight_actor = None
        self.highlight_mode = 'coarse'
        self.selected_points = []        # node-mode picks (coordinates) for point loads / supports
        self.selected_regions = []       # node-mode box selections: all surface nodes inside, resolved at solve time
        self.stl_geom = None
        self.hex_mesh = None

        self.topopt_options = None
        self.optimization_spec = None  # OptimizationSpec edited in TopOpt Execute (gui/optimization_setup.py)

    def create_initial_state(self):
        """Create the initial LivVar state dictionary"""
        return {
            "geometry_loaded": False,
            "material_defined": False,
            "structural_loads": {
                "applied": False,
                "fixed_constraints": False,
                "forces_applied": False
            },
            "thermal_loads": {
                "applied": False,
                "heat_sources": False,
                "convection_applied": False
            },
            "mesh_generated": False,
            "analysis": {
                "performed": False,
                "structural": False,
                "thermal": False
            },
            "topopt": {
                "options_defined": False,
                "structural_performed": False,
                "thermal_performed": False
            },
            "current_step": "init"
        }

    def init_ui(self):
        """Initialize the ui"""
        # Main layout setup
        self.main_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        
        #Create main horizontal layout
        self.create_main_layout()
        self.create_sidebar()
        self.create_message_frame()
        self.create_status_bar()

    def create_main_layout(self):
        """Create the main horizontal layout with PyVista frame"""
        self.h_layout = QtWidgets.QHBoxLayout()
        self.h_layout.setSpacing(10)
        self.h_layout.setContentsMargins(10, 10, 10, 10)
        
        #PyVista Frame
        self.pv_frame = QtWidgets.QFrame()
        self.pv_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.pv_layout = QtWidgets.QVBoxLayout(self.pv_frame)
        self.pv_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(self.pv_frame)
        self.pv_layout.addWidget(self.plotter.interactor)
        self.h_layout.addWidget(self.pv_frame, stretch=4)

    def create_sidebar(self):
        """Create sidebar with buttons based on configuration"""
        self.sidebar = QtWidgets.QFrame()
        sidebar_layout = QtWidgets.QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(5, 5, 5, 5)
        self.sidebar.setFixedWidth(self.SIDEBAR_WIDTH)
        self.sidebar_buttons = {}
        
        button_style = """
            QPushButton {
                background-color: white;
                color: black;
                border: 1px solid #CCCCCC;
                text-align: left;
                padding: 4px;
                font-family: 'Segoe UI';
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #F0F0F0;
            }
        """
        
        for config in self.BUTTON_CONFIG:
            btn = self.create_button(config, button_style)
            sidebar_layout.addWidget(btn)
            self.sidebar_buttons[config["name"]] = btn
            
        sidebar_layout.addStretch()
        self.h_layout.addWidget(self.sidebar, stretch=0)

    def create_button(self, config, style):
        """Create a single button based on configuration"""
        btn = QtWidgets.QPushButton(config["name"])
        btn.setStyleSheet(style)
        btn.setIcon(self.get_icon(config["icon"]))
        btn.setIconSize(QSize(16, 16))
        btn.clicked.connect(lambda: self.handle_button_click(config))
        return btn

    def create_message_frame(self):
        """Create message display frame"""
        self.message_frame = QtWidgets.QFrame()
        self.message_frame.setFrameStyle(QtWidgets.QFrame.Box | QtWidgets.QFrame.Plain)
        message_layout = QtWidgets.QVBoxLayout(self.message_frame)
        
        self.message_text = QtWidgets.QTextEdit()
        self.message_text.setFixedHeight(self.MESSAGE_HEIGHT)
        self.message_text.setStyleSheet("""
            QTextEdit {
                background-color: #F0F0F0;
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
        """)
        self.message_text.setReadOnly(True)
        self.message_text.setText("Welcome to PyTO!")
        message_layout.addWidget(self.message_text)
        
        # Add to main layout
        self.main_layout.addLayout(self.h_layout, stretch=4)
        self.main_layout.addWidget(self.message_frame, stretch=1)

    def create_status_bar(self):
        """Create status bar with version information"""
        status_bar = self.statusBar()
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #F0F0F0;
                font-family: 'Segoe UI';
                font-size: 9pt;
            }
        """)
        
        labels_config = [
            ("PyTO GUI Version 2026.09. ", ""),
            ("GUI Build Date 12.06.2025. ", ""),
            ("Open source under the GNU GPL v3.0 (see LICENSE).", "")
        ]
        
        for text, style in labels_config:
            label = QtWidgets.QLabel(text)
            if style:
                label.setStyleSheet(style)
            status_bar.addWidget(label)

    def setup_visualization(self):
        """Setup PyVista visualization environment"""
        self.picker = pv._vtk.vtkCellPicker()
        self.plotter.set_background('white')
        self.plotter.enable_parallel_projection()
        self.plotter.add_axes()  # Add this line - axes added once and stay visibl

        
#################################################################
    def connect_events(self):
        """Connect event handlers"""
        # Event connections
        pass
#################################################################

    def update_geometry_info_text(self):
        """Update the geometry info text in the upper left corner with current units and material."""
        if not self.stl_geom:
            return
        area, volume, _, _ = self.stl_geom.compute_mass_properties()
        bounds = self.stl_geom.get_bounding_box()
        length_unit = self.settings.get_length_unit_string()
        material_name = self.applied_material.get("name", "None") if getattr(self, "applied_material", None) else "None"
        info_lines = [
            f"Model: {os.path.basename(self.stl_geom.file_path)}",
            f"Volume: {volume:.2e} {length_unit}³",
            f"Length: {bounds[1] - bounds[0]:.2e} {length_unit}" if bounds else "Length: N/A",
            f"Material: {material_name}"
        ]
        self.plotter.remove_actor("geometry_info")
        self.plotter.add_text(
            "\n".join(info_lines),
            position="upper_left",
            font_size=12,
            color="black",
            name="geometry_info",
            font="arial"
        )

    def handle_button_click(self, config):
        """button click handler"""
        # cam_pos = self.plotter.camera_position
        # print("="*60)
        # print(f"Camera Position:")
        # print(f"plotter.camera_position = {cam_pos}")
        # print("="*60)
        if not self.check_button_availability(config):
            return
            
        handler_method = getattr(self, config["handler"], None)
        if handler_method and callable(handler_method):
            handler_method()
        else:
            self.show_coming_soon()

    def check_button_availability(self, config):
        """Check if button is available based on workflow state"""
        if config.get("always_enabled", False):
            return True
            
        requirement = config.get("requires")
        if not requirement:
            return True
            
        if requirement == "loads_applied":
            loads_ready = (self.LivVar['structural_loads']['applied'] or 
                        self.LivVar['thermal_loads']['applied'])
            if not loads_ready:
                self.show_workflow_warning("Please apply structural or thermal loads first.")
                return False
        elif requirement == "topopt_options_defined":
            if not self.LivVar.get('topopt', {}).get('options_defined', False):
                self.show_workflow_warning("Please define topology optimization options first.")
                return False
        elif requirement in ["material_defined", "geometry_loaded"]:
            if not self.LivVar.get(requirement, False):
                self.show_workflow_warning(f"Please complete {requirement.replace('_', ' ')} first.")
                return False
                
        return True

    def show_workflow_warning(self, message):
        """Show workflow warning message"""
        QtWidgets.QMessageBox.warning(self, "Workflow Error", message)

    def update_LivVar(self, key, value=True):
        """Update state and trigger UI updates"""
        self.set_nested_value(self.LivVar, key.split('.'), value)
        self.handle_state_change(key, value)

    def set_nested_value(self, dictionary, keys, value):
        """Set value in nested dictionary using key path"""
        for key in keys[:-1]:
            dictionary = dictionary.setdefault(key, {})
        dictionary[keys[-1]] = value

    def handle_state_change(self, key, value):
        """Handle state changes and update UI accordingly"""
        state_handlers = {
            'geometry_loaded': self.handle_geometry_loaded,
            'material_defined': self.handle_material_defined,
            'structural_loads.forces_applied': self.handle_structural_loads_change,
            'structural_loads.fixed_constraints': self.handle_structural_loads_change,
            'thermal_loads.applied': self.handle_thermal_loads_applied,
            'topopt.options_defined': self.handle_topopt_options_defined,
        }
        
        handler = state_handlers.get(key)
        if handler:
            handler(value)

    def handle_geometry_loaded(self, value):
        if value:
            self.update_workflow_step('geometry_loaded')

    def handle_material_defined(self, value):
        if value:
            self.update_workflow_step('material_defined')

    def handle_structural_loads_change(self, value):
        if (self.LivVar['structural_loads']['forces_applied'] and 
            self.LivVar['structural_loads']['fixed_constraints']):
            self.LivVar['structural_loads']['applied'] = True
            self.update_workflow_step('structural_loads_complete')

    def handle_thermal_loads_applied(self, value):
        if value:
            self.update_workflow_step('thermal_loads_complete')

    def handle_topopt_options_defined(self, value):
        if value:
            self.update_workflow_step('topopt_options_defined')

    def update_workflow_step(self, step_key):
        """Update workflow step and associated UI elements"""
        step_config = self.WORKFLOW_STEPS.get(step_key, {})
        
        #Update enabled buttons
        for button_name in step_config.get('enables', []):
            self.set_sidebar_icon(button_name, "arrow")
            
        #Update current step icon
        icon = step_config.get('icon', 'check')
        button_name = self.get_button_name_for_step(step_key)
        if button_name:
            self.set_sidebar_icon(button_name, icon)
            
        #Show message
        message = step_config.get('message', '')
        if message:
            self.message_text.append(message)

    def get_button_name_for_step(self, step_key):
        """Map workflow step to button name"""
        step_to_button = {
            'material_defined': 'Material',
            'structural_loads_complete': 'Structural Loads',
            'thermal_loads_complete': 'Thermal Loads',
            'analysis_performed': 'Analysis',
            'topopt_options_defined': 'TopOpt Options'
        }
        return step_to_button.get(step_key)

    def get_icon(self, icon_type):
        """Get icon based on type"""
        base_path = os.path.dirname(__file__)
        icon_files = {
            "arrow": "arrow-right.png",
            "arrow-blue": "arrow-blue.png", 
            "cross": "cross.png",
            "check": "check.png"
        }
        
        icon_file = os.path.join(base_path, icon_files.get(icon_type, ""))
        return QIcon(icon_file) if os.path.exists(icon_file) else QIcon()

    def set_sidebar_icon(self, button_text, icon_type):
        """Update sidebar button icon"""
        btn = self.sidebar_buttons.get(button_text)
        if btn:
            btn.setIcon(self.get_icon(icon_type))


    def update_highlights(self):
        """Update triangle highlights"""
        if not self.stl_geom:
            return

        camera_position = self.plotter.camera_position
        self.remove_highlight_actor()
        
        highlight_ids = self.get_highlight_ids()
        if not highlight_ids:
            self.restore_camera_and_render(camera_position)
            return

        highlight_mesh = self.create_highlight_mesh(highlight_ids)
        self.add_highlight_mesh(highlight_mesh)
        self.restore_camera_and_render(camera_position)

    def remove_highlight_actor(self):
        """Remove existing highlight actor"""
        if self.highlight_actor:
            self.plotter.remove_actor(self.highlight_actor, reset_camera=False)
            self.highlight_actor = None

    def get_highlight_ids(self):
        """Get IDs of triangles to highlight"""
        return [i for i, h in enumerate(self.stl_geom.tri_highlight) 
                if h and i not in self.constrained_triangles]

    def create_highlight_mesh(self, highlight_ids):
        """Create mesh for highlighted triangles"""
        vertices, faces = [], []
        vertex_count = 0
        
        for tri_id in highlight_ids:
            triangle_vertices = self.stl_geom.mesh.vectors[tri_id]
            vertices.extend(triangle_vertices)
            faces.extend([3, vertex_count, vertex_count + 1, vertex_count + 2])
            vertex_count += 3
        
        highlight_mesh = pv.PolyData(np.array(vertices), np.array(faces))
        return highlight_mesh.compute_normals(cell_normals=True, point_normals=True)

    def add_highlight_mesh(self, highlight_mesh):
        """Add highlight mesh to plotter"""
        bounds = self.plotter.bounds
        offset = (0.001 * ((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + 
                          (bounds[5]-bounds[4])**2)**0.5 if bounds else 0.001)
        
        # Create double-sided mesh
        front_mesh = highlight_mesh.copy()
        front_mesh.points = front_mesh.points + front_mesh.point_data['Normals'] * offset
        back_mesh = highlight_mesh.copy() 
        back_mesh.points = back_mesh.points - back_mesh.point_data['Normals'] * offset
        
        self.highlight_actor = self.plotter.add_mesh(
            front_mesh + back_mesh,
            color="red",
            opacity=0.6,
            culling=False,
            lighting=False,
            reset_camera=False
        )

    def restore_camera_and_render(self, camera_position):
        """Restore camera position and render"""
        self.plotter.camera_position = camera_position
        self.plotter.render()

    def on_left_button_press(self, picked_point, picker=None):
        """Handle left click for triangle selection"""
        if not picker or not self.stl_geom:
            return
        
        # Get the picked actor and cell ID
        cell_id = picker.GetCellId()
        picked_actor = picker.GetActor()
        
        if cell_id < 0:
            self.message_text.append("No triangle selected.")
            return

        if getattr(self, 'highlight_mode', 'coarse') == 'node':
            # Node mode: snap to the nearest candidate node (mesh surface node, or STL vertex before meshing). Any
            # surface may be clicked (geometry or mesh view): only the clicked 3D point matters.
            candidates = candidate_node_coordinates(self)
            point = candidates[int(np.argmin(np.linalg.norm(candidates - np.asarray(picked_point), axis=1)))]
            self.add_selected_points([point])
            return
        
        # Validate we picked the STL geometry actor (not highlight/constraint overlay)
        if picked_actor:
            n_stl_triangles = len(self.stl_geom.mesh.vectors)
            if hasattr(picked_actor, 'GetMapper'):
                mapper = picked_actor.GetMapper()
                if mapper and hasattr(mapper, 'GetInput'):
                    mesh_input = mapper.GetInput()
                    if mesh_input and hasattr(mesh_input, 'GetNumberOfCells'):
                        # Reject if picked actor doesn't have same number of cells as STL
                        if mesh_input.GetNumberOfCells() != n_stl_triangles:
                            self.message_text.append("Please click on the geometry surface.")
                            return
        
        if cell_id in self.constrained_triangles:
            self.message_text.append("Cannot select constrained triangle.")
            return
        
        count, area = self.select_triangles(cell_id)
        if count > 0:
            self.update_highlights()
            self.message_text.append(f"Selected {count} triangle{'s' if count > 1 else ''} ")

    def select_triangles(self, cell_id):
        """Select triangles based on current mode"""
        mode = getattr(self, 'highlight_mode', 'coarse')
        depth, angle = (0, 0) if mode == 'triangle' else (500, 15)
        return self.stl_geom.highlight_triangles_recursive(cell_id, depth, angle)

    def add_selected_points(self, points):
        """Add node-mode picks (coordinates), skipping ones already selected."""
        added = 0
        for point in np.asarray(points, dtype=float).reshape(-1, 3):
            if any(np.linalg.norm(np.asarray(p) - point) < 1e-12 for p in self.selected_points):
                continue
            self.selected_points.append(point.tolist())
            added += 1
        self.update_point_markers()
        self.message_text.append(f"Selected {added} node(s); {len(self.selected_points)} selected in total."
                                 if added else "Node(s) already selected.")
        return added

    def show_candidate_nodes(self, show):
        """Grey dots on the nodes the Node selection mode can pick (mesh surface nodes once meshed)."""
        if 'candidate_nodes' in self.plotter.actors:
            self.plotter.remove_actor('candidate_nodes', reset_camera=False)
        if show and self.stl_geom:
            self.plotter.add_mesh(pv.PolyData(candidate_node_coordinates(self)), color='dimgray', point_size=5,
                                  render_points_as_spheres=True, pickable=False, name='candidate_nodes')
        self.plotter.render()

    def start_box_node_selection(self):
        """Drag a rectangle: the box is kept as a region. Every mesh surface node inside it (through the whole depth)
        is selected when the analysis/optimization runs, so it works before meshing and after remeshing."""
        self.plotter.disable_picking()   # click picking is always on; pyvista allows one picking mode at a time
        self.plotter.enable_rectangle_picking(callback=self.on_box_node_selection, start=True,
                                              show_message="Drag a box to select nodes")

    def on_box_node_selection(self, selection):
        """Called by VTK at the end of the rubber-band drag, INSIDE its event handler. Only copy the box here:
        tearing down the picker / interaction style from within its own event handler crashes VTK (segmentation
        fault, reproduced). Drawing and restoring click picking run right after VTK returns (QTimer, 0 ms)."""
        region = frustum_to_region(selection.frustum)          # plain data, independent of the picker
        QtCore.QTimer.singleShot(0, lambda: self._finish_box_node_selection(region))

    def _finish_box_node_selection(self, region):
        preview = int(points_in_region(candidate_node_coordinates(self), region).sum())
        if preview == 0:
            self.message_text.append("The box contains no part of the model; nothing selected.")
        else:
            self.selected_regions.append(region)
            self.update_point_markers()
            self.message_text.append(
                f"Box selection added ({preview} {'mesh nodes' if getattr(self, 'hex_mesh', None) else 'surface points'} "
                f"inside now; with a mesh, all its surface nodes inside the box are used).")
        self.restore_surface_picking()

    def restore_surface_picking(self):
        """Back to normal rotation + click picking after a box selection."""
        self.plotter.disable_picking()
        self.plotter.enable_trackball_style()
        self.plotter.enable_point_picking(callback=self.on_left_button_press, use_picker=True, picker='cell',
                                          show_message=False, left_clicking=True, show_point=False)

    def node_selection_preview(self, points=None, regions=None):
        """Coordinates to draw for a node selection: the picked points plus the candidates inside each box."""
        points = self.selected_points if points is None else points
        regions = self.selected_regions if regions is None else regions
        parts = [np.asarray(points, dtype=float).reshape(-1, 3)]
        if regions:
            cand = candidate_node_coordinates(self)
            parts += [cand[points_in_region(cand, r)] for r in regions]
        return np.vstack(parts) if parts else np.zeros((0, 3))

    def update_point_markers(self):
        """Draw the node-mode selection (orange): picked nodes and the points inside box selections."""
        if 'selected_points' in self.plotter.actors:
            self.plotter.remove_actor('selected_points', reset_camera=False)
        preview = self.node_selection_preview()
        if len(preview):
            self.plotter.add_mesh(pv.PolyData(preview), color='orange', point_size=12,
                                  render_points_as_spheres=True, pickable=False, name='selected_points')
        self.plotter.render()

    def on_right_button_press(self, obj, event):
        """Handle right click for deselecting all triangles (and all node-mode picks)"""
        if not hasattr(self, 'stl_geom') or not self.stl_geom:
            return
        if getattr(self, 'selected_points', None) or getattr(self, 'selected_regions', None):
            n, b = len(self.selected_points), len(self.selected_regions)
            self.selected_points, self.selected_regions = [], []
            self.update_point_markers()
            self.message_text.append(f"Deselected {n} node(s) and {b} box selection(s).")
            
        selected_count = sum(1 for h in self.stl_geom.tri_highlight if h)
        if selected_count == 0:
            self.message_text.append("No triangles selected to deselect.")
            return
        
        # Deselect all triangles
        for i in range(len(self.stl_geom.tri_highlight)):
            self.stl_geom.tri_highlight[i] = False
        
        self.update_highlights()
        self.message_text.append(f"Deselected all {selected_count} selected triangles.")

    def notify_display_options_update(self):
        """Notify display options window to update checkbox states"""
        # Check if display options window exists and is visible
        for child in self.findChildren(QtWidgets.QDialog):
            if isinstance(child, DisplayOptionsWindow) and child.isVisible():
                child.update_checkbox_states()

    #Window opening methods
    def open_units_window(self):
        dialog = UnitsWindow(self, self.settings)
        dialog.exec_()

    def open_geometry_window(self):
        dialog = GeometryWindow(self)
        dialog.exec_()

    def open_material_window(self):
        dialog = MaterialWindow(self)
        dialog.exec_()

    def open_structural_loads_window(self):
        self.structural_loads_window = StructuralLoadsWindow(self)
        self.structural_loads_window.show()

    def open_thermal_loads_window(self):
        self.thermal_loads_window = ThermalLoadsWindow(self)
        self.thermal_loads_window.show()

    def open_analysis_window(self):
        dialog = AnalysisWindow(self)
        dialog.show()

    def open_topopt_options_window(self):
        # Only remove mesh/analysis/result actors, keep geometry, loads, constraints, and info
        keep_names = {
            "geometry_info",
            "stl_geometry",
        }
        # Add all structural/thermal load actors and constraint actors to keep_names
        if hasattr(self, "force_actors"):
            for actor in self.force_actors:
                if hasattr(actor, "GetName"):
                    keep_names.add(actor.GetName())
        if hasattr(self, "constraint_actors"):
            for actor in self.constraint_actors:
                if hasattr(actor, "GetName"):
                    keep_names.add(actor.GetName())
        if hasattr(self, "thermal_loads_window"):
            for attr in ["fixed_temp_actors", "heat_source_actors", "total_heat_actors"]:
                if hasattr(self.thermal_loads_window, attr):
                    for actor in getattr(self.thermal_loads_window, attr):
                        if hasattr(actor, "GetName"):
                            keep_names.add(actor.GetName())
        if hasattr(self, "topopt_option_actors"):
            for actor in self.topopt_option_actors.values():
                if isinstance(actor, list):
                    for a in actor:
                        if hasattr(a, "GetName"):
                            keep_names.add(a.GetName())
                else:
                    if hasattr(actor, "GetName"):
                        keep_names.add(actor.GetName())

        # Remove all actors not in keep_names
        for name in list(self.plotter.actors.keys()):
            if name not in keep_names:
                self.plotter.remove_actor(name, reset_camera=False)

        # Re-plot the STL geometry if not present
        if self.stl_geom and "stl_geometry" not in self.plotter.actors:
            self.stl_geom.plotGeometry(
                show_edges=False,
                show_axes=True,
                show_bounding_box=False,
                plotter=self.plotter,
                color=MATERIAL_COLOR
            )
        self.update_highlights()
        self.topopt_options_window = TopOptOptionsWindow(self)
        self.topopt_options_window.show()

    def open_structural_topopt_window(self):
        self.structural_topopt_window = StructuralTopOptWindow(self)
        self.structural_topopt_window.show()

    def open_projects_window(self):
        dialog = ProjectsWindow(self)
        dialog.exec_()

    def open_display_options_window(self):
        dialog = DisplayOptionsWindow(self)
        dialog.show()

    def open_topopt_results_window(self):
        self.topopt_results_window = TopOptResultsWindow(self)
        self.topopt_results_window.show()

    def open_body_force_window(self):
        dialog = BodyForceWindow(self)
        dialog.exec_()

    def show_coming_soon(self):
        QtWidgets.QMessageBox.information(self, "Coming Soon", 
                                         "This feature will be implemented soon.")

    def show_help(self):
        QtWidgets.QMessageBox.information(self, "Help", 
                                         "PyTO Help documentation will be available soon.")
#---------------------------------------------------------------------------
class Settings:
    def __init__(self):
        self.unit_system = "MKS"
        self.temperature_unit = "Kelvin"
        self.angle_unit = "Degree"

        #Add conversion factors for different unit systems
        self.unit_conversions = {
            #Length conversions to/from meters
            "length": {
                "MKS": 1.0,       # meters (base)
                "mmKS": 1000.0,   # millimeters
                "IPS": 39.37      # inches
            },
            #Force conversions to/from Newtons
            "force": {
                "MKS": 1.0,      # Newtons (base)
                "mmKS": 1.0,     # Newtons
                "IPS": 0.2248    # pounds-force
            },
            #Stress conversions to/from Pascal
            "stress": {
                "MKS": 1.0,           # Pascal (base) 
                "mmKS": 1.0e-6,       # MPa
                "IPS": 1.45038e-4     # psi
            }
        }

    def update_settings(self, unit_system, temperature_unit, angle_unit):
        self.unit_system = unit_system
        self.temperature_unit = temperature_unit
        self.angle_unit = angle_unit

    def convert_length(self, value, from_system="MKS", to_system=None):
        """Convert a length value between unit systems"""
        if to_system is None:
            to_system = self.unit_system
            
        if from_system == to_system:
            return value
            
        #Convert to meters first (base unit)
        value_in_meters = value / self.unit_conversions["length"][from_system]
        
        #Then convert to target unit
        return value_in_meters * self.unit_conversions["length"][to_system]
    
    def convert_force(self, value, from_system="MKS", to_system=None):
        """Convert a force value between unit systems"""
        if to_system is None:
            to_system = self.unit_system
            
        if from_system == to_system:
            return value
            
        #Convert to newtons first (base unit)
        value_in_newtons = value / self.unit_conversions["force"][from_system]
        
        #Then convert to target unit
        return value_in_newtons * self.unit_conversions["force"][to_system]
    
    def convert_stress(self, value, from_system="MKS", to_system=None):
        """Convert a stress value between unit systems"""
        if to_system is None:
            to_system = self.unit_system
            
        if from_system == to_system:
            return value
            
        #Convert to Pascal first (base unit)
        value_in_pascal = value / self.unit_conversions["stress"][from_system]
        
        #Then convert to target unit
        return value_in_pascal * self.unit_conversions["stress"][to_system]

    def get_length_unit_string(self):
        if self.unit_system == "MKS":
            return "m"
        elif self.unit_system == "mmKS":
            return "mm"
        elif self.unit_system == "IPS":
            return "in"
        return "m"
    
    def get_force_unit_string(self):
        """Return the force unit string"""
        if self.unit_system == "MKS" or self.unit_system == "mmKS":
            return "N"
        elif self.unit_system == "IPS":
            return "lbf"
        return "N"  # Default
    
    def get_stress_unit_string(self):
        """Return the appropriate stress unit string"""
        if self.unit_system == "MKS":
            return "Pa"
        elif self.unit_system == "mmKS":
            return "MPa"
        elif self.unit_system == "IPS":
            return "psi"
        return "Pa"  # Default

    def get_temperature_unit_symbol(self):
        if self.temperature_unit == "Celsius":
            return "°C"
        elif self.temperature_unit == "Fahrenheit":
            return "°F"
        else:
            return "K"
#---------------------------------------------------------------------------
class UnitsWindow(QtWidgets.QDialog):
    def __init__(self, parent, settings):
        super().__init__(parent)
        self.setWindowTitle("Units")
        self.resize(220, 180)
        self.settings = settings

        layout = QtWidgets.QVBoxLayout(self)

        #Unit system
        layout.addWidget(QtWidgets.QLabel("Unit System:"))
        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(["MKS", "mmKS", "IPS"])
        self.unit_combo.setCurrentText(settings.unit_system)
        layout.addWidget(self.unit_combo)

        #Temperature unit
        layout.addWidget(QtWidgets.QLabel("Temperature Unit:"))
        self.temp_combo = QtWidgets.QComboBox()
        self.temp_combo.addItems(["Kelvin", "Celsius", "Fahrenheit"])
        self.temp_combo.setCurrentText(settings.temperature_unit)
        layout.addWidget(self.temp_combo)

        #Angle unit
        layout.addWidget(QtWidgets.QLabel("Angle Unit:"))
        self.angle_combo = QtWidgets.QComboBox()
        self.angle_combo.addItems(["Degree", "Radian"])
        self.angle_combo.setCurrentText(settings.angle_unit)
        layout.addWidget(self.angle_combo)

        #Apply button
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_settings)
        layout.addWidget(apply_btn)

    def apply_settings(self):
        self.settings.update_settings(
            self.unit_combo.currentText(),
            self.temp_combo.currentText(),
            self.angle_combo.currentText()
        )
        if hasattr(self.parent(), "set_sidebar_icon"):
            self.parent().set_sidebar_icon("Units", "check")
        if hasattr(self.parent(), "message_text"):
            self.parent().message_text.append(
                f"Units defined: {self.unit_combo.currentText()}, "
                f"{self.temp_combo.currentText()}, {self.angle_combo.currentText()}"
            )
        # Update all open windows with new units
        if hasattr(self.parent(), "update_geometry_info_text"):
            self.parent().update_geometry_info_text()
        if hasattr(self.parent(), "structural_loads_window"):
            self.parent().structural_loads_window.update_units()
        if hasattr(self.parent(), "material_window"):
            self.parent().material_window.update_units()
        if hasattr(self.parent(), "thermal_loads_window"):
            self.parent().thermal_loads_window.update_units()
        
        self.accept()
#---------------------------------------------------------------------------
class GeometryWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Geometry")
        self.resize(200, 120)  
        self.parent = parent 

        layout = QtWidgets.QVBoxLayout(self)

        load_btn = QtWidgets.QPushButton("Load STL Geometry")
        load_btn.clicked.connect(self.load_geometry)
        layout.addWidget(load_btn)

        # # Add Update Geometry button
        # self.update_btn = QtWidgets.QPushButton("Update Geometry")
        # self.update_btn.clicked.connect(self.update_geometry)
        # self.update_btn.setEnabled(False)  # Initially disabled
        # layout.addWidget(self.update_btn)

        get_stl_btn = QtWidgets.QPushButton("Get STL from SolidWorks")
        get_stl_btn.clicked.connect(self.get_stl_from_solidworks)
        layout.addWidget(get_stl_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

    def load_geometry(self):
        """Load STL geometry (always clears and reloads)"""
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select STL File",
            "",
            "STL Files (*.stl *.STL);;All Files (*)",
            options=options
        )
        if file_path:
            # Clear the plotter completely
            self.parent.plotter.clear_actors()
            self.parent.plotter.reset_camera()
            self.parent.plotter.disable_picking()

            # Clear all stored data
            self.parent.constrained_triangles.clear()
            self.parent.constraint_actors.clear()
            self.parent.constraint_data.clear()
            self.parent.force_data.clear()
            self.parent.force_actors.clear()
            self.parent.material_data = None
            self.parent.applied_material = None
            self.parent.highlight_actor = None

            # Create new STL geometry
            self.stl_geom = STLGeom(file_path)

            # Plot geometry
            self.stl_geom.plotGeometry(
                show_edges=False, 
                show_axes=True, 
                show_bounding_box=False, 
                plotter=self.parent.plotter,
                color=MATERIAL_COLOR
            )

            # Set geometry reference
            self.parent.stl_geom = self.stl_geom

            # Use the shared info text update
            self.parent.update_geometry_info_text()

            # Enable picking
            picker = pv._vtk.vtkCellPicker()
            picker.SetTolerance(1e-8)
            self.parent.plotter.enable_point_picking(
                callback=self.parent.on_left_button_press,
                use_picker=True,
                picker=picker,  
                show_message=False,
                left_clicking=True,
                show_point=False
            )

            # Add right-click observer
            self.parent.plotter.iren.add_observer("RightButtonPressEvent", self.parent.on_right_button_press)

            # Update LivVar
            self.parent.update_LivVar('geometry_loaded', True)

            # Update sidebar icons
            self.parent.set_sidebar_icon("Geometry", "check")
            self.parent.set_sidebar_icon("Material", "arrow")

            # Success message
            self.parent.message_text.append(f"Geometry loaded: {os.path.basename(file_path)}")

            self.close()

    def get_stl_from_solidworks(self):
        """Get STL from currently open SolidWorks part using solidworks_interface.py"""

        if 'SolidWorksInterface' not in globals():
            return
        
        try:
            sw_cls = globals().get('SolidWorksInterface')
            if sw_cls is None:
                return
            sw = sw_cls()
            stl_geom = sw.getSTL()
            if not stl_geom:
                QtWidgets.QMessageBox.warning(self, "SolidWorks Error", "Failed to get STL from SolidWorks.")
                return
            # Plot and set in parent
            stl_geom.plotGeometry(show_edges=False, show_axes=True, show_bounding_box=False, plotter=self.parent.plotter,
                               color=MATERIAL_COLOR)
            self.parent.stl_geom = stl_geom

            # Use the shared info text update
            self.parent.update_geometry_info_text()

            self.parent.plotter.disable_picking()
            self.parent.plotter.enable_point_picking(callback=self.parent.on_left_button_press, use_picker=True, 
                                                    picker='cell', show_message=False, left_clicking=True, show_point=False)
            self.parent.plotter.iren.add_observer("RightButtonPressEvent", self.parent.on_right_button_press)
            self.parent.update_LivVar('geometry_loaded', True)
            self.parent.set_sidebar_icon("Geometry", "check")
            self.parent.set_sidebar_icon("Material", "arrow")
            self.parent.message_text.append("STL geometry loaded from SolidWorks.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "SolidWorks Error", f"Error: {str(e)}")

#---------------------------------------------------------------------------
class MaterialWindow(QtWidgets.QDialog):
    """Dialog for selecting and editing material properties."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Material")
        self.setBaseSize(300, 400)
        self.parent = parent 

        # Store all materials in SI units only!
        self.materials_SI = {
            "Steel": {
                "Young's Modulus": 2.1e11,
                "Poisson's Ratio": 0.3,
                "Yield Strength": 2.5e8,
                "Density": 7850,
                "Thermal Conductivity": 50,
                "Thermal Expansion": 1.2e-5,
                "Specific Heat Capacity": 500,
                "Price": 0.8
            },
            "Aluminum": {
                "Young's Modulus": 7.0e10,
                "Poisson's Ratio": 0.33,
                "Yield Strength": 2.7e8,
                "Density": 2700,
                "Thermal Conductivity": 237,
                "Thermal Expansion": 2.3e-5,
                "Specific Heat Capacity": 900,
                "Price": 2.0
            },
            "Titanium": {
                "Young's Modulus": 1.16e11,
                "Poisson's Ratio": 0.34,
                "Yield Strength": 8.8e8,
                "Density": 4500,
                "Thermal Conductivity": 21.9,
                "Thermal Expansion": 8.6e-6,
                "Specific Heat Capacity": 520,
                "Price": 30.0
            },
            "Custom": {
                "Young's Modulus": 0.0,
                "Poisson's Ratio": 0.0,
                "Yield Strength": 0.0,
                "Density": 0.0,
                "Thermal Conductivity": 0.0,
                "Thermal Expansion": 0.0,
                "Specific Heat Capacity": 0.0,
                "Price": 0.0
            }
        }

        self.material_applied = False

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        form_layout.setLabelAlignment(QtCore.Qt.AlignRight)

        # Material name dropdown
        self.material_combo = QtWidgets.QComboBox()
        self.material_combo.addItems(self.materials_SI.keys())
        self.material_combo.currentTextChanged.connect(self.on_material_changed)
        form_layout.addRow("Material Name:", self.material_combo)

        # Create fields
        self.fields = {}
        self.fields["Young's Modulus"] = QtWidgets.QLineEdit()
        form_layout.addRow(self.label_with_unit("Young's Modulus"), self.fields["Young's Modulus"])
        self.fields["Poisson's Ratio"] = QtWidgets.QLineEdit()
        form_layout.addRow("Poisson's Ratio (-):", self.fields["Poisson's Ratio"])
        self.fields["Yield Strength"] = QtWidgets.QLineEdit()
        form_layout.addRow(self.label_with_unit("Yield Strength"), self.fields["Yield Strength"])
        self.fields["Density"] = QtWidgets.QLineEdit()
        form_layout.addRow(self.label_with_unit("Density"), self.fields["Density"])
        self.fields["Thermal Conductivity"] = QtWidgets.QLineEdit()
        form_layout.addRow("Thermal Conductivity (W/m-K):", self.fields["Thermal Conductivity"])
        self.fields["Thermal Expansion"] = QtWidgets.QLineEdit()
        form_layout.addRow("Thermal Expansion (m/m-K):", self.fields["Thermal Expansion"])
        self.fields["Specific Heat Capacity"] = QtWidgets.QLineEdit()
        form_layout.addRow("Spec. Heat Capacity (J/kg-K):", self.fields["Specific Heat Capacity"])
        self.fields["Price"] = QtWidgets.QLineEdit()
        form_layout.addRow("Price (US$/kg):", self.fields["Price"])

        for field in self.fields.values():
            field.setMaximumWidth(120)

        layout.addLayout(form_layout)

        # Checkbox and buttons
        self.optimize_check = QtWidgets.QCheckBox("Do not optimize")
        layout.addWidget(self.optimize_check)

        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self.apply_material)
        layout.addWidget(self.apply_btn)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        layout.addWidget(self.close_btn)

        self.on_material_changed(self.material_combo.currentText())

    def label_with_unit(self, key):
        s = self.parent.settings
        if key == "Young's Modulus" or key == "Yield Strength":
            return f"{key} ({s.get_stress_unit_string()}):"
        elif key == "Density":
            return f"{key} (kg/{s.get_length_unit_string()}³):"
        return key

    def on_material_changed(self, name):
        mat_SI = self.materials_SI[name]
        s = self.parent.settings
        for key, field in self.fields.items():
            value = mat_SI[key]
            # Convert for display
            if key in ["Young's Modulus", "Yield Strength"]:
                value_disp = s.convert_stress(value, from_system="MKS", to_system=s.unit_system)
            elif key == "Density":
                # Optionally implement density conversion if needed
                value_disp = value
            else:
                value_disp = value
            # Show large/small numbers in scientific notation, others as normal
            if isinstance(value_disp, float) and (abs(value_disp) >= 1e5 or (abs(value_disp) < 1e-2 and value_disp != 0)):
                field.setText(f"{value_disp:.2e}")
            else:
                field.setText(str(value_disp))
            field.setReadOnly(name != "Custom")
        if self.material_applied:
            self.material_combo.setEnabled(False)
            for field in self.fields.values():
                field.setReadOnly(True)
            self.apply_btn.setEnabled(False)

    def apply_material(self):
        name = self.material_combo.currentText()
        s = self.parent.settings
        if name == "Custom":
            # Save custom values (convert back to SI)
            for key in self.fields:
                try:
                    val_disp = float(self.fields[key].text())
                    # Convert back to SI for storage
                    if key in ["Young's Modulus", "Yield Strength"]:
                        val_SI = s.convert_stress(val_disp, from_system=s.unit_system, to_system="MKS")
                    elif key == "Density":
                        val_SI = val_disp  # Add conversion if needed
                    else:
                        val_SI = val_disp
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, "Input Error", f"Invalid value for {key}")
                    return
                self.materials_SI["Custom"][key] = val_SI
        self.material_applied = True
        self.on_material_changed(name)

        # Store the applied material in parent for later use (in SI units)
        self.parent.applied_material = {
            'name': name,
            'properties': self.materials_SI[name].copy()
        }

        # Update geometry info text if geometry is loaded
        if getattr(self.parent, "stl_geom", None):
            self.parent.plotter.remove_actor("geometry_info")
            area, volume, _, _ = self.parent.stl_geom.compute_mass_properties()
            bounds = self.parent.stl_geom.get_bounding_box()
            length_unit = self.parent.settings.get_length_unit_string()
            material_name = self.parent.applied_material.get("name", "None")
            info_lines = [
                f"Model: {os.path.basename(self.parent.stl_geom.file_path)}",
                f"Volume: {volume:.2e} {length_unit}³",
                f"Length: {bounds[1] - bounds[0]:.2e} {length_unit}" if bounds else "Length: N/A",
                f"Material: {material_name}"
            ]
            self.parent.plotter.add_text(
                "\n".join(info_lines),
                position="upper_left",
                font_size=12,
                color="black",
                name="geometry_info",
                font="arial",
            )

        self.parent.update_LivVar('material_defined', True)
        self.parent.message_text.append(f"Material '{name}' applied successfully.")
        self.parent.set_sidebar_icon("Material", "check")
        self.parent.set_sidebar_icon("Structural Loads", "arrow")
        self.close()

    def update_units(self):
        form_layout = self.layout().itemAt(0).layout()
        form_layout.labelForField(self.fields["Young's Modulus"]).setText(self.label_with_unit("Young's Modulus"))
        form_layout.labelForField(self.fields["Yield Strength"]).setText(self.label_with_unit("Yield Strength"))
        form_layout.labelForField(self.fields["Density"]).setText(self.label_with_unit("Density"))
        self.on_material_changed(self.material_combo.currentText())

    def showEvent(self, event):
        super().showEvent(event)
        self.update_units()
#---------------------------------------------------------------------------
class ScientificDoubleSpinBox(QtWidgets.QDoubleSpinBox):
    def valueFromText(self, text):
        try:
            # Accept both normal and scientific notation
            return float(text)
        except Exception:
            return super().valueFromText(text)
    def textFromValue(self, value):
        # Show in scientific notation for large/small values, else normal
        if abs(value) >= 1e5 or (abs(value) < 1e-2 and value != 0):
            return f"{value:.2e}"
        else:
            return f"{value:.2f}"
#---------------------------------------------------------------------------
class StructuralLoadsWindow(QtWidgets.QDialog):
    # Class-level configuration
    FORCE_RANGE = (-1e6, 1e6)
    FORCE_DECIMALS = 1
    ARROW_THRESHOLD = 25
    MAX_MARKERS = 5
    ARROW_SCALE_FACTOR = 0.1
    CONSTRAINT_OFFSET_FACTOR = 0.002
    show_load_values = True   # numeric force / torque label at the arrow tip; set False to hide it
    HIGHLIGHT_OFFSET_FACTOR = 0.001
    
    #Load type configuration
    LOAD_TYPES = {
        "Force": {
            "handler": "apply_force",
            "show_force": True,
            "show_torque": False,
            "directions": {"X": [1,0,0], "Y": [0,1,0], "Z": [0,0,1]},
            "color": LOAD_COLOR
        },
        "Torque": {
            "handler": "apply_torque", 
            "show_force": False,
            "show_torque": True,
            "color": "blue"
        },
        "Fixed XYZ": {
            "handler": "apply_constraint",
            "constraint_type": "Fixed XYZ",
            "axes": ["X", "Y", "Z"],
            "show_force": False,
            "show_torque": False
        },
        "Fixed X": {
            "handler": "apply_constraint",
            "constraint_type": "Fixed X", 
            "axes": ["X"],
            "show_force": False,
            "show_torque": False
        },
        "Fixed Y": {
            "handler": "apply_constraint",
            "constraint_type": "Fixed Y",
            "axes": ["Y"], 
            "show_force": False,
            "show_torque": False
        },
        "Fixed Z": {
            "handler": "apply_constraint",
            "constraint_type": "Fixed Z",
            "axes": ["Z"],
            "show_force": False,
            "show_torque": False
        }
    }
    
    SELECTION_MODES = {
        "Facet": "coarse",
        "Triangle": "triangle",
        "Node": "node",          # click snaps to the nearest STL vertex; force / supports on those nodes
    }

    

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_window()
        self.init_ui()
        self.connect_events()

    def init_window(self):
        """Initialize window properties"""
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowTitle("Structural Loads")

    def init_ui(self):
        """Initialize user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        self.create_load_set_control(layout)
        self.create_selection_control(layout) 
        self.create_load_type_control(layout)
        self.create_force_controls(layout)
        self.create_torque_controls(layout)
        self.create_buttons(layout)

    def create_load_set_control(self, parent_layout):
        """Create load set spinbox control"""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Load Set"))
        
        self.load_set_spin = QtWidgets.QSpinBox()
        self.load_set_spin.setMinimum(0)
        layout.addWidget(self.load_set_spin)
        
        parent_layout.addLayout(layout)

    def closeEvent(self, event):
        if getattr(self.parent, "stl_geom", None) and hasattr(self.parent, "show_candidate_nodes"):
            self.parent.show_candidate_nodes(False)          # node dots only while this window is open
        super().closeEvent(event)

    def create_selection_control(self, parent_layout):
        """Create selection mode control"""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Selection"))
        
        self.selection_combo = QtWidgets.QComboBox()
        self.selection_combo.addItems(list(self.SELECTION_MODES.keys()))
        layout.addWidget(self.selection_combo)

        # Node mode only: drag a rectangle to select all nodes inside it
        self.box_select_button = QtWidgets.QPushButton("Box select nodes")
        self.box_select_button.setToolTip("Then drag a rectangle in the 3D view: every node inside it is selected "
                                          "(through the whole depth). Click adds single nodes; right-click clears.")
        self.box_select_button.clicked.connect(lambda: self.parent.start_box_node_selection())
        self.box_select_button.setVisible(False)
        layout.addWidget(self.box_select_button)
        
        parent_layout.addLayout(layout)

    def create_load_type_control(self, parent_layout):
        """Create load type control"""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Load Type"))
        
        self.load_type = QtWidgets.QComboBox()
        self.load_type.addItems(list(self.LOAD_TYPES.keys()))
        layout.addWidget(self.load_type)
        
        parent_layout.addLayout(layout)

    def create_force_controls(self, parent_layout):
        """Create force component controls"""
        self.force_group = QtWidgets.QGroupBox("Force Components")
        layout = QtWidgets.QVBoxLayout(self.force_group)
        
        # Create force spinboxes dynamically
        self.force_spinboxes = {}
        for axis in ["X", "Y", "Z"]:
            spinbox = self.create_force_spinbox(f"{axis}: ")
            self.force_spinboxes[axis] = spinbox
            layout.addWidget(spinbox)
        
        parent_layout.addWidget(self.force_group)

    def create_force_spinbox(self, prefix):
        """Create a single force spinbox with scientific notation support"""
        spinbox = ScientificDoubleSpinBox()
        spinbox.setRange(*self.FORCE_RANGE)
        spinbox.setDecimals(self.FORCE_DECIMALS)
        spinbox.setPrefix(prefix)
        return spinbox

    def create_torque_controls(self, parent_layout):
        """Create torque controls"""
        self.torque_group = QtWidgets.QGroupBox("Torque")
        layout = QtWidgets.QVBoxLayout(self.torque_group)
        
        self.torque_magnitude_spin = self.create_force_spinbox("Magnitude: ")
        layout.addWidget(self.torque_magnitude_spin)
        
        self.torque_group.hide()
        parent_layout.addWidget(self.torque_group)

    def create_buttons(self, parent_layout):
        """Create control buttons"""
        layout = QtWidgets.QHBoxLayout()
        
        self.apply_button = QtWidgets.QPushButton("Apply")
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        
        layout.addWidget(self.apply_button)
        layout.addWidget(close_button)
        parent_layout.addLayout(layout)

    def connect_events(self):
        """Connect all event handlers"""
        self.selection_combo.currentIndexChanged.connect(self.on_selection_mode_changed)
        self.load_type.currentTextChanged.connect(self._on_load_type_changed)
        self.apply_button.clicked.connect(self.on_apply_clicked)
        
        #Set initial state
        self.on_selection_mode_changed(0)
        self._on_load_type_changed(self.load_type.currentText())

    def on_selection_mode_changed(self, index):
        """Handle selection mode changes"""
        mode_text = self.selection_combo.currentText()
        self.parent.highlight_mode = self.SELECTION_MODES[mode_text]
        node_mode = self.parent.highlight_mode == "node"
        if hasattr(self, "box_select_button"):
            self.box_select_button.setVisible(node_mode)
        if self.parent.stl_geom and hasattr(self.parent, "show_candidate_nodes"):
            self.parent.show_candidate_nodes(node_mode)
        if self.parent.stl_geom:
            # Turn ON triangle edges in "Triangle" mode, OFF otherwise
            for name, actor in self.parent.plotter.actors.items():
                if hasattr(actor, 'GetProperty'):
                    prop = actor.GetProperty()
                    if self.parent.highlight_mode == "triangle":
                        if hasattr(prop, 'EdgeVisibilityOn'):
                            prop.EdgeVisibilityOn()
                            prop.SetEdgeColor(0, 0, 0)
                            prop.SetLineWidth(1)
                    else:
                        if hasattr(prop, 'EdgeVisibilityOff'):
                            prop.EdgeVisibilityOff()
            if self.parent.highlight_mode == "triangle":
                self.parent.update_highlights()
            else:
                if getattr(self.parent, "highlight_actor", None):
                    self.parent.plotter.remove_actor(self.parent.highlight_actor, reset_camera=False)
                    self.parent.highlight_actor = None
                self.parent.plotter.render()

    def _on_load_type_changed(self, load_type):
        """Handle load type changes and update UI visibility"""
        config = self.LOAD_TYPES[load_type]
        
        self.force_group.setVisible(config.get("show_force", False))
        self.torque_group.setVisible(config.get("show_torque", False))
        
        self.adjustSize()

    def on_apply_clicked(self):
        """apply button"""
        load_type = self.load_type.currentText()
        config = self.LOAD_TYPES[load_type]
        handler_name = config["handler"]
        
        handler_method = getattr(self, handler_name, None)
        if handler_method and callable(handler_method):
            if handler_name == "apply_constraint":
                handler_method(config)
            else:
                handler_method()

    def get_selected_triangles(self):
        """Get selected triangles with validation"""
        selected_triangles = self.parent.stl_geom.store_selected_triangles()
        
        if not selected_triangles:
            QtWidgets.QMessageBox.warning(self, "No Selection", 
                                         "No triangles selected for load application.")
            return None
        
        return selected_triangles

    def convert_forces_to_base_units(self, forces):
        """Convert forces from current units to base units"""
        # current_unit_system = self.parent.settings.unit_system
        # if current_unit_system == "MKS":
        #     return forces
        
        # converted = {}
        # for axis, force_value in forces.items():
        #     converted[axis] = self.parent.settings.convert_force(
        #         force_value, current_unit_system, "MKS"
        #     )
        # return converted
        return forces

    def get_force_values(self):
        """Get force values from spinboxes"""
        forces = {}
        for axis, spinbox in self.force_spinboxes.items():
            forces[axis] = spinbox.value()
        return forces

    def validate_forces(self, forces):
        """Validate that at least one force component is non-zero"""
        if all(force == 0 for force in forces.values()):
            QtWidgets.QMessageBox.warning(self, "Zero Force", 
                                         "Please enter at least one non-zero force component.")
            return False
        return True

    def clear_force_inputs(self):
        """Clear all force input spinboxes"""
        for spinbox in self.force_spinboxes.values():
            spinbox.setValue(0)

    def _node_mode(self):
        return getattr(self.parent, 'highlight_mode', 'coarse') == 'node'

    def _take_selected_points(self):
        """(points, regions) of the current node selection, or None (with a message) if it is empty."""
        points = list(getattr(self.parent, 'selected_points', []) or [])
        regions = list(getattr(self.parent, 'selected_regions', []) or [])
        if not points and not regions:
            QtWidgets.QMessageBox.warning(self, "No Selection", "No nodes selected. In Node mode, click the "
                                                                  "geometry or use Box select (right-click clears).")
            return None
        return points, regions

    def _clear_selected_points(self):
        self.parent.selected_points, self.parent.selected_regions = [], []
        self.parent.update_point_markers()

    def apply_force_to_points(self):
        """Force on picked nodes: the entered force is the total, split evenly over the nodes (as for faces)."""
        selection = self._take_selected_points()
        if not selection:
            return
        points, regions = selection
        forces = self.get_force_values()
        if not self.validate_forces(forces):
            return
        f = self.convert_forces_to_base_units(forces)
        entry = {'type': 'force_xyz', 'triangles': [], 'points': points, 'regions': regions, 'force_x': f['X'],
                 'force_y': f['Y'], 'force_z': f['Z'], 'load_set': self.load_set_spin.value()}
        self.parent.force_data.append(entry)
        self._rebuild_force_visualization(entry)
        self._clear_selected_points()
        force_unit = self.parent.settings.get_force_unit_string()
        self.parent.message_text.append(
            f"Applied force ({f['X']:+.4g}, {f['Y']:+.4g}, {f['Z']:+.4g}) {force_unit} (total) to "
            f"{len(points)} node(s) and {len(regions)} box selection(s).")
        self.parent.update_LivVar('structural_loads.forces_applied', True)
        self.clear_force_inputs()
        self.parent.notify_display_options_update()

    def apply_constraint_to_points(self, config):
        """Point support: fix the picked nodes in the constraint's directions."""
        selection = self._take_selected_points()
        if not selection:
            return
        points, regions = selection
        self.parent.constraint_data.append({'type': config["constraint_type"], 'triangles': [], 'points': points,
                                            'regions': regions})
        self._clear_selected_points()
        self.visualize_constraints()
        self.parent.update_LivVar('structural_loads.fixed_constraints', True)
        self.parent.message_text.append(f"Applied {config['constraint_type']} to {len(points)} node(s) and "
                                        f"{len(regions)} box selection(s).")
        self.parent.notify_display_options_update()

    def draw_point_force(self, entry):
        """Arrows at the nodes of a point load, pointing along the force."""
        f = np.array([entry['force_x'], entry['force_y'], entry['force_z']], dtype=float)
        norm = np.linalg.norm(f)
        if norm == 0:
            return
        scale = self.calculate_arrow_scale()
        direction = f / norm
        at = self.parent.node_selection_preview(entry.get('points') or [], entry.get('regions') or [])
        if len(at) > self.MAX_MARKERS * 10:           # many nodes in a box: draw a subset of the arrows
            at = at[np.linspace(0, len(at) - 1, self.MAX_MARKERS * 10).astype(int)]
        for i, p in enumerate(at):
            arrow = pv.Arrow(start=p - direction * scale, direction=direction, scale=scale)
            actor = self.parent.plotter.add_mesh(arrow, color=LOAD_COLOR, pickable=False,
                                                 name=f"point_force_{id(entry)}_{i}")
            self.parent.force_actors.append(actor)

    def apply_force(self):
        """Apply resultant force to selected triangles"""
        if self._node_mode():
            return self.apply_force_to_points()
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return

        # Only remove force data for triangles that overlap with current selection
        selected_indices = set(tri['index'] for tri in selected_triangles)
        
        # Remove force data only for overlapping triangles
        self.parent.force_data = [
            fd for fd in self.parent.force_data
            if not selected_indices.intersection(set(fd['triangles']))
        ]
        
        # Remove only the force actors associated with overlapping triangles
        # Since we don't have a direct mapping, we'll need to be more careful here
        # For now, let's modify to only remove actors if there's complete overlap
        actors_to_remove = []
        remaining_actors = []
        
        # Check if we have stored metadata about which actors belong to which triangles
        # If not, we'll need to rebuild all visualizations
        if hasattr(self.parent, 'force_actor_triangle_mapping'):
            # If we have mapping, remove only specific actors
            for i, actor in enumerate(self.parent.force_actors):
                actor_triangles = self.parent.force_actor_triangle_mapping.get(i, set())
                if actor_triangles.intersection(selected_indices):
                    actors_to_remove.append(actor)
                    try:
                        self.parent.plotter.remove_actor(actor, reset_camera=False)
                    except Exception:
                        pass
                else:
                    remaining_actors.append(actor)
            self.parent.force_actors = remaining_actors
        else:
            # If no mapping exists, we need to rebuild all force visualizations
            # This is a safer approach but less efficient
            for actor in self.parent.force_actors:
                try:
                    self.parent.plotter.remove_actor(actor, reset_camera=False)
                except Exception:
                    pass
            self.parent.force_actors = []
            
            # Rebuild visualizations for all remaining force data
            for fd in self.parent.force_data:
                if fd.get('type') == 'force_xyz':
                    self._rebuild_force_visualization(fd)

        forces = self.get_force_values()
        if not self.validate_forces(forces):
            return

        # Convert to base units
        converted_forces = self.convert_forces_to_base_units(forces)
        force_vec = np.array([converted_forces['X'], converted_forces['Y'], converted_forces['Z']])
        nonzero = [k for k, v in converted_forces.items() if abs(v) > 1e-12]
        norm = np.linalg.norm(force_vec)
        if norm == 0:
            return

        unit_direction = force_vec / norm

        # Decide label value: show signed value if only one axis, else resultant (always positive)
        if len(nonzero) == 1:
            label_value = converted_forces[nonzero[0]]
        else:
            label_value = norm

        self.visualize_force_arrows(selected_triangles, norm, unit_direction, "red", "Resultant", label_value)

        self.store_force_data(selected_triangles, converted_forces)
        self.update_force(selected_triangles, [f"Resultant: {label_value:+.1f}"])

    def _rebuild_force_visualization(self, force_data):
        """Rebuild visualization for a force data entry"""
        if force_data.get('points') or force_data.get('regions'):
            return self.draw_point_force(force_data)
        forces = {
            'X': force_data['force_x'],
            'Y': force_data['force_y'], 
            'Z': force_data['force_z']
        }
        
        force_vec = np.array([forces['X'], forces['Y'], forces['Z']])
        nonzero = [k for k, v in forces.items() if abs(v) > 1e-12]
        norm = np.linalg.norm(force_vec)
        
        if norm == 0:
            return
        
        unit_direction = force_vec / norm
        
        # Decide label value
        if len(nonzero) == 1:
            label_value = forces[nonzero[0]]
        else:
            label_value = norm
        
        # Reconstruct triangle data from stored info
        selected_triangles = force_data['triangle_data']
        
        self.visualize_force_arrows(selected_triangles, norm, unit_direction, "red", "Resultant", label_value)

    def store_force_data(self, selected_triangles, forces):
        """Store force data for analysis"""
        force_info = {
            'triangles': [tri_data['index'] for tri_data in selected_triangles],
            'triangle_data': selected_triangles,
            'force_x': forces['X'],
            'force_y': forces['Y'], 
            'force_z': forces['Z'],
            'load_set': self.load_set_spin.value(),
            'type': 'force_xyz'
        }
        self.parent.force_data.append(force_info)

    def update_force(self, selected_triangles, applied_forces):
        """update force application - clear highlights, update state, show message"""
        #Clear highlights
        for tri_data in selected_triangles:
            self.parent.stl_geom.tri_highlight[tri_data['index']] = False
        
        self.parent.update_highlights()
        
        #Show success message
        force_unit = self.parent.settings.get_force_unit_string()
        force_components = ", ".join(applied_forces)
        self.parent.message_text.append(
            f"Applied forces ({force_components}) {force_unit} to {len(selected_triangles)} triangles."
        )
        
        #Update state and clear inputs
        self.parent.update_LivVar('structural_loads.forces_applied', True)
        self.clear_force_inputs()

        #Notify display options window
        self.parent.notify_display_options_update()

    def visualize_force_arrows(self, selected_triangles, force_value, direction_vector, color, axis_name, label_value=None):
        """Create directional arrow visualization for forces"""
        display_triangles = self.get_display_triangles(selected_triangles)
        if not display_triangles:
            return

        arrow_scale = self.calculate_arrow_scale()
        actual_direction = np.array(direction_vector) * (1.0 if force_value >= 0 else -1.0)
        centers = np.array([tri_data['center'] for tri_data in display_triangles])
        arrows = self.create_arrow_mesh(centers, actual_direction, arrow_scale)
        force_actor = self.parent.plotter.add_mesh(
            arrows,
            color=color,
            show_edges=False,
            name=f'{axis_name.lower()}_force_arrows_{len(self.parent.force_actors)}'
        )
        self.parent.force_actors.append(force_actor)

        # Pass label_value to the label function
        self.add_force_text_label(display_triangles[0], label_value if label_value is not None else force_value, actual_direction, color, axis_name)
        self.parent.plotter.render()

    

    def apply_torque(self):
        """Apply torque to selected planar or cylindrical surfaces with visualization and unit conversion."""
        if self._node_mode():
            QtWidgets.QMessageBox.warning(self, "Torque on nodes",
                                          "A torque needs a face (its axis and radius). Use Facet or Triangle mode.")
            return
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return

        # Remove previous torque data and actors for these triangles
        selected_indices = set(tri['index'] for tri in selected_triangles)
        self.parent.force_data = [
            fd for fd in self.parent.force_data
            if not (fd.get('type') == 'torque' and selected_indices.intersection(set(fd['triangles'])))
        ]
        if hasattr(self.parent, 'torque_actors'):
            for actor in self.parent.torque_actors:
                self.parent.plotter.remove_actor(actor, reset_camera=False)
            self.parent.torque_actors = []
        else:
            self.parent.torque_actors = []

        torque_value = self.torque_magnitude_spin.value()
        if abs(torque_value) < 1e-12:
            QtWidgets.QMessageBox.warning(self, "Zero Torque", "Please enter a non-zero torque value.")
            return

        # Use STLGeom's assign_highlighted_triangles_to_group for surface info
        stl = self.parent.stl_geom
        surface_type, avg_normal, area, cyl_axis, axis_point, cyl_radius = stl.assign_highlighted_triangles_to_group(
            group=1, stl_verbose=False
        )
        if surface_type not in ["PLANAR", "CYLINDER"]:
            QtWidgets.QMessageBox.warning(self, "Error", "Torque can only be applied to planar or cylindrical surfaces.")
            return

        # Direction: normal for planar, axis for cylinder
        direction = avg_normal if surface_type == "PLANAR" else cyl_axis
        direction = np.array(direction)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            QtWidgets.QMessageBox.warning(self, "Error", "Invalid torque direction.")
            return
        direction = direction / norm

        # Store torque data
        torque_info = {
            'triangles': [tri['index'] for tri in selected_triangles],
            'triangle_data': selected_triangles,
            'torque': torque_value,
            'direction': direction.tolist(),
            'axis_point': axis_point,
            'surface_type': surface_type,
            'type': 'torque',
            'load_set': self.load_set_spin.value()
        }
        self.parent.force_data.append(torque_info)

        # Visualization
        self.visualize_torque(axis_point, direction, torque_value, surface_type, cyl_radius)

        # Message and state update
        force_unit = self.parent.settings.get_force_unit_string()
        self.parent.message_text.append(
            f"Applied torque ({torque_value:+.1f} {force_unit}·m) to {len(selected_triangles)} triangles ({surface_type.lower()})"
        )
        self.parent.update_LivVar('structural_loads.forces_applied', True)
        self.torque_magnitude_spin.setValue(0)
        self.parent.notify_display_options_update()
    
    TORQUE_RADIUS_FACTOR = 0.18

    def visualize_torque(self, axis_point, direction, torque_value, surface_type, cyl_radius):
        """Visualize torque as a curved arrow (arc + conical tip) and text label."""
        plotter = self.parent.plotter
        color = "green"
        arc_degrees = 270
        arc_resolution = 36
        
        # Ensure torque_actors exists
        if not hasattr(self.parent, "torque_actors"):
            self.parent.torque_actors = []

        axis_point = np.array(axis_point, dtype=float)
        direction = np.array(direction, dtype=float)
        bbox = plotter.bounds
        model_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]) if bbox else 1.0
        if surface_type == "CYLINDER":
            if cyl_radius is None:
                cyl_radius = model_size * 0.25 
            radius = cyl_radius * self.TORQUE_RADIUS_FACTOR
        else:
            radius = model_size * self.TORQUE_RADIUS_FACTOR

        radius = np.clip(radius, model_size*0.01, model_size*0.12)

        # Find two perpendicular vectors to direction
        perp1 = np.cross(direction, [1,0,0]) if abs(direction[0]) < 0.9 else np.cross(direction, [0,1,0])
        if np.linalg.norm(perp1) < 1e-8:
            perp1 = np.cross(direction, [0,0,1])
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        perp2 /= np.linalg.norm(perp2)

        # Arc points
        arc_points = []
        for i in range(arc_resolution):
            angle = np.deg2rad(i * arc_degrees / (arc_resolution-1))
            pt = (axis_point +
                radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2))
            arc_points.append(pt)
        arc_poly = pv.lines_from_points(np.array(arc_points))
        arc_actor = plotter.add_mesh(arc_poly, color=color, line_width=4, name=f"torque_arc_{len(self.parent.torque_actors)}")
        self.parent.torque_actors.append(arc_actor)

        # Conical tip at end of arc
        tip_dir = arc_points[-1] - arc_points[-2]
        tip_dir /= np.linalg.norm(tip_dir)
        tip_start = arc_points[-1]
        cone_height = model_size * 0.02
        cone_radius = model_size * 0.010
        cone = pv.Cone(center=tip_start + tip_dir * (cone_height/2), direction=tip_dir, height=cone_height, radius=cone_radius, resolution=32)
        tip_actor = plotter.add_mesh(cone, color=color, name=f"torque_tip_{len(self.parent.torque_actors)}")
        self.parent.torque_actors.append(tip_actor)

        # Text label
        text_offset = model_size * 0.05
        text_pos = axis_point + perp1 * text_offset
        force_unit = self.parent.settings.get_force_unit_string()
        torque_text = f"{torque_value:+.1f} {force_unit}·m"
        if self.show_load_values:
            text_actor = plotter.add_point_labels(
                [text_pos], [torque_text],
                font_size=DEFAULT_FONT_SIZE, text_color=color,
                shape_opacity=0,
                name=f'torque_text_{len(self.parent.torque_actors)}'
            )
            self.parent.torque_actors.append(text_actor)
        plotter.render()

    def get_display_triangles(self, selected_triangles):
        """Get triangles to display based on threshold system"""
        if len(selected_triangles) > self.ARROW_THRESHOLD:
            step = len(selected_triangles) // self.MAX_MARKERS
            indices = range(0, len(selected_triangles), step)[:self.MAX_MARKERS]
            return [selected_triangles[i] for i in indices]
        return selected_triangles

    def calculate_arrow_scale(self):
        """Calculate appropriate arrow scale based on model size"""
        bounds = self.parent.plotter.bounds
        if bounds:
            model_size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            return model_size * self.ARROW_SCALE_FACTOR
        return 1.0

    def create_arrow_mesh(self, centers, direction, scale):
        """Create arrow mesh for visualization"""
        directions = np.tile(direction, (len(centers), 1))
        scales = np.full(len(centers), scale)
        
        arrows = pv.PolyData(centers)
        arrows['vectors'] = directions * scales.reshape(-1, 1)
        
        return arrows.glyph(orient='vectors', scale='vectors', factor=1.0)

    def add_force_text_label(self, triangle, force_value, direction, color, axis_name):
        """Add text label for force visualization at the tip of the first arrow"""
        if not self.show_load_values:
            return
        # Calculate arrow scale and offset
        geom_size = self.calculate_arrow_scale() / self.ARROW_SCALE_FACTOR
        text_offset = 0.12 * geom_size

        # Normalize direction for text placement
        magnitude = np.linalg.norm(direction)
        if magnitude > 0:
            unit_dir = direction / magnitude
            # Arrow tip position: center + direction * arrow_scale
            arrow_tip = np.array(triangle['center']) + unit_dir * self.calculate_arrow_scale()
            text_pos = arrow_tip.tolist()

            # Create and add text
            force_unit = self.parent.settings.get_force_unit_string()
            force_text = f"{force_value:+.1f} {force_unit}"

            text_actor = self.parent.plotter.add_point_labels(
                [text_pos], [force_text],
                font_size=DEFAULT_FONT_SIZE, text_color=color,
                shape_opacity=0,
                name=f'{axis_name.lower()}_force_text_{len(self.parent.force_actors)}'
            )
            self.parent.force_actors.append(text_actor)

    def apply_constraint(self, config):
        """constraint application method"""
        if self._node_mode():
            return self.apply_constraint_to_points(config)
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return
        
        # Extract triangle indices
        selected_faces = [tri_data['index'] for tri_data in selected_triangles]
        
        # Apply constraint
        self.apply_constraint_to_triangles(selected_faces, config["constraint_type"])
        
        # Update visualizations and state
        self.update_constraint_application(selected_faces, config["constraint_type"])

    def apply_constraint_to_triangles(self, triangle_ids, constraint_type):
        """Apply constraint to specific triangles"""
        # Add to constrained set
        self.parent.constraint_data = [
        c for c in self.parent.constraint_data
        if not (c.get('triangles') and set(c['triangles']).intersection(triangle_ids))
        ]
        # Add to constrained set
        self.parent.constrained_triangles.update(triangle_ids)
        # Clear highlights
        for face_id in triangle_ids:
            self.parent.stl_geom.tri_highlight[face_id] = False
        # Store constraint data with triangle info
        self.parent.constraint_data.append({'type': constraint_type, 'triangles': list(triangle_ids)})

    def update_constraint_application(self, triangle_ids, constraint_type):
        """update constraint application"""
        #Update visualizations
        self.parent.update_highlights()
        self.visualize_constraints()
        
        #Update state
        self.parent.update_LivVar('structural_loads.fixed_constraints', True)
        
        #Show message
        self.parent.message_text.append(
            f"Applied {constraint_type} constraint to {len(triangle_ids)} triangles."
        )

        #Notify display options window
        self.parent.notify_display_options_update()

    def visualize_constraints(self):
        """Create black visualization for constrained triangles, and black nodes for point supports"""
        point_supports = [p for c in self.parent.constraint_data if (c.get('points') or c.get('regions'))
                          for p in self.parent.node_selection_preview(c.get('points') or [], c.get('regions') or [])]
        if not self.parent.stl_geom or not (self.parent.constrained_triangles or point_supports):
            return

        #Remove old constraint actors
        self.remove_constraint_actors()
        
        #Create constraint mesh
        constraint_mesh = self.create_constraint_mesh()
        if constraint_mesh:
            self.add_constraint_visualization(constraint_mesh)
        if point_supports:
            actor = self.parent.plotter.add_mesh(pv.PolyData(np.asarray(point_supports, dtype=float)), color=SUPPORT_COLOR,
                                                 point_size=16, render_points_as_spheres=True, pickable=False,
                                                 name="point_supports")
            self.parent.constraint_actors.append(actor)
        
        self.parent.plotter.render()

    def remove_constraint_actors(self):
        """Remove existing constraint actors"""
        for actor in self.parent.constraint_actors:
            self.parent.plotter.remove_actor(actor, reset_camera=False)
        self.parent.constraint_actors.clear()

    def create_constraint_mesh(self):
        """Create mesh for constrained triangles"""
        vertices, faces = [], []
        vertex_count = 0
        
        for tri_id in self.parent.constrained_triangles:
            triangle_vertices = self.parent.stl_geom.mesh.vectors[tri_id]
            
            vertices.extend(triangle_vertices)
            faces.extend([3, vertex_count, vertex_count + 1, vertex_count + 2])
            vertex_count += 3
        
        if not vertices:
            return None
        
        #Create PyVista mesh
        vertices = np.array(vertices)
        faces = np.array(faces)
        return pv.PolyData(vertices, faces).compute_normals(cell_normals=True, point_normals=True)

    def add_constraint_visualization(self, constraint_mesh):
        """Add constraint visualization to plotter"""
        #Calculate offset
        bounds = self.parent.plotter.bounds
        offset = (self.CONSTRAINT_OFFSET_FACTOR * 
                 ((bounds[1]-bounds[0])**2 + (bounds[3]-bounds[2])**2 + (bounds[5]-bounds[4])**2)**0.5 
                 if bounds else self.CONSTRAINT_OFFSET_FACTOR)
        
        #Create offset mesh
        offset_mesh = constraint_mesh.copy()
        offset_mesh.points = offset_mesh.points + offset_mesh.point_data['Normals'] * offset
        
        #Add visualization
        constraint_actor = self.parent.plotter.add_mesh(
            offset_mesh,
            color=SUPPORT_COLOR,
            opacity=1.0,
            culling=False,
            lighting=False                                   # flat colour: reads as the support colour inside holes
        )
        self.parent.constraint_actors.append(constraint_actor)
    
    def update_units(self):
        """Update force spinbox prefixes and units."""
        force_unit = self.parent.settings.get_force_unit_string()
        for axis, spinbox in self.force_spinboxes.items():
            spinbox.setPrefix(f"{axis}: ")
            spinbox.setSuffix(f" {force_unit}")
#---------------------------------------------------------------------------
class ThermalLoadsWindow(QtWidgets.QDialog):
    ARROW_THRESHOLD = 25
    MAX_MARKERS = 5
    ARROW_SCALE_FACTOR = 0.1
    TEXT_OFFSET_FACTOR = 0.12
    
    # Selection mode configuration
    SELECTION_MODES = {
        "Facet": "coarse",
        "Triangle": "triangle"
    }
    
    # Thermal load type configuration
    THERMAL_TYPES = {
        "Temperature": {
            "handler": "apply_temperature",
            "group": "temp_group",
            "value_field": "temp_value_spin",
            "data_key": "temperature",
            "storage_list": "fixed_temps",
            "actor_list": "fixed_temp_actors",
            "color": "blue",
            "label": "Temp",
            "type_name": "fixed_temperature",
            "default_value": 300,
            "range": (0, 2000),
            "decimals": 1,
            "arrow_direction": "inward"  # Special handling for temperature
        },
        "Heat Flux": {
            "handler": "apply_heat_flux",
            "group": "heat_flux_group", 
            "value_field": "heat_flux_value_spin",
            "data_key": "heat_flux",
            "storage_list": "heat_sources",
            "actor_list": "heat_source_actors",
            "color": "orange",
            "label": "Heat Flux",
            "type_name": "heat_flux",
            "default_value": 1000,
            "range": (-1e6, 1e6),
            "decimals": 1,
            "arrow_direction": "outward"
        },
        "Total Heat": {
            "handler": "apply_total_heat",
            "group": "total_heat_group",
            "value_field": "total_heat_value_spin", 
            "data_key": "total_heat",
            "storage_list": "total_heat_sources",
            "actor_list": "total_heat_actors",
            "color": "red",
            "label": "Total Heat",
            "type_name": "total_heat",
            "default_value": 100,
            "range": (-1e6, 1e6),
            "decimals": 1,
            "arrow_direction": "outward"
        }
    }

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_window()
        self.init_data_structures()
        self.init_ui()
        self.connect_events()

    def init_window(self):
        """Initialize window properties"""
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setWindowTitle("Thermal Loads")

    def init_data_structures(self):
        """Initialize thermal loads data structures and actor lists"""
        self.thermal_loads = {
            "fixed_temps": [],
            "heat_sources": [], 
            "total_heat_sources": []
        }
        
        # Initialize actor lists dynamically from configuration
        for config in self.THERMAL_TYPES.values():
            setattr(self, config["actor_list"], [])

    def init_ui(self):
        """Initialize user interface"""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        self.create_selection_control(layout)
        self.create_thermal_type_control(layout)
        self.create_thermal_groups(layout)
        self.create_buttons(layout)

    def create_selection_control(self, parent_layout):
        """Create selection mode control"""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Selection"))
        
        self.selection_combo = QtWidgets.QComboBox()
        self.selection_combo.addItems(list(self.SELECTION_MODES.keys()))
        layout.addWidget(self.selection_combo)
        
        parent_layout.addLayout(layout)

    def create_thermal_type_control(self, parent_layout):
        """Create thermal type control"""
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(QtWidgets.QLabel("Thermal Type"))
        
        self.thermal_type = QtWidgets.QComboBox()
        self.thermal_type.addItems(list(self.THERMAL_TYPES.keys()))
        layout.addWidget(self.thermal_type)
        
        parent_layout.addLayout(layout)

    def create_thermal_groups(self, parent_layout):
        """Create all thermal input groups"""
        # Temperature group
        self.temp_group = self.create_temperature_group()
        parent_layout.addWidget(self.temp_group)
        
        # Heat Flux group  
        self.heat_flux_group = self.create_heat_flux_group()
        parent_layout.addWidget(self.heat_flux_group)
        
        # Total Heat group
        self.total_heat_group = self.create_total_heat_group()
        parent_layout.addWidget(self.total_heat_group)
        
        # Set initial visibility
        self.update_group_visibility("Temperature")

    def create_temperature_group(self):
        """Create temperature input group"""
        temp_unit = self.parent.settings.get_temperature_unit_symbol()
        config = self.THERMAL_TYPES["Temperature"]
        
        group = QtWidgets.QGroupBox("Temperature")
        layout = QtWidgets.QVBoxLayout(group)
        
        value_layout = QtWidgets.QHBoxLayout()
        value_layout.addWidget(QtWidgets.QLabel(f"Temperature ({temp_unit})"))
        
        self.temp_value_spin = self.create_spinbox(config)
        value_layout.addWidget(self.temp_value_spin)
        layout.addLayout(value_layout)
        
        return group

    def create_heat_flux_group(self):
        """Create heat flux input group"""
        length_unit = self.parent.settings.get_length_unit_string()
        config = self.THERMAL_TYPES["Heat Flux"]
        
        group = QtWidgets.QGroupBox("Heat Flux")
        layout = QtWidgets.QVBoxLayout(group)
        
        value_layout = QtWidgets.QHBoxLayout()
        value_layout.addWidget(QtWidgets.QLabel(f"Heat Flux (W/{length_unit}²)"))
        
        self.heat_flux_value_spin = self.create_spinbox(config)
        value_layout.addWidget(self.heat_flux_value_spin)
        layout.addLayout(value_layout)
        
        return group

    def create_total_heat_group(self):
        """Create total heat input group"""
        config = self.THERMAL_TYPES["Total Heat"]
        
        group = QtWidgets.QGroupBox("Total Heat")
        layout = QtWidgets.QVBoxLayout(group)
        
        value_layout = QtWidgets.QHBoxLayout()
        value_layout.addWidget(QtWidgets.QLabel("Total Heat (W)"))
        
        self.total_heat_value_spin = self.create_spinbox(config)
        value_layout.addWidget(self.total_heat_value_spin)
        layout.addLayout(value_layout)
        
        return group

    def create_spinbox(self, config):
        """Create a spinbox with configuration"""
        spinbox = QtWidgets.QDoubleSpinBox()
        spinbox.setRange(*config["range"])
        spinbox.setValue(config["default_value"])
        spinbox.setDecimals(config["decimals"])
        return spinbox

    def create_buttons(self, parent_layout):
        """Create control buttons"""
        layout = QtWidgets.QHBoxLayout()
        
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self.on_apply_clicked)
        
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        
        layout.addWidget(apply_button)
        layout.addWidget(close_button)
        parent_layout.addLayout(layout)

    def connect_events(self):
        """Connect event handlers"""
        self.selection_combo.currentIndexChanged.connect(self.on_selection_mode_changed)
        self.thermal_type.currentTextChanged.connect(self.on_thermal_type_changed)
        
        # Set initial state
        self.on_selection_mode_changed(0)

    def on_selection_mode_changed(self, index):
        """Handle selection mode changes"""
        mode_text = self.selection_combo.currentText()
        self.parent.highlight_mode = self.SELECTION_MODES[mode_text]

    def on_selection_mode_changed(self, index):
        """Handle selection mode changes (make it work like structural loads)"""
        mode_text = self.selection_combo.currentText()
        self.parent.highlight_mode = self.SELECTION_MODES[mode_text]
        if self.parent.stl_geom:
            # Turn ON triangle edges in "Triangle" mode, OFF otherwise
            for name, actor in self.parent.plotter.actors.items():
                if hasattr(actor, 'GetProperty'):
                    prop = actor.GetProperty()
                    if self.parent.highlight_mode == "triangle":
                        if hasattr(prop, 'EdgeVisibilityOn'):
                            prop.EdgeVisibilityOn()
                            prop.SetEdgeColor(0, 0, 0)
                            prop.SetLineWidth(1)
                    else:
                        if hasattr(prop, 'EdgeVisibilityOff'):
                            prop.EdgeVisibilityOff()
            if self.parent.highlight_mode == "triangle":
                self.parent.update_highlights()
            else:
                if getattr(self.parent, "highlight_actor", None):
                    self.parent.plotter.remove_actor(self.parent.highlight_actor, reset_camera=False)
                    self.parent.highlight_actor = None
                self.parent.plotter.render()
    
    def on_thermal_type_changed(self, text):
        """Update input group visibility when thermal type changes"""
        self.update_group_visibility(text)

    def update_group_visibility(self, thermal_type):
        """Update visibility of thermal input groups"""
        # Hide all groups first
        self.temp_group.setVisible(False)
        self.heat_flux_group.setVisible(False)
        self.total_heat_group.setVisible(False)
        
        # Show the relevant group
        if thermal_type == "Temperature":
            self.temp_group.setVisible(True)
        elif thermal_type == "Heat Flux":
            self.heat_flux_group.setVisible(True)
        elif thermal_type == "Total Heat":
            self.total_heat_group.setVisible(True)

    def on_apply_clicked(self):
        """apply button handler"""
        thermal_type = self.thermal_type.currentText()
        config = self.THERMAL_TYPES[thermal_type]
        handler_name = config["handler"]
        
        handler_method = getattr(self, handler_name, None)
        if handler_method and callable(handler_method):
            handler_method(config)

    def get_selected_triangles(self):
        """Get selected triangles"""
        selected_triangles = self.parent.stl_geom.store_selected_triangles()
        
        if not selected_triangles:
            QtWidgets.QMessageBox.warning(self, "No Selection", 
                                         "No triangles selected for thermal load application.")
            return None
        
        return selected_triangles

    def get_spinbox_value(self, config):
        """Get value from the appropriate spinbox"""
        field_name = config["value_field"]
        spinbox = getattr(self, field_name)
        return spinbox.value()

    def store_thermal_data(self, selected_triangles, value, config):
        """Store thermal load data"""
        thermal_info = {
            'triangles': [tri_data['index'] for tri_data in selected_triangles],
            'triangle_data': selected_triangles,
            config["data_key"]: value,
            'type': config["type_name"]
        }
        
        storage_list = self.thermal_loads[config["storage_list"]]
        storage_list.append(thermal_info)

    def create_thermal_visualization(self, selected_triangles, value, config):
        """Create thermal arrows and text visualization"""
        display_triangles = self.get_display_triangles(selected_triangles)
        
        if not display_triangles:
            return
        
        arrow_scale = self.calculate_arrow_scale()
        centers, normals = self.extract_triangle_data(display_triangles)
        
        # Create arrows with direction based on config
        arrow_centers, arrow_directions = self.calculate_arrow_vectors(
            centers, normals, arrow_scale, config["arrow_direction"]
        )
        
        # Create and add arrow visualization
        thermal_actor = self.create_arrow_mesh(arrow_centers, arrow_directions, config)
        self.store_actor(thermal_actor, config)
        
        # Add text label
        self.add_thermal_text_label(display_triangles[0], value, config)
        
        self.parent.plotter.render()

    def get_display_triangles(self, selected_triangles):
        """Get triangles to display based on threshold system"""
        if len(selected_triangles) > self.ARROW_THRESHOLD:
            step = len(selected_triangles) // self.MAX_MARKERS
            indices = range(0, len(selected_triangles), step)[:self.MAX_MARKERS]
            return [selected_triangles[i] for i in indices]
        return selected_triangles

    def calculate_arrow_scale(self):
        """Calculate appropriate arrow scale based on model size"""
        bounds = self.parent.plotter.bounds
        if bounds:
            model_size = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
            return model_size * self.ARROW_SCALE_FACTOR
        return 1.0

    def extract_triangle_data(self, display_triangles):
        """Extract centers and normals from triangle data"""
        centers = np.array([tri_data['center'] for tri_data in display_triangles])
        normals = np.array([tri_data['normal'] for tri_data in display_triangles])
        return centers, normals

    def calculate_arrow_vectors(self, centers, normals, arrow_scale, direction_type):
        """Calculate arrow positions and directions based on type"""
        if direction_type == "inward":
            # Temperature arrows point inward (reversed) and are offset
            arrow_directions = -normals * arrow_scale
            arrow_centers = centers + normals * arrow_scale
        else:
            # Heat flux and total heat arrows point outward
            arrow_directions = normals * arrow_scale
            arrow_centers = centers
        
        return arrow_centers, arrow_directions

    def create_arrow_mesh(self, arrow_centers, arrow_directions, config):
        """Create arrow mesh for visualization"""
        arrows = pv.PolyData(arrow_centers)
        arrows['vectors'] = arrow_directions
        arrow_glyph = arrows.glyph(orient='vectors', scale='vectors', factor=1.0)
        
        actor_count = len(getattr(self, config["actor_list"]))
        name = f'{config["label"].lower().replace(" ", "_")}_arrows_{actor_count}'
        
        return self.parent.plotter.add_mesh(
            arrow_glyph,
            color=config["color"],
            show_edges=False,
            name=name
        )

    def store_actor(self, actor, config):
        """Store actor in appropriate list"""
        actor_list = getattr(self, config["actor_list"])
        actor_list.append(actor)

    def add_thermal_text_label(self, triangle, value, config):
        """Add text label for thermal visualization"""
        arrow_scale = self.calculate_arrow_scale()
        
        # Get triangle normal for positioning
        normal = np.array(triangle['normal'])
        center = np.array(triangle['center'])
        
        # Calculate text position with more prominent offset
        if config["arrow_direction"] == "inward":
            # Temperature: place text clearly visible above the surface
            text_pos = center + normal * (arrow_scale * 1.5)
        else:
            # Heat flux and total heat: place text at the arrow tip
            text_pos = center + normal * (arrow_scale * 1.5)
        
        # Ensure text position is a list for PyVista
        text_pos = text_pos.tolist()
        
        # Create text with appropriate units
        text = self.format_thermal_text(value, config)
        
        # Add text label with enhanced visibility
        actor_count = len(getattr(self, config["actor_list"]))
        name = f'{config["label"].lower().replace(" ", "_")}_text_{actor_count}'
        
        try:
            text_actor = self.parent.plotter.add_point_labels(
                [text_pos], [text],
                font_size=16,  # Even larger font
                text_color=config["color"],
                name=name,
                always_visible=True,
                shape_opacity=0,
                pickable=False  # Prevent interference with triangle picking
            )
            
            # Store text actor
            self.store_actor(text_actor, config)
            #print(f"Successfully created text actor for {config['label']}")
            
        except Exception as e:
            #print(f"Error creating text label for {config['label']}: {e}")
            # Fallback: try simpler text creation
            text_actor = self.parent.plotter.add_point_labels(
                [text_pos], [text],
                font_size=16,
                text_color=config["color"]
            )
            self.store_actor(text_actor, config)

    def format_thermal_text(self, value, config):
        """Format text display based on thermal type"""
        label = config["label"]
        
        if label == "Temp":
            temp_unit = self.parent.settings.get_temperature_unit_symbol()
            return f"{value:.1f} {temp_unit}"
        elif label == "Heat Flux":
            length_unit = self.parent.settings.get_length_unit_string()
            return f"{value:.1f} W/{length_unit}²"
        else:  # Total Heat
            return f"{value:.1f} W"

    def clear_selection(self, selected_triangles):
        """Clear selection after applying thermal load"""
        for tri_data in selected_triangles:
            self.parent.stl_geom.tri_highlight[tri_data['index']] = False
        self.parent.update_highlights()

    def show_success_message(self, thermal_type, value, triangle_count):
        """Show success message for applied thermal load"""
        config = self.THERMAL_TYPES[thermal_type]
        text = self.format_thermal_text(value, config)
        
        if thermal_type == "Temperature":
            message = f"Applied temperature {text} to {triangle_count} triangles."
        elif thermal_type == "Heat Flux":
            message = f"Applied heat flux {text} to {triangle_count} triangles."
        else:  # Total Heat
            message = f"Applied total heat {text} to {triangle_count} triangles."
        
        self.parent.message_text.append(message)

    # thermal load application methods
    def apply_temperature(self, config):
        """Apply fixed temperature to selected triangles"""
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return
        
        value = self.get_spinbox_value(config)
        
        self.store_thermal_data(selected_triangles, value, config)
        self.create_thermal_visualization(selected_triangles, value, config)
        self.clear_selection(selected_triangles)
        
        self.parent.update_LivVar('thermal_loads.applied', True)
        self.show_success_message("Temperature", value, len(selected_triangles))

        # Notify display options window
        self.parent.notify_display_options_update()

    def apply_heat_flux(self, config):
        """Apply heat flux to selected triangles"""
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return
        
        value = self.get_spinbox_value(config)
        
        self.store_thermal_data(selected_triangles, value, config)
        self.create_thermal_visualization(selected_triangles, value, config)
        self.clear_selection(selected_triangles)
        
        self.parent.update_LivVar('thermal_loads.applied', True)
        self.show_success_message("Heat Flux", value, len(selected_triangles))

        # Notify display options window
        self.parent.notify_display_options_update()

    def apply_total_heat(self, config):
        """Apply total heat to selected triangles"""
        selected_triangles = self.get_selected_triangles()
        if not selected_triangles:
            return
        
        value = self.get_spinbox_value(config)
        
        self.store_thermal_data(selected_triangles, value, config)
        self.create_thermal_visualization(selected_triangles, value, config)
        self.clear_selection(selected_triangles)
        
        self.parent.update_LivVar('thermal_loads.applied', True)
        self.show_success_message("Total Heat", value, len(selected_triangles))

        # Notify display options window
        self.parent.notify_display_options_update()

    def update_units(self):
        """Update thermal spinbox prefixes and units."""
        temp_unit = self.parent.settings.get_temperature_unit_symbol()
        length_unit = self.parent.settings.get_length_unit_string()
        
        # Update temperature spinbox
        self.temp_value_spin.setPrefix(f"Temperature ({temp_unit}): ")
        self.temp_value_spin.setSuffix(f" {temp_unit}")
        
        # Update heat flux spinbox
        self.heat_flux_value_spin.setPrefix(f"Heat Flux (W/{length_unit}²): ")
        self.heat_flux_value_spin.setSuffix(f" W/{length_unit}²")
        
        # Update total heat spinbox
        self.total_heat_value_spin.setPrefix("Total Heat (W): ")
        self.total_heat_value_spin.setSuffix(" W")
#---------------------------------------------------------------------------
class BodyForceWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Body Force")
        self.setFixedSize(260, 140)
        self.parent = parent

        # Ensure parent has storage
        if not hasattr(self.parent, "body_force"):
            self.parent.body_force = {"X": 0.0, "Y": 0.0, "Z": 0.0}
        if not hasattr(self.parent, "body_force_actors"):
            self.parent.body_force_actors = []

        layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()
        self.x_edit = QtWidgets.QLineEdit(str(self.parent.body_force.get("X", 0.0)))
        self.y_edit = QtWidgets.QLineEdit(str(self.parent.body_force.get("Y", 0.0)))
        self.z_edit = QtWidgets.QLineEdit(str(self.parent.body_force.get("Z", 0.0)))
        form_layout.addRow("X (m/s^2):", self.x_edit)
        form_layout.addRow("Y (m/s^2):", self.y_edit)
        form_layout.addRow("Z (m/s^2):", self.z_edit)
        layout.addLayout(form_layout)

        btn_layout = QtWidgets.QHBoxLayout()
        apply_btn = QtWidgets.QPushButton("Apply")
        apply_btn.clicked.connect(self.apply_body_force)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def apply_body_force(self):
        try:
            ax = float(self.x_edit.text())
            ay = float(self.y_edit.text())
            az = float(self.z_edit.text())
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please enter valid numbers for X, Y, Z.")
            return

        self.parent.body_force = {"X": ax, "Y": ay, "Z": az}
        self.visualize_body_force(ax, ay, az)
        self.parent.message_text.append(
            f"Body force set: [{ax:.3f}, {ay:.3f}, {az:.3f}] m/s²"
        )
        self.parent.update_LivVar('structural_loads.body_force', True)
        self.parent.set_sidebar_icon("Structural Loads", "check")
        self.parent.set_sidebar_icon("Analysis", "arrow")
        self.close()

    BODY_FORCE_SCALE_FACTOR = 0.25   # arrow length as a fraction of model size
    BODY_FORCE_COLOR = "green"

    def clear_body_force_actors(self):
        """Remove any previously drawn body-force arrow/label actors."""
        for name in list(getattr(self.parent, "body_force_actors", [])):
            try:
                self.parent.plotter.remove_actor(name, reset_camera=False)
            except Exception:
                pass
        self.parent.body_force_actors = []

    def visualize_body_force(self, ax, ay, az):
        """Draw a single arrow through the model centre showing the body-force vector.

        A body force acts on the whole domain, so it is drawn once at the centre of
        the model rather than per-face like the surface tractions in
        StructuralLoadsWindow.
        """
        self.clear_body_force_actors()

        vec = np.array([ax, ay, az], dtype=float)
        magnitude = float(np.linalg.norm(vec))
        if magnitude <= 0.0:
            self.parent.plotter.render()
            return

        bounds = self.parent.plotter.bounds
        if not bounds:
            return
        model_size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        if model_size <= 0:
            model_size = 1.0
        arrow_scale = model_size * self.BODY_FORCE_SCALE_FACTOR

        centre = np.array([
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ])
        unit_dir = vec / magnitude
        tail = centre - unit_dir * (arrow_scale / 2.0)

        arrows = pv.PolyData(np.array([tail]))
        arrows["vectors"] = np.array([unit_dir * arrow_scale])
        glyph = arrows.glyph(orient="vectors", scale="vectors", factor=1.0)

        arrow_name = "body_force_arrow"
        self.parent.plotter.add_mesh(
            glyph, color=self.BODY_FORCE_COLOR, name=arrow_name, show_scalar_bar=False
        )
        self.parent.body_force_actors.append(arrow_name)

        label_name = "body_force_label"
        tip = tail + unit_dir * arrow_scale
        self.parent.plotter.add_point_labels(
            [tip.tolist()],
            [f"Body force: {magnitude:.3g} m/s²"],
            name=label_name,
            font_size=12,
            text_color=self.BODY_FORCE_COLOR,
            shape=None,
            show_points=False,
            always_visible=True,
        )
        self.parent.body_force_actors.append(label_name)

        self.parent.plotter.render()
#--------------------------------------------------------------------------
class AnalysisWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Analysis")
        self.setFixedSize(360, 560)
        self.parent = parent
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Load Set
        load_set_layout = QtWidgets.QHBoxLayout()
        load_set_layout.addWidget(QtWidgets.QLabel("Load Set"))
        self.load_set_spin = QtWidgets.QSpinBox()
        self.load_set_spin.setMinimum(0)
        load_set_layout.addWidget(self.load_set_spin)
        layout.addLayout(load_set_layout)
        
        # Mesh Quality
        mesh_layout = QtWidgets.QHBoxLayout()
        mesh_layout.addWidget(QtWidgets.QLabel("Mesh Quality"))
        self.mesh_combo = QtWidgets.QComboBox()
        self.mesh_combo.addItems(["Very Coarse", "Coarse", "Normal", "Fine", "Very Fine"])
        self.mesh_combo.currentTextChanged.connect(self.on_mesh_quality_changed)
        mesh_layout.addWidget(self.mesh_combo)
        layout.addLayout(mesh_layout)
        
        # Number of Elements
        elements_layout = QtWidgets.QHBoxLayout()
        elements_layout.addWidget(QtWidgets.QLabel("#Elements"))
        self.elements_spin = QtWidgets.QSpinBox()
        self.elements_spin.setRange(1000, 1000000)
        self.elements_spin.setValue(10000)
        elements_layout.addWidget(self.elements_spin)
        layout.addLayout(elements_layout)
        
        # Solver Type
        solver_layout = QtWidgets.QHBoxLayout()
        solver_layout.addWidget(QtWidgets.QLabel("Solver Type"))
        self.solver_combo = QtWidgets.QComboBox()
        self.solver_combo.addItems(["PARDISO", "DPCG", "PCG", "PYAMG", "SPSOLVE"])
        solver_layout.addWidget(self.solver_combo)
        layout.addLayout(solver_layout)
        
        # Thermal Effect
        
        # Zero-strain Temperature
        temp_layout = QtWidgets.QHBoxLayout()
        zero_label = QtWidgets.QLabel("Zero-strain T:")
        zero_label.setToolTip("Stress-free temperature for Thermo-Structural Analysis, in the unit of the thermal loads")
        temp_layout.addWidget(zero_label)
        self.temp_spin = QtWidgets.QDoubleSpinBox()
        self.temp_spin.setRange(0, 1000)
        self.temp_spin.setValue(300)
        temp_layout.addWidget(self.temp_spin)
        layout.addLayout(temp_layout)
        
        # Buttons
        self.mesh_button = QtWidgets.QPushButton("Generate Mesh")
        self.mesh_button.clicked.connect(self.generate_mesh)
        layout.addWidget(self.mesh_button)
        
        self.thermal_button = QtWidgets.QPushButton("Thermal Analysis")
        self.thermal_button.clicked.connect(self.run_thermal_analysis)
        layout.addWidget(self.thermal_button)
        
        self.structural_button = QtWidgets.QPushButton("Structural Analysis")
        self.structural_button.clicked.connect(lambda: self._run_guarded(self.run_structural_analysis, coupled=False))
        layout.addWidget(self.structural_button)

        # Thermal solve -> thermal expansion load -> structural solve (needs structural AND thermal loads)
        self.thermostructural_button = QtWidgets.QPushButton("Thermo-Structural Analysis")
        self.thermostructural_button.clicked.connect(lambda: self._run_guarded(self.run_structural_analysis, coupled=True))
        layout.addWidget(self.thermostructural_button)

        # Results: deformation / stress / temperature of the initial design (the analyses above) or of the last
        # optimized design
        group = QtWidgets.QGroupBox("Show results")
        grid = QtWidgets.QGridLayout(group)
        self.design_combo = QtWidgets.QComboBox()
        self.design_combo.addItems(["Initial design", "Optimized design"])
        self.design_combo.currentIndexChanged.connect(self.refresh_result_buttons)
        grid.addWidget(self.design_combo, 0, 0, 1, 3)
        self.field_buttons = {}
        for i, (field, label) in enumerate(zip(RESULT_FIELDS, ("Deformation", "Stress", "Temperature"))):
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(lambda _checked=False, f=field: self.show_result(f))
            grid.addWidget(b, 1, i)
            self.field_buttons[field] = b
        layout.addWidget(group)
        self.refresh_result_buttons()
        
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
    
    def on_mesh_quality_changed(self, quality):
        """Update element count based on mesh quality"""
        element_counts = {"Very Coarse": 10000, "Coarse": 25000, "Normal": 50000, "Fine": 75000, "Very Fine": 100000}
        self.elements_spin.setValue(element_counts.get(quality, 10000))
        # Clear mesh_elements so project value doesn't override user choice
        if hasattr(self.parent, 'mesh_elements'):
            delattr(self.parent, 'mesh_elements')

    def generate_mesh(self):
        """Generate mesh and visualize with element colors"""
        if not self.parent.stl_geom:
            QtWidgets.QMessageBox.warning(self, "No Geometry", "Please load geometry first.")
            return

        # Use parent's stored mesh_elements count if available
        num_elements = self.elements_spin.value()
        if hasattr(self.parent, 'mesh_elements'):
            num_elements = self.parent.mesh_elements
            self.elements_spin.setValue(num_elements)
            # Clear the stored value to avoid unexpected reuse
            delattr(self.parent, 'mesh_elements')
        
        # Create mesh. Keep the mesh only once it is complete: a half-built mesher left in parent.hex_mesh made
        # the analysis buttons fail later with "no attribute elemArray".
        try:
            mesher = HexMesher()
            mesher.createMeshFromSTLFile(self.parent.stl_geom.file_path, self.elements_spin.value())
            mesher.createEdofMatStructural()
            mesher.createEdofMatThermal()
        except Exception as e:
            self.parent.hex_mesh = None
            self.parent.message_text.append(f"Mesh generation failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Mesh generation failed", str(e))
            return
        self.parent.hex_mesh = mesher

        # Prepare mesh for FEA
        self.prepare_mesh_for_analysis(mesher, "structural")

        # Check if thermal loads exist
        if (hasattr(self.parent, 'thermal_loads_window') and 
            self.parent.thermal_loads_window and
            self.parent.thermal_loads_window.thermal_loads and
            any(self.parent.thermal_loads_window.thermal_loads.get(key, []) for key in ['fixed_temps', 'heat_sources', 'total_heat_sources'])):
            # Show thermal colored mesh
            self.visualize_colored_mesh("thermal")
        else:
            # Show structural colored mesh
            self.visualize_colored_mesh("structural")

        # Update state
        self.parent.update_LivVar('mesh_generated', True)
        self.parent.message_text.append(f"Mesh generated: {mesher.num_elems} elements, {mesher.num_nodes} nodes")

    def get_boundary_mapping_data(self):
        """Get common boundary mapping data used by multiple methods"""
        mesh = self.parent.hex_mesh
        boundary_nodes = mesh.get_boundary_nodes()
        boundary_points = mesh.node_xyz[boundary_nodes]
        tolerance = min(mesh.elem_size) * 0.8
        return boundary_nodes, boundary_points, tolerance

    def map_triangles_to_surface_nodes(self, triangle_indices, boundary_nodes=None, boundary_points=None, tolerance=None):
        """Map triangles to surface nodes - SURFACE ONLY (excludes interior nodes)"""
        if not triangle_indices:
            return set()
        
        # Use cached boundary data if not provided
        if boundary_nodes is None:
            boundary_nodes, boundary_points, tolerance = self.get_boundary_mapping_data()
        
        surface_nodes = set()
        for tri_idx in triangle_indices:
            distances = self.parent.stl_geom.find_points_triangle_distances_vectorized(boundary_points, tri_idx)
            close_mask = distances < tolerance
            surface_nodes.update(boundary_nodes[close_mask])
        
        return surface_nodes
    
    def map_bc_entry_to_nodes(self, entry, boundary_nodes=None, boundary_points=None, tolerance=None):
        """Surface nodes of a support/load entry: its picked triangles and/or its picked nodes (points)."""
        if boundary_nodes is None:
            boundary_nodes, boundary_points, tolerance = self.get_boundary_mapping_data()
        nodes = set(self.map_triangles_to_surface_nodes(entry.get('triangles') or [], boundary_nodes,
                                                        boundary_points, tolerance))
        points = entry.get('points') or []
        if len(points):
            idx, dist = nearest_surface_nodes(points, boundary_nodes, boundary_points)
            far = dist > 2.0 * max(self.parent.hex_mesh.elem_size)
            if np.any(far):
                self.parent.message_text.append(
                    f"Warning: {int(far.sum())} picked node(s) are more than two elements from the meshed part "
                    f"(max {dist.max():.3g}); they are mapped to the nearest surface node.")
            nodes.update(int(n) for n in idx)
        for region in entry.get('regions') or []:        # box selections: every surface node inside the box
            inside = points_in_region(np.asarray(boundary_points), region)
            if not inside.any():
                self.parent.message_text.append("Warning: a box selection contains no surface node of this mesh.")
            nodes.update(int(n) for n in np.asarray(boundary_nodes)[inside])
        return nodes

    def build_node_to_elem_map(self, mesh):
        """Build a mapping from node index to set of element indices."""

        node_to_elem = defaultdict(set)
        elemArray = np.asarray(mesh.elemArray)
        for elem_id, nodes in enumerate(elemArray):
            for node in nodes:
                node_to_elem[node].add(elem_id)
        return node_to_elem

    def map_triangles_to_elements(self, triangle_indices, boundary_nodes=None, boundary_points=None, tolerance=None, node_to_elem=None):
        """Map triangles to elements - SURFACE ONLY"""
        surface_nodes = self.map_triangles_to_surface_nodes(triangle_indices, boundary_nodes, boundary_points, tolerance)
        if node_to_elem is None:
            node_to_elem = self.build_node_to_elem_map(self.parent.hex_mesh)
        
        # Get boundary elements only
        boundary_elements = set(self.parent.hex_mesh.get_boundary_elements())
        
        # Union all elements containing any of the surface nodes, but only boundary elements
        surface_elements = set()
        for node_id in surface_nodes:
            elements_with_node = node_to_elem[node_id]
            # Only include elements that are boundary elements
            surface_elements.update(elements_with_node.intersection(boundary_elements))
        
        return surface_elements

    def map_triangles_to_thermal_nodes(self, triangle_indices, boundary_nodes=None, boundary_points=None, tolerance=None):
        """Map triangles to thermal nodes - SURFACE ONLY"""
        return list(self.map_triangles_to_surface_nodes(triangle_indices, boundary_nodes, boundary_points, tolerance))

    def get_element_colors(self, color_type="structural"):
        mesh = self.parent.hex_mesh
        element_colors = np.full(mesh.num_elems, 0.65)  # Default gray

        # Build node-to-element map once
        node_to_elem = self.build_node_to_elem_map(mesh)
        boundary_nodes, boundary_points, tolerance = self.get_boundary_mapping_data()

        if color_type == "structural":
            to_elements = lambda nodes: set().union(*(node_to_elem[n] for n in nodes)) if nodes else set()
            for constraint in self.parent.constraint_data:
                constrained_elements = to_elements(self.map_bc_entry_to_nodes(
                    constraint, boundary_nodes, boundary_points, tolerance))
                element_colors[list(constrained_elements)] = 0.0  # Black

            for force_info in self.parent.force_data:
                force_elements = to_elements(self.map_bc_entry_to_nodes(
                    force_info, boundary_nodes, boundary_points, tolerance))
                if force_info.get('type') == 'torque':
                    element_colors[list(force_elements)] = 0.33  # Green for torque
                else:
                    element_colors[list(force_elements)] = 1.0  # Red for force

        elif color_type == "thermal":
            if not (self.parent.thermal_loads_window and self.parent.thermal_loads_window.thermal_loads):
                return element_colors

            thermal_loads = self.parent.thermal_loads_window.thermal_loads
            thermal_color_map = {
                'fixed_temps': 0.0,      # Blue
                'heat_sources': 0.5,     # Orange  
                'total_heat_sources': 1.0 # Red
            }
            for load_type, color_value in thermal_color_map.items():
                for load_data in thermal_loads.get(load_type, []):
                    affected_elements = self.map_triangles_to_elements(
                        load_data['triangles'], boundary_nodes, boundary_points, tolerance, node_to_elem
                    )
                    element_colors[list(affected_elements)] = color_value

        return element_colors

    def visualize_colored_mesh(self, visualization_type="structural"):
        """Mesh visualization showing nodes as colored spheres for boundary conditions"""
        # Clear actors except geometry info
        for name in list(self.parent.plotter.actors.keys()):
            if name != 'geometry_info':
                self.parent.plotter.remove_actor(name, reset_camera=False)

        mesh = self.parent.hex_mesh

        boundary_nodes, boundary_points, tolerance = self.get_boundary_mapping_data()

        if visualization_type == "structural":
            constrained_nodes = set()
            loaded_nodes = set()

            # Constrained nodes (faces and point supports)
            for constraint in self.parent.constraint_data:
                constrained_nodes.update(self.map_bc_entry_to_nodes(constraint, boundary_nodes, boundary_points, tolerance))

            # Loaded nodes (faces and point loads)
            for force_info in self.parent.force_data:
                loaded_nodes.update(self.map_bc_entry_to_nodes(force_info, boundary_nodes, boundary_points, tolerance))

            # Plot mesh as wireframe
            mesh_polydata = self.create_mesh_polydata(np.full(mesh.num_elems, 0.65))
            self.parent.plotter.add_mesh(
                mesh_polydata,
                color='gray',
                show_edges=True,
                edge_color='black',
                line_width=1,
                show_scalar_bar=False,
                name="colored_mesh"
            )
        elif visualization_type == "thermal":
            fixed_temp_nodes = set()
            heat_flux_nodes = set()
            total_heat_nodes = set()

            if self.parent.thermal_loads_window and self.parent.thermal_loads_window.thermal_loads:
                thermal_loads = self.parent.thermal_loads_window.thermal_loads
                # Fixed temperature nodes (blue)
                for temp_load in thermal_loads.get('fixed_temps', []):
                    nodes = self.map_triangles_to_surface_nodes(
                        temp_load['triangles'], boundary_nodes, boundary_points, tolerance
                    )
                    fixed_temp_nodes.update(nodes)
                # Heat flux nodes (orange)
                for heat_load in thermal_loads.get('heat_sources', []):
                    nodes = self.map_triangles_to_surface_nodes(
                        heat_load['triangles'], boundary_nodes, boundary_points, tolerance
                    )
                    heat_flux_nodes.update(nodes)
                # Total heat nodes (red)
                for total_heat_load in thermal_loads.get('total_heat_sources', []):
                    nodes = self.map_triangles_to_surface_nodes(
                        total_heat_load['triangles'], boundary_nodes, boundary_points, tolerance
                    )
                    total_heat_nodes.update(nodes)

            # Plot mesh as wireframe
            mesh_polydata = self.create_mesh_polydata(np.full(mesh.num_elems, 0.65))
            self.parent.plotter.add_mesh(
                mesh_polydata,
                color='gray',
                show_edges=True,
                edge_color='black',
                line_width=1,
                show_scalar_bar=False,
                name="colored_mesh"
            )

            # Plot fixed temperature nodes (blue spheres)
            if fixed_temp_nodes:
                pts = mesh.node_xyz[list(fixed_temp_nodes)]
                self.parent.plotter.add_mesh(
                    pv.PolyData(pts),
                    color='blue',
                    point_size=5,
                    render_points_as_spheres=True,
                    name="fixed_temp_nodes"
                )

            # Plot heat flux nodes (orange spheres)
            if heat_flux_nodes:
                pts = mesh.node_xyz[list(heat_flux_nodes)]
                self.parent.plotter.add_mesh(
                    pv.PolyData(pts),
                    color='orange',
                    point_size=5,
                    render_points_as_spheres=True,
                    name="heat_flux_nodes"
                )

            # Plot total heat nodes (red spheres)
            if total_heat_nodes:
                pts = mesh.node_xyz[list(total_heat_nodes)]
                self.parent.plotter.add_mesh(
                    pv.PolyData(pts),
                    color='red',
                    point_size=5,
                    render_points_as_spheres=True,
                    name="total_heat_nodes"
                )

        self.parent.plotter.reset_camera()

    def create_mesh_polydata(self, element_colors):
        mesh = self.parent.hex_mesh
        num_elems = mesh.num_elems
        elemArray = np.asarray(mesh.elemArray, dtype=np.int64)
        cells = np.hstack([np.full((num_elems, 1), 8, dtype=np.int64), elemArray])
        cells = cells.flatten()
        grid = pv.UnstructuredGrid(cells, [pv.CellType.HEXAHEDRON] * num_elems, mesh.node_xyz)
        grid.cell_data['element_colors'] = element_colors
        return grid

    def prepare_mesh_for_analysis(self, mesh, analysis_type="structural"):
        """Mesh preparation for both structural and thermal FEA"""
        # Common attributes for both analysis types
        mesh.elemPseudoDensity = np.ones(mesh.num_elems)
        mesh.elemNeighborsArray = np.zeros((mesh.num_elems, 6), dtype=int)
        mesh.node_indices = np.zeros((mesh.num_nodes, 4), dtype=int)
        mesh.elemComponentId = np.zeros(mesh.num_elems, dtype=int)
      
        if analysis_type == "thermal":
            # Thermal: 1 DOF per node
            mesh.edofMat = np.zeros((mesh.num_elems, 8), dtype=int)
            for elem_id in range(mesh.num_elems):
                mesh.edofMat[elem_id] = mesh.elemArray[elem_id]

            # Create node_idx for thermal FEA
            row_indices, col_indices = [], []
            for elem_id in range(mesh.num_elems):
                element_nodes = mesh.elemArray[elem_id]
                for node_i in element_nodes:
                    for node_j in element_nodes:
                        row_indices.append(node_i)
                        col_indices.append(node_j)
            mesh.node_idx = np.column_stack((row_indices, col_indices))

        else:  # structural
            # Structural: 3 DOFs per node
            mesh.edofMat = np.zeros((mesh.num_elems, 24), dtype=int)
            for elem_id in range(mesh.num_elems):
                element_nodes = mesh.elemArray[elem_id]
                dofs = []
                for node in element_nodes:
                    dofs.extend([3*node, 3*node+1, 3*node+2])
                mesh.edofMat[elem_id] = dofs
  
    def create_material_properties(self):
        """Create material properties object for both analysis types"""
        material_props = self.parent.applied_material['properties']
        return mat_lib.Material(
            name="Applied_Material",
            youngs_modulus=material_props['Young\'s Modulus'],
            poissons_ratio=material_props['Poisson\'s Ratio'],
            mass_density=material_props['Density'],
            thermal_conductivity=material_props['Thermal Conductivity'],
            specific_heat=material_props['Specific Heat Capacity'],
            thermal_expansion_coefficient=material_props['Thermal Expansion'],
            cost=material_props['Price'],
            yield_strength=material_props['Yield Strength']
        )

    def get_solver(self):
        """Get solver based on current selection"""
        solver_map = {
            "PARDISO": linear_solvers.Solvers.PARDISO,
            "DPCG": linear_solvers.Solvers.DPCG,
            "PCG": linear_solvers.Solvers.PCG,
            "PYAMG": linear_solvers.Solvers.PYAMG,
            "SPSOLVE": linear_solvers.Solvers.SPSOLVE
        }
        return solver_map[self.solver_combo.currentText()]

    def _run_guarded(self, analysis, **kwargs):
        """Run an analysis and show solver errors (e.g. supports that leave a rigid-body motion free, likely with
        a few point supports) in a message box instead of only in the terminal."""
        try:
            analysis(**kwargs)
        except Exception as e:
            self.parent.message_text.append(f"Analysis failed: {e}")
            QtWidgets.QMessageBox.critical(self, "Analysis failed", str(e))

    def refresh_result_buttons(self, *_):
        design = "optimized" if self.design_combo.currentText() == "Optimized design" else "initial"
        available = result_fields_available(self.parent, design)
        for field, b in self.field_buttons.items():
            b.setEnabled(field in available)
            b.setToolTip("" if field in available else
                         ("Run the matching analysis first (Structural / Thermal Analysis)." if design == "initial"
                          else "Not available for the last optimization (or none run yet)."))

    def show_result(self, field):
        design = "optimized" if self.design_combo.currentText() == "Optimized design" else "initial"
        try:
            msg = show_result_field(self.parent, design, field)
        except Exception as e:
            msg = f"Could not show {field.lower()}: {e}"
        if msg:
            QtWidgets.QMessageBox.information(self, "Results", msg)

    def run_structural_analysis(self, coupled: bool = False):
        """Run structural analysis using existing mesh and boundary conditions.

        coupled=True (Thermo-Structural Analysis button): thermal solve with the thermal loads, then the structural
        solve with the thermal expansion load added, stress-free at the Zero-strain temperature."""
        if not self.parent.hex_mesh:
            QtWidgets.QMessageBox.warning(self, "No Mesh", "Please generate mesh first.")
            return
        
        if not self.parent.applied_material:
            QtWidgets.QMessageBox.warning(self, "No Material", "Please define material properties first.")
            return
        
        # Prepare mesh for structural analysis
        self.prepare_mesh_for_analysis(self.parent.hex_mesh, "structural")
        
        # Get boundary mapping data
        boundary_nodes, boundary_points, tolerance = self.get_boundary_mapping_data()
        
        # Map constrained triangles to fixed nodes - SURFACE ONLY
        fixed_nodes = {'xyz': set(), 'x': set(), 'y': set(), 'z': set()}
        
        if self.parent.constraint_data:          # face supports and point (node) supports
            for constraint in self.parent.constraint_data:
                constraint_type = constraint['type']
                
                # Use surface-only mapping
                constrained_nodes = self.map_bc_entry_to_nodes(
                    constraint, boundary_nodes, boundary_points, tolerance
                )
                
                if constraint_type == 'Fixed XYZ':
                    fixed_nodes['xyz'].update(constrained_nodes)
                elif constraint_type == 'Fixed X':
                    fixed_nodes['x'].update(constrained_nodes)
                elif constraint_type == 'Fixed Y':
                    fixed_nodes['y'].update(constrained_nodes)
                elif constraint_type == 'Fixed Z':
                    fixed_nodes['z'].update(constrained_nodes)
        
        # Map force and torque triangles to load nodes - SURFACE ONLY
        load_nodes_groups = []
        load_forces = []

        for force_info in self.parent.force_data:
            # Use surface-only mapping
            force_nodes = self.map_bc_entry_to_nodes(force_info, boundary_nodes, boundary_points, tolerance)

            if force_info.get('type') == 'torque':
                # Distribute torque, tangential force proportional to radius
                axis_point = np.array(force_info.get('axis_point', [0, 0, 0]))
                direction = np.array(force_info.get('direction', [0, 0, 1]))
                torque_value = force_info.get('torque', 0.0)
                nodes = list(force_nodes)
                if not nodes or np.linalg.norm(direction) < 1e-12:
                    continue
 
                node_xyz = self.parent.hex_mesh.node_xyz[nodes]
                direction = direction / np.linalg.norm(direction)
                face_center = np.mean(node_xyz, axis=0)
                r_vecs = node_xyz - face_center
                r_proj = r_vecs - np.outer(np.dot(r_vecs, direction), direction)
                r_norm = np.linalg.norm(r_proj, axis=1)
                r_norm[r_norm < 1e-12] = 1e-12
                tangent_dirs = np.cross(direction, r_proj)
                tangent_dirs = tangent_dirs / np.linalg.norm(tangent_dirs, axis=1)[:, None]
                force_vecs = tangent_dirs * r_norm[:, None]
                torque_actual = np.sum(np.cross(r_proj, force_vecs), axis=0)
                scale = torque_value / (np.dot(torque_actual, direction) + 1e-12)
                force_vecs = force_vecs * scale
                for node, fvec in zip(nodes, force_vecs):
                    load_nodes_groups.append([node])
                    load_forces.append(fvec.tolist())
            else:
                # Standard force
                if force_nodes:
                    load_nodes_groups.append(list(force_nodes))
                    load_forces.append([
                        force_info.get('force_x', 0.0),
                        force_info.get('force_y', 0.0),
                        force_info.get('force_z', 0.0)
                    ])

        # Body force: applied once, as element forces (gui_elem_body_force, rho * a * element volume) when the
        # solver is built below. The nodal version that used to be here assumed a total volume of 1 m^3 (the mesh
        # has no elemVolume/node_volume) and, together with the element force, applied gravity twice.

        # Prepare load data for solver
        load_data = {
            'load_nodes_groups': load_nodes_groups,
            'load_forces': load_forces
        }
        
        # Process data for solver
        mesh, mat_prop, bc = self.process_data_for_solver(
            self.parent.hex_mesh,
            fixed_nodes,
            load_data,
            self.parent.applied_material['properties']
        )

        # Before calling fe_solver.solve()
        if self.get_solver() == linear_solvers.Solvers.DPCG:
            self.parent.dsolver = deflation.DeflationSolver()
            nGroups = min(self.parent.dsolver.maxGroups, max(self.parent.dsolver.minGroups, round(3*self.parent.hex_mesh.num_nodes/self.parent.dsolver.dofPerGroup)))
            self.parent.dsolver.create_deflation_groups(self.parent.hex_mesh, nGroups)
            self.parent.dsolver.create_deflation_matrix(self.parent.hex_mesh)
            #self.parent.dsolver.W = self.parent.dsolver.W[bc.free_dofs, :]
        else:
            self.parent.dsolver = None
        
        # Create FEA solver: structural, or coupled thermo-structural (Thermo-Structural Analysis button)
        thermal_loads = getattr(getattr(self.parent, "thermal_loads_window", None), "thermal_loads", None) or {}
        has_thermal = any(thermal_loads.get(k) for k in ("fixed_temps", "heat_sources", "total_heat_sources"))
        if coupled and not has_thermal:
            QtWidgets.QMessageBox.warning(self, "No Thermal Loads",
                                          "Thermo-structural analysis needs thermal loads. Apply them first.")
            return
        if coupled:
            full_mat = self.create_material_properties()          # includes conductivity and expansion
            thermal_bc = self.process_thermal_boundary_conditions(
                self.parent.hex_mesh, thermal_loads, boundary_nodes, boundary_points, tolerance)
            fe_solver = ThermoStructuralFEA(
                mesh=mesh, mat_prop=full_mat, structural_bc=bc, thermal_bc=thermal_bc, solver=self.get_solver(),
                dsolver=getattr(self.parent, 'dsolver', None), rtol=1e-8,
                elem_body_force=gui_elem_body_force(self.parent, mesh, full_mat),
                thermoElasticReferenceTemperature=self.temp_spin.value())
            self.parent.message_text.append(
                f"Starting thermo-structural analysis (zero-strain temperature {self.temp_spin.value():g})...")
        else:
            fe_solver = hex_structural_fea.HexStructuralFEA(
                mesh=mesh,
                mat_prop=mat_prop,
                bc=bc,
                solver=self.get_solver(),
                dsolver=getattr(self.parent, 'dsolver', None),
                rtol=1e-8,
                elem_body_force=gui_elem_body_force(self.parent, mesh, mat_prop),
            )
            self.parent.message_text.append("Starting structural analysis...")

        # Solve
        solution = fe_solver.solve()
        fe_solver.postprocess()
        if isinstance(fe_solver, ThermoStructuralFEA):   # its temperature field is the initial-design temperature
            self.parent.thermal_fe_solver = fe_solver.thermal_fea

        # Store solver for visualization
        self.parent.fe_solver = fe_solver
        
        if 'colored_mesh' in self.parent.plotter.actors:
            self.parent.plotter.remove_actor('colored_mesh', reset_camera=False)

        if hasattr(self.parent.plotter, 'axes_actor'):
            self.parent.plotter.remove_actor('axes_actor', reset_camera=False)

        scale_info = fe_solver.plot_deformation(plotter=self.parent.plotter)
        
        if scale_info:
            self.parent.message_text.append(scale_info)
        
        # Store results
        self.parent.analysis_results = {
            'displacement': solution,
            'stress': fe_solver.vonMisesStress,
            'max_deformation': fe_solver.max_deformation,
            'max_stress': np.max(fe_solver.vonMisesStress)
        }
        
        # Update UI
        self.parent.update_LivVar('analysis.structural', True)
        self.parent.update_LivVar('analysis.performed', True)
        
        self.parent.message_text.append(f"Analysis complete. Max deformation: {fe_solver.max_deformation:.4e}")
        self.parent.message_text.append(f"Max von Mises stress: {np.max(fe_solver.vonMisesStress):.4e}")
        self.refresh_result_buttons()

    def run_thermal_analysis(self):
        """Run thermal analysis using existing mesh and thermal boundary conditions"""
        # Validate prerequisites
        if not self.parent.hex_mesh:
            QtWidgets.QMessageBox.warning(self, "No Mesh", "Please generate mesh first.")
            return
        
        if not self.parent.applied_material:
            QtWidgets.QMessageBox.warning(self, "No Material", "Please define material properties first.")
            return
        
        # Check thermal loads
        if not self.parent.thermal_loads_window or not self.parent.thermal_loads_window.thermal_loads:
            QtWidgets.QMessageBox.warning(self, "No Thermal Loads", "Please apply thermal loads first.")
            return
        
        thermal_loads = self.parent.thermal_loads_window.thermal_loads
        has_loads = any(thermal_loads.get(key, []) for key in ['fixed_temps', 'heat_sources', 'total_heat_sources'])
        if not has_loads:
            QtWidgets.QMessageBox.warning(self, "No Thermal Loads", "Please apply thermal loads first.")
            return
        
        # Prepare mesh for thermal FEA
        self.prepare_mesh_for_analysis(self.parent.hex_mesh, "thermal")
        
        # Get boundary mapping data
        boundary_nodes, boundary_points, tolerance = self.get_boundary_mapping_data()
        
        # Process thermal boundary conditions using surface-only mapping
        thermal_bc = self.process_thermal_boundary_conditions(
            self.parent.hex_mesh, thermal_loads, boundary_nodes, boundary_points, tolerance
        )
        
        # Create material properties and solver
        mat_prop = self.create_material_properties()
        solver = self.get_solver()
        
        # Create and solve thermal FEA
        fe_solver = HexThermalFEA(mesh=self.parent.hex_mesh, mat_prop=mat_prop, bc=thermal_bc, solver=solver)
        
        self.parent.message_text.append("Starting thermal analysis...")
        temperature_solution = fe_solver.solve()
        fe_solver.postprocess()
        
        # Store solver and results
        self.parent.thermal_fe_solver = fe_solver
        self.parent.thermal_results = {
            'temperature': temperature_solution,
            'heat_flux': fe_solver.strain if fe_solver.strain is not None else None,
            'max_temperature': np.max(temperature_solution),
            'min_temperature': np.min(temperature_solution)
        }
        
        # Clear colored mesh and use thermal FEA's built-in visualization
        if 'colored_mesh' in self.parent.plotter.actors:
            self.parent.plotter.remove_actor('colored_mesh', reset_camera=False)

        if 'stl_geometry' in self.parent.plotter.actors:
            self.parent.plotter.remove_actor('stl_geometry', reset_camera=False)
        
        # Use thermal FEA's built-in plot_temperature method
        temp_info = fe_solver.plot_temperature(plotter=self.parent.plotter)
        if temp_info:
            self.parent.message_text.append(temp_info)
        
        # Update UI state
        self.parent.update_LivVar('analysis.thermal', True)
        self.parent.update_LivVar('analysis.performed', True)
        
        # Success messages
        self.parent.message_text.append("Thermal analysis complete.")
        self.parent.message_text.append(f"Temperature range: {np.min(temperature_solution):.2f} - {np.max(temperature_solution):.2f} K")
        self.refresh_result_buttons()

    def process_thermal_boundary_conditions(self, mesh, thermal_loads, boundary_nodes, boundary_points, tolerance):
        """Process thermal loads and create boundary conditions using surface-only mapping"""
        thermal_force = np.zeros(mesh.num_nodes)
        fixed_temp_nodes = []
        fixed_temp_values = []
        
        # Process fixed temperature BCs - SURFACE ONLY
        for temp_load in thermal_loads.get('fixed_temps', []):
            affected_nodes = self.map_triangles_to_thermal_nodes(
                temp_load['triangles'], boundary_nodes, boundary_points, tolerance
            )
            fixed_temp_nodes.extend(affected_nodes)
            fixed_temp_values.extend([temp_load['temperature']] * len(affected_nodes))
        
        # Process heat flux BCs - SURFACE ONLY. The flux is entered per display area (W/unit^2); the total heat is
        # flux x (area of the picked STL triangles), shared equally by the face's nodes like Total Heat.
        settings = getattr(self.parent, "settings", None)
        units_per_m = settings.unit_conversions["length"][settings.unit_system] if settings else 1.0
        tri_areas = np.asarray(self.parent.stl_geom.tri_areas)
        for heat_load in thermal_loads.get('heat_sources', []):
            affected_nodes = self.map_triangles_to_thermal_nodes(
                heat_load['triangles'], boundary_nodes, boundary_points, tolerance
            )
            if affected_nodes:
                total_heat = heat_flux_total(heat_load['heat_flux'], tri_areas[np.asarray(heat_load['triangles'])],
                                             units_per_m)
                heat_per_node = total_heat / len(affected_nodes)
                for node_id in affected_nodes:
                    thermal_force[node_id] += heat_per_node
        
        # Process total heat BCs - SURFACE ONLY
        for total_heat_load in thermal_loads.get('total_heat_sources', []):
            affected_nodes = self.map_triangles_to_thermal_nodes(
                total_heat_load['triangles'], boundary_nodes, boundary_points, tolerance
            )
            if affected_nodes:
                heat_per_node = total_heat_load['total_heat'] / len(affected_nodes)
                for node_id in affected_nodes:
                    thermal_force[node_id] += heat_per_node
        
        # Create boundary conditions
        fixed_temp_nodes = np.array(fixed_temp_nodes, dtype=int) if fixed_temp_nodes else np.array([], dtype=int)
        fixed_temp_values = np.array(fixed_temp_values, dtype=float) if fixed_temp_values else np.array([], dtype=float)
        
        return bound_cond.BC(
            force=thermal_force,
            fixed_dofs=fixed_temp_nodes,
            dirichlet_values=fixed_temp_values
        )

    @staticmethod
    def process_data_for_solver(existing_mesh, fixed_nodes, load_data, material_props):
        """Process mesh and boundary condition data for structural solver"""
        mesh = existing_mesh
        
        # Process fixed nodes
        fixed_dofs = []
        for node in fixed_nodes['xyz']:
            fixed_dofs.extend([3*node, 3*node + 1, 3*node + 2])
            
        for node in fixed_nodes['x']:
            fixed_dofs.append(3*node)
            
        for node in fixed_nodes['y']:
            fixed_dofs.append(3*node + 1)
            
        for node in fixed_nodes['z']:
            fixed_dofs.append(3*node + 2)
            
        fixed_dofs = np.array(fixed_dofs).astype(int)
        dirichlet_values = np.zeros_like(fixed_dofs, dtype=float)
        
        # Process loads
        force = np.zeros(3*mesh.num_nodes)
        for nodes, force_vector in zip(load_data['load_nodes_groups'], load_data['load_forces']):
            if nodes:
                force_per_node = np.array(force_vector) / len(nodes)
                for node in nodes:
                    force[3*node:3*node + 3] += force_per_node

        # Create boundary conditions and material properties
        bc = bound_cond.BC(force=force,
                        fixed_dofs=fixed_dofs,
                        dirichlet_values=dirichlet_values)
        
        mat_prop = mat_lib.Material(
            name="Applied_Material",
            youngs_modulus=material_props['Young\'s Modulus'],
            poissons_ratio=material_props['Poisson\'s Ratio'],
            mass_density=material_props['Density'],
            thermal_conductivity=material_props['Thermal Conductivity'],
            specific_heat=material_props['Specific Heat Capacity'],
            thermal_expansion_coefficient=material_props['Thermal Expansion'],
            cost=material_props['Price'],
            yield_strength=material_props['Yield Strength']
        )
        
        return mesh, mat_prop, bc
    
class TopOptOptionsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("TopOpt Options")
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(300, 600)
        self.parent = parent
        self.applied = False
        
        # Initialize constraint actors
        if not hasattr(self.parent, 'topopt_option_actors'):
            self.parent.topopt_option_actors = {}
        
        # Define constraint configuration
        self.constraint_config = {
            'manufacturing': [
                ('extrude', 'Extrude', 'combo', ["XDir", "YDir", "ZDir"]),
                ('am_build', 'AM Build', 'combo', ["+XDir", "+YDir", "+ZDir", "-XDir", "-YDir", "-ZDir"]),
                ('draw_direction', 'Draw Direction', 'combo', ["XDir", "YDir", "ZDir"]),
                ('cyclic_symmetry', 'CyclicSym(Z)', 'combo', ["(2) 180 deg", "(3) 120 deg", "(4) 90 deg", "(5) 72 deg", "(6) 60 deg", "(7) 51 deg","(8) 45 deg"])
            ],
            # Performance limits (stress, displacement, temperature) are constraints in TopOpt Execute now; the fields
            # that used to be here were collected but never applied.
            'symmetry': [
                ('x_symmetry', 'X-Symmetry', 'check_only'),
                ('y_symmetry', 'Y-Symmetry', 'check_only'),
                ('z_symmetry', 'Z-Symmetry', 'check_only')
            ],
            'other': [
                ('keep_fixed_faces', 'Keep Fixed Faces', 'check_only')
            ]
        }
        
        self.widgets = {}
        self.setup_ui()
        
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create scroll area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        
        container = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout(container)
        
        # Create widgets from configuration
        for category, constraints in self.constraint_config.items():
            for constraint in constraints:
                self.create_constraint_widget(constraint, form_layout)
        
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        # Apply button
        apply_button = QtWidgets.QPushButton("Apply")
        apply_button.clicked.connect(self.apply_constraints)
        layout.addWidget(apply_button)
        
        self.connect_signals()

        # Restore checkbox states from parent.topopt_options if available
        if hasattr(self.parent, 'topopt_options') and self.parent.topopt_options:
            for category, constraint_data in self.parent.topopt_options.items():
                for constraint_key, constraint_value in constraint_data.items():
                    if isinstance(constraint_value, bool):
                        widget_name = f'{constraint_key}_check'
                        if widget_name in self.widgets:
                            self.widgets[widget_name].setChecked(constraint_value)
                    elif isinstance(constraint_value, dict):
                        check_widget = f'{constraint_key}_check'
                        if check_widget in self.widgets:
                            self.widgets[check_widget].setChecked(constraint_value.get('enabled', False))
                        if 'value' in constraint_value:
                            for widget_type in ['combo', 'spin']:
                                value_widget = f'{constraint_key}_{widget_type}'
                                if value_widget in self.widgets:
                                    widget = self.widgets[value_widget]
                                    if widget_type == 'combo':
                                        widget.setCurrentText(str(constraint_value['value']))
                                    else:
                                        widget.setValue(constraint_value['value'])
                                    break
    
    def create_constraint_widget(self, constraint_def, layout):
        """Create widget based on constraint definition"""
        key = constraint_def[0]
        label = constraint_def[1]
        widget_type = constraint_def[2]
        
        # Create checkbox
        checkbox = QtWidgets.QCheckBox(label)
        self.widgets[f'{key}_check'] = checkbox
        
        if widget_type == 'combo':
            # Combo box widget
            combo = QtWidgets.QComboBox()
            combo.addItems(constraint_def[3])
            self.widgets[f'{key}_combo'] = combo
            layout.addRow(checkbox, combo)
            
        elif widget_type == 'spin':
            # Spin box widget
            spin = QtWidgets.QSpinBox()
            min_val, max_val, default = constraint_def[3]
            spin.setRange(min_val, max_val)
            spin.setValue(default)
            self.widgets[f'{key}_spin'] = spin
            layout.addRow(checkbox, spin)
            
        elif widget_type == 'double_spin':
            # Double spin box widget
            spin = QtWidgets.QDoubleSpinBox()
            params = constraint_def[3]
            spin.setRange(params[0], params[1])
            spin.setValue(params[2])
            if len(params) > 3:  # decimals specified
                spin.setDecimals(params[3])
            self.widgets[f'{key}_spin'] = spin
            layout.addRow(checkbox, spin)
            
        elif widget_type == 'check_only':
            # Checkbox only
            if len(constraint_def) > 3 and constraint_def[3]:  # default checked
                checkbox.setChecked(True)
            layout.addRow(checkbox)
    
    def connect_signals(self):
        """Connect all widget signals to update methods"""
        for widget_name, widget in self.widgets.items():
            if 'check' in widget_name:
                widget.stateChanged.connect(self.update_visualizations)
            elif 'combo' in widget_name or 'spin' in widget_name:
                if hasattr(widget, 'currentTextChanged'):
                    widget.currentTextChanged.connect(self.update_visualizations)
                elif hasattr(widget, 'valueChanged'):
                    widget.valueChanged.connect(self.update_visualizations)
    
    def update_visualizations(self):
        """Update all constraint visualizations based on current widget states"""
        # Clear all existing visualizations
        self.clear_all_visualizations()
        
        if not self.parent.stl_geom:
            return
            
        bbox = self.parent.stl_geom.get_bounding_box()
        center = [(bbox[0] + bbox[1])/2, (bbox[2] + bbox[3])/2, (bbox[4] + bbox[5])/2]
        
        # Update each constraint type
        self.update_arrow_constraints(bbox, center)
        self.update_grid_patterns(bbox)
        self.update_symmetry_planes(bbox, center)
        
        self.parent.plotter.render()
    
    def update_arrow_constraints(self, bbox, center):
        """Update arrow-based constraints (extrude, draw_direction)"""
        arrow_configs = {
            'extrude': {'colors': {'XDir': 'red', 'YDir': 'green', 'ZDir': 'blue'}, 'bidirectional': False},
            'draw_direction': {'colors': {'XDir': 'green', 'YDir': 'green', 'ZDir': 'green'}, 'bidirectional': True}
        }
        
        for constraint_key, config in arrow_configs.items():
            if self.get_widget_value(f'{constraint_key}_check'):
                direction_key = self.get_widget_value(f'{constraint_key}_combo')
                self.create_directional_arrows(constraint_key, direction_key, bbox, center, config)
    
    def create_directional_arrows(self, constraint_key, direction_key, bbox, center, config):
        """Create arrows for directional constraints"""
        directions = {"XDir": [1, 0, 0], "YDir": [0, 1, 0], "ZDir": [0, 0, 1]}
        sizes = {"XDir": bbox[1] - bbox[0], "YDir": bbox[3] - bbox[2], "ZDir": bbox[5] - bbox[4]}
        
        direction = directions[direction_key]
        color = config['colors'][direction_key]
        size = sizes[direction_key]
        
        actors = []
        
        if config['bidirectional']:
            # Create two arrows from center
            half_size = size / 2
            actors.append(self.create_arrow_actor(center, direction, color, half_size))
            actors.append(self.create_arrow_actor(center, [-d for d in direction], color, half_size))
        else:
            # Create single arrow from boundary
            start_pos = self.get_boundary_start_position(direction_key, bbox, center)
            actors.append(self.create_arrow_actor(start_pos, direction, color, size))
        
        self.parent.topopt_option_actors[constraint_key] = actors if len(actors) > 1 else actors[0]
    
    def get_boundary_start_position(self, direction_key, bbox, center):
        """Get starting position at model boundary for arrows"""
        positions = {
            "XDir": [bbox[0], center[1], center[2]],
            "YDir": [center[0], bbox[2], center[2]],
            "ZDir": [center[0], center[1], bbox[4]]
        }
        return positions[direction_key]
    
    def update_grid_patterns(self, bbox):
        """Update grid pattern visualizations"""
        any_grid_enabled = any(self.get_widget_value(f'{axis}_grid_check') for axis in ['x', 'y', 'z'])
        
        if any_grid_enabled:
            # Create bounding box
            box = pv.Box(bounds=bbox)
            actor = self.parent.plotter.add_mesh(box, style='wireframe', color='gray', line_width=2)
            self.parent.topopt_option_actors['bounding_box'] = actor
            
            # Create grid planes for each enabled axis
            grid_config = {
                'x': {'normal': [1, 0, 0], 'color': 'red', 'bbox_indices': (0, 1)},
                'y': {'normal': [0, 1, 0], 'color': 'green', 'bbox_indices': (2, 3)},
                'z': {'normal': [0, 0, 1], 'color': 'blue', 'bbox_indices': (4, 5)}
            }
            
            for axis, config in grid_config.items():
                if self.get_widget_value(f'{axis}_grid_check'):
                    divisions = self.get_widget_value(f'{axis}_grid_spin')
                    actors = self.create_grid_planes_for_axis(axis, divisions, bbox, config)
                    self.parent.topopt_option_actors[f'{axis}_grid'] = actors
    
    def create_grid_planes_for_axis(self, axis, divisions, bbox, config):
        """Create grid planes for a specific axis"""
        actors = []
        min_pos, max_pos = config['bbox_indices']
        
        for i in range(divisions):
            if divisions == 1:
                pos = (bbox[min_pos] + bbox[max_pos]) / 2
            else:
                pos = bbox[min_pos] + (bbox[max_pos] - bbox[min_pos]) * i / (divisions - 1)
            
            center = self.get_plane_center(axis, pos, bbox)
            size = self.get_plane_size(axis, bbox)
            actor = self.create_plane_actor(center, config['normal'], size, config['color'])
            actors.append(actor)
        
        return actors
    
    def get_plane_center(self, axis, position, bbox):
        """Get center position for plane based on axis"""
        centers = {
            'x': [position, (bbox[2]+bbox[3])/2, (bbox[4]+bbox[5])/2],
            'y': [(bbox[0]+bbox[1])/2, position, (bbox[4]+bbox[5])/2],
            'z': [(bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2, position]
        }
        return centers[axis]
    
    def get_plane_size(self, axis, bbox):
        """Get appropriate size for plane based on axis"""
        sizes = {
            'x': max(bbox[3] - bbox[2], bbox[5] - bbox[4]) * 1.1,
            'y': max(bbox[1] - bbox[0], bbox[5] - bbox[4]) * 1.1,
            'z': max(bbox[1] - bbox[0], bbox[3] - bbox[2]) * 1.1
        }
        return sizes[axis]
    
    def update_symmetry_planes(self, bbox, center):
        """Update symmetry plane visualizations"""
        symmetry_config = {
            'x_symmetry': {'normal': [1, 0, 0], 'color': 'red'},
            'y_symmetry': {'normal': [0, 1, 0], 'color': 'green'},
            'z_symmetry': {'normal': [0, 0, 1], 'color': 'blue'}
        }
        
        size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4]) * 1.2
        
        for symmetry_key, config in symmetry_config.items():
            if self.get_widget_value(f'{symmetry_key}_check'):
                actor = self.create_plane_actor(center, config['normal'], size, config['color'])
                self.parent.topopt_option_actors[symmetry_key] = actor
        
        # Handle cyclic symmetry
        if self.get_widget_value('cyclic_symmetry_check'):
            angle_text = self.get_widget_value('cyclic_symmetry_combo')
            
            # Parse the number of planes from the text
            # Format: "(2) 180 deg", "(3) 120 deg", etc.
            num_planes = int(angle_text.split('(')[1].split(')')[0])
            
            actors = []
            
            # Calculate adaptive dimensions
            x_size = bbox[1] - bbox[0]
            y_size = bbox[3] - bbox[2] 
            z_size = bbox[5] - bbox[4]
            max_radius = max(x_size, y_size) / 2
            
            # Create the specified number of planes starting from center
            for i in range(num_planes):
                # Calculate angle for each plane (evenly distributed around Z-axis)
                angle = (2 * np.pi * i) / num_planes
                
                # Create normal vector in XY plane (rotating around Z-axis)
                normal = [np.cos(angle), np.sin(angle), 0]
                
                # Create adaptive plane that extends from center to boundary
                actor = self.create_cyclic_plane(center, normal, bbox, 'cyan')
                actors.append(actor)
            
            self.parent.topopt_option_actors['cyclic_symmetry'] = actors

    def create_cyclic_plane(self, center, normal, bbox, color):
        """Create plane for cyclic symmetry"""
        # Calculate bounding box dimensions
        x_size = bbox[1] - bbox[0]
        y_size = bbox[3] - bbox[2]
        z_size = bbox[5] - bbox[4]
        
        max_radius = max(x_size, y_size) / 2 * 1.5 
        
        # Create a vertical plane that extends radially from center
        plane_points = vtk.vtkPoints()
        
        # Rotate the normal 180 degrees (opposite direction)
        rotated_normal = [-normal[0], -normal[1], 0]
        
        # Calculate plane height (vertical extent in Z direction)
        plane_height = z_size * 1.2  # Extend beyond model bounds
        
        # Define the four corners of the vertical radial plane
        # Bottom center point
        plane_points.InsertNextPoint(center[0], center[1], center[2] - plane_height/2)
        
        # Top center point  
        plane_points.InsertNextPoint(center[0], center[1], center[2] + plane_height/2)
        
        # Bottom outer point (along rotated normal direction)
        outer_x = center[0] + max_radius * rotated_normal[0]
        outer_y = center[1] + max_radius * rotated_normal[1]
        plane_points.InsertNextPoint(outer_x, outer_y, center[2] - plane_height/2)
        
        # Top outer point
        plane_points.InsertNextPoint(outer_x, outer_y, center[2] + plane_height/2)
        
        # Create the quad (rectangle) - vertical plane from center outward
        quad = vtk.vtkQuad()
        quad.GetPointIds().SetId(0, 0)  # Bottom center
        quad.GetPointIds().SetId(1, 2)  # Bottom outer
        quad.GetPointIds().SetId(2, 3)  # Top outer  
        quad.GetPointIds().SetId(3, 1)  # Top center
        
        # Create cell array and add the quad
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(quad)
        
        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(plane_points)
        polydata.SetPolys(cells)
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set appearance - make more visible
        color_rgb = self._get_color_rgb(color)
        actor.GetProperty().SetColor(*color_rgb)
        actor.GetProperty().SetOpacity(0.6)  # Increased opacity for better visibility
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0, 0, 0)
        actor.GetProperty().SetLineWidth(3)  # Thicker edge lines
        
        # Enable two-sided rendering so plane is visible from both sides
        actor.GetProperty().SetBackfaceCulling(False)
        actor.GetProperty().SetFrontfaceCulling(False)
        
        # Add lighting for better visibility
        actor.GetProperty().SetInterpolationToGouraud()
        
        self.parent.plotter.add_actor(actor)
        return actor
    
    def get_widget_value(self, widget_name):
        """Get value from any widget type"""
        widget = self.widgets.get(widget_name)
        if not widget:
            return None
            
        if hasattr(widget, 'isChecked'):
            return widget.isChecked()
        elif hasattr(widget, 'currentText'):
            return widget.currentText()
        elif hasattr(widget, 'value'):
            return widget.value()
        return None
    
    def clear_all_visualizations(self):
        """Remove all constraint visualizations"""
        for actor_key, actor in list(self.parent.topopt_option_actors.items()):
            if actor:
                if isinstance(actor, list):
                    for a in actor:
                        self.parent.plotter.remove_actor(a, reset_camera=False)
                else:
                    self.parent.plotter.remove_actor(actor, reset_camera=False)
        
        self.parent.topopt_option_actors.clear()
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        if not self.applied:
            self.clear_all_visualizations()
            self.parent.plotter.render()
        event.accept()
    
    def create_arrow_actor(self, position, direction, color, scale):
        """Create an arrow actor"""
        if isinstance(scale, (list, tuple, np.ndarray)):
            scale = float(scale[0]) if len(scale) > 0 else 1.0
        else:
            scale = float(scale)

        bbox = self.parent.stl_geom.get_bounding_box()
        # model_size = max(bbox[1]-bbox[0], bbox[3]-bbox[2], bbox[5]-bbox[4])
        
        # shaft_radius = model_size * 0.002 
        # tip_radius = model_size * 0.004     
        # tip_length = model_size * 0.02

        #bbox = self.parent.stl_geom.get_bounding_box()
        # Calculate the diagonal length of the bounding box
        dx = bbox[1] - bbox[0]
        dy = bbox[3] - bbox[2]
        dz = bbox[5] - bbox[4]
        diag_length = (dx**2 + dy**2 + dz**2) ** 0.1

        # Use the diagonal length for scaling
        shaft_radius = diag_length * 0.01   # Increase as needed
        tip_radius = diag_length * 0.02     # Increase as needed
        tip_length = diag_length * 0.08     # Increase as needed

        arrow = pv.Arrow(
            start=position, 
            direction=direction, 
            scale=scale * 1.1,
            shaft_radius=shaft_radius,
            tip_radius=tip_radius,
            tip_length=tip_length
        )
        
        return self.parent.plotter.add_mesh(arrow, color=color)
        
    def create_plane_actor(self, center, normal, size, color):
        """Create a plane actor"""
        bbox = self.parent.stl_geom.get_bounding_box()
        x_size = (bbox[1] - bbox[0]) * 1.08
        y_size = (bbox[3] - bbox[2]) * 1.08
        z_size = (bbox[5] - bbox[4]) * 1.08
        
        if abs(normal[0]) > 0.5:
            i_size, j_size = y_size, z_size
        elif abs(normal[1]) > 0.5:
            i_size, j_size = x_size, z_size
        else:
            i_size, j_size = x_size, y_size
        
        plane = vtk.vtkPlaneSource()
        plane.SetCenter(center)
        plane.SetNormal(normal)
        
        # Calculate perpendicular vectors
        v1 = [1, 0, 0] if abs(normal[0]) < 0.9 else [0, 1, 0]
        t1 = [normal[1] * v1[2] - normal[2] * v1[1],
              normal[2] * v1[0] - normal[0] * v1[2],
              normal[0] * v1[1] - normal[1] * v1[0]]
        
        mag = (t1[0]**2 + t1[1]**2 + t1[2]**2)**0.5
        t1 = [t1[0]/mag, t1[1]/mag, t1[2]/mag]
        
        t2 = [normal[1] * t1[2] - normal[2] * t1[1],
              normal[2] * t1[0] - normal[0] * t1[2],
              normal[0] * t1[1] - normal[1] * t1[0]]
        
        t1, t2 = t2, [-t1[0], -t1[1], -t1[2]]
        
        half_i, half_j = i_size / 2, j_size / 2
        
        origin = [center[0] - half_i * t1[0] - half_j * t2[0],
                  center[1] - half_i * t1[1] - half_j * t2[1],
                  center[2] - half_i * t1[2] - half_j * t2[2]]
        
        point1 = [center[0] + half_i * t1[0] - half_j * t2[0],
                  center[1] + half_i * t1[1] - half_j * t2[1],
                  center[2] + half_i * t1[2] - half_j * t2[2]]
        
        point2 = [center[0] - half_i * t1[0] + half_j * t2[0],
                  center[1] - half_i * t1[1] + half_j * t2[1],
                  center[2] - half_i * t1[2] + half_j * t2[2]]
        
        plane.SetOrigin(origin)
        plane.SetPoint1(point1)
        plane.SetPoint2(point2)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())
        
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        color_rgb = self._get_color_rgb(color)
        actor.GetProperty().SetColor(*color_rgb)
        actor.GetProperty().SetOpacity(0.3)
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0, 0, 0)
        actor.GetProperty().SetLineWidth(2)
        
        self.parent.plotter.add_actor(actor)
        return actor

    def _get_color_rgb(self, color_name):
        """Convert color name to RGB tuple"""
        color_map = {
            'red': (1.0, 0.0, 0.0), 'green': (0.0, 1.0, 0.0), 'blue': (0.0, 0.0, 1.0),
            'cyan': (0.0, 1.0, 1.0), 'magenta': (1.0, 0.0, 1.0), 'yellow': (1.0, 1.0, 0.0),
            'white': (1.0, 1.0, 1.0), 'black': (0.0, 0.0, 0.0)
        }
        return color_map.get(color_name, (0.5, 0.5, 0.5))
        
    def apply_constraints(self):
        """Apply all constraint settings"""
        constraints = {}
        
        # Build constraints dictionary from configuration
        for category, constraint_list in self.constraint_config.items():
            constraints[category] = {}
            for constraint_def in constraint_list:
                key = constraint_def[0]
                widget_type = constraint_def[2]
                
                if widget_type == 'check_only':
                    constraints[category][key] = self.get_widget_value(f'{key}_check')
                else:
                    constraints[category][key] = {
                        'enabled': self.get_widget_value(f'{key}_check'),
                        'value': self.get_widget_value(f'{key}_combo') or self.get_widget_value(f'{key}_spin')
                    }
        
        # Store constraints in parent
        self.parent.topopt_options = constraints
        self.applied = True
        
        # FIX: Update UI state 
        self.parent.update_LivVar('topopt.options_defined', True)
        self.parent.set_sidebar_icon("TopOpt Options", "check")
        self.parent.set_sidebar_icon("TopOpt Execute", "arrow")
        self.parent.message_text.append(f"TopOpt options applied successfully. Options stored: {len(constraints)} categories")
        
        # ADD DEBUG: Print the LivVar state to verify
        #print(f"DEBUG: LivVar topopt state: {self.parent.LivVar.get('topopt', {})}")

        # Notify display options window
        self.parent.notify_display_options_update()
        self.close()
#----------------------------------------------------------------------------
class _SignalLineStream:
    """File-like object that sends each complete printed line to emit (a Qt signal, so it is thread safe)."""

    def __init__(self, emit):
        self.emit, self.buffer = emit, ""

    def write(self, text):
        self.buffer += text
        *lines, self.buffer = self.buffer.split("\n")
        for line in lines:
            self.emit(line)
        return len(text)

    def flush(self):
        pass


class StructuralTopOptWindow(QtWidgets.QDialog):

    optimization_progress = pyqtSignal(str)
    optimization_update = pyqtSignal(object, int)
    optimization_done = pyqtSignal(bool, str, object, object, object)
    history_update = pyqtSignal(object)
    log_line = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("TopOpt Execute")
        self.setWindowModality(QtCore.Qt.NonModal)
        self.setBaseSize(300, 400)
        self.parent = parent
        
        self.optimization_running = False
        self.optimization_thread = None

        self.optimization_progress.connect(self.parent.message_text.append, QtCore.Qt.QueuedConnection)
        self.optimization_update.connect(
            self.update_visualization,
            QtCore.Qt.QueuedConnection  # Force main thread execution!
        )
        self.optimization_done.connect(self.optimization_completed)
        self.history_update.connect(self.update_history_plot, QtCore.Qt.QueuedConnection)
        
        self.setup_ui()
        self.log_line.connect(self.formulation.log_view.appendPlainText, QtCore.Qt.QueuedConnection)

        if hasattr(self.parent, "fe_solver") and self.parent.fe_solver is not None:
            # Clear all actors except geometry info
            for name in list(self.parent.plotter.actors.keys()):
                if name != 'geometry_info':
                    self.parent.plotter.remove_actor(name, reset_camera=False)
            # Show mesh
            self.parent.fe_solver.plot_mesh(plotter=self.parent.plotter)
            self.parent.plotter.render()
    
    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Formulation: objective, constraints, method (gui/optimization_setup.py)
        self.formulation = FormulationPanel(self)
        spec = getattr(self.parent, "optimization_spec", None)
        if spec is None:
            spec = default_spec()
            spec.constraints += migrate_performance_options(getattr(self.parent, "topopt_options", None))
        self.formulation.set_spec(spec)
        self.formulation.set_context(self.problem_physics(), {n: {"nodes": [0], "elements": [0]}
                                                              for n in self.selection_names()},
                                     body_force=self.has_body_force())
        self.formulation.check_gradient_button.clicked.connect(self.check_gradient)
        layout.addWidget(self.formulation)

        self.validate_button = QtWidgets.QPushButton("Validate formulation")
        self.validate_button.clicked.connect(self.validate_formulation)
        layout.addWidget(self.validate_button)

        # Live objective / constraint plot (MMA reports every iteration)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self.history_figure = Figure(figsize=(4, 2.4), tight_layout=True)
        self.history_canvas = FigureCanvasQTAgg(self.history_figure)
        self.history_canvas.setMinimumHeight(200)       # otherwise the layout squeezes it to a thin strip
        layout.addWidget(self.history_canvas)

        # Draw the design every iteration (off by default: redrawing costs time on large meshes). Read by the worker
        # thread through a plain bool, so it can be toggled during a run.
        self.show_each_iteration = False
        self.live_topology_check = QtWidgets.QCheckBox("Show topology every iteration")
        self.live_topology_check.setChecked(False)
        self.live_topology_check.setToolTip("Redraw the density in the 3D view after each iteration "
                                            "(MMA).")
        self.live_topology_check.toggled.connect(lambda on: setattr(self, "show_each_iteration", bool(on)))
        layout.addWidget(self.live_topology_check)

        # Fields of the optimized design (enabled once a run has finished)
        self.field_buttons = add_topopt_field_buttons(self, layout)
        refresh_topopt_field_buttons(self.parent, self.field_buttons)

        # Optimize button
        self.optimize_button = QtWidgets.QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.start_optimization)
        layout.addWidget(self.optimize_button)
        
        # Stop button
        self.stop_button = QtWidgets.QPushButton("STOP OPTIMIZATION!")
        self.stop_button.clicked.connect(self.stop_optimization)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        # Close button
        close_button = QtWidgets.QPushButton("Close")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)

    # ------------------------------------------------------------------ problem context
    def problem_physics(self):
        """structural / thermal / thermo-structural from the loads the user applied."""
        tw = getattr(self.parent, "thermal_loads_window", None)
        thermal_loads = getattr(tw, "thermal_loads", None) or {}
        has_thermal = any(thermal_loads.get(k) for k in ("fixed_temps", "heat_sources", "total_heat_sources"))
        # supports in any form (faces or nodes); with thermal loads, thermal expansion alone is a load
        has_struct = bool(getattr(self.parent, "constraint_data", None)) and bool(
            getattr(self.parent, "force_data", None) or self.has_body_force() or has_thermal)
        if has_thermal and has_struct:
            return "thermo-structural"
        return "thermal" if has_thermal else "structural"

    def has_body_force(self):
        bf = getattr(self.parent, "body_force", None) or {}
        return any(abs(float(bf.get(k, 0.0))) > 0 for k in ("X", "Y", "Z"))

    def selection_names(self):
        """Named selections the expressions can use: one per support/load/thermal face group, plus unions."""
        names = []
        if getattr(self.parent, "constraint_data", None):
            names += [f"Support{i + 1}" for i in range(len(self.parent.constraint_data))] + ["Supports"]
        if getattr(self.parent, "force_data", None):
            names += [f"Load{i + 1}" for i in range(len(self.parent.force_data))] + ["Loads"]
        tw = getattr(self.parent, "thermal_loads_window", None)
        tl = getattr(tw, "thermal_loads", None) or {}
        names += [f"FixedTemp{i + 1}" for i in range(len(tl.get("fixed_temps", [])))]
        names += [f"HeatSource{i + 1}" for i in range(len(tl.get("heat_sources", [])) + len(tl.get("total_heat_sources", [])))]
        if (getattr(self.parent, "topopt_options", None) or {}).get("other", {}).get("keep_fixed_faces"):
            names.append("FixedFaces")
        return names

    # ------------------------------------------------------------------ validate / check / start
    def _plan(self):
        """Build the FE problem and compile the formulation. Returns (plan, fe_solver, physics) or shows errors."""
        from pyto.topopt.spec import PYTHON_PREFIX
        try:
            spec = self.formulation.get_spec()
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Formulation", str(e))
            return None
        allow_code = False
        texts = [spec.objective.expression] + [c.expression for c in spec.active_constraints]
        if any(t.strip().startswith(PYTHON_PREFIX) for t in texts):
            reply = QtWidgets.QMessageBox.question(
                self, "Run Python code?",
                "This formulation calls a Python file. Loading it runs the code in that file.\n\nRun it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return None
            allow_code = True
        spec.manufacturing = manufacturing_from_topopt_options(getattr(self.parent, "topopt_options", None))
        fe_solver, physics, selections = self.build_problem()
        self.formulation.set_context(physics, selections, self.has_body_force())
        plan = prepare_run(spec, physics, selections, body_force=self.has_body_force(), allow_code=allow_code,
                           fe_solver=fe_solver)
        self.parent.optimization_spec = spec
        return plan, fe_solver, physics, spec

    def validate_formulation(self):
        result = self._plan()
        if result is None:
            return
        plan, fe_solver, physics, spec = result
        lines = [f"Physics: {physics}", f"Method: {spec.method.name}", f"Gradient: {plan.gradient}"]
        if plan.cost:
            lines.append(plan.cost)
        lines += [str(i) for i in plan.issues] or ["No problems found."]
        if plan.to_params is not None and not plan.errors:
            try:   # dry run: one FE solve, value of every function
                import torch as _torch
                from pyto.autodiff.qoi import compute_objective_and_gradient, compute_constraint_and_gradient
                from pyto.topopt.drivers._shared import compute_element_stiffness
                from pyto.autodiff.material_model import MaterialModel
                vf0 = next((c.bound for c in spec.active_constraints
                            if c.expression.replace(" ", "") == "VolumeFraction()"), 0.5)
                x = _torch.full((fe_solver.mesh.num_elems,), float(vf0), dtype=_torch.float64)
                KE = _torch.tensor(compute_element_stiffness(fe_solver))
                sol = fe_solver.solve(x, MaterialModel.SIMP)
                obj = compute_objective_and_gradient(plan.to_params, sol, x, fe_solver, KE, MaterialModel.SIMP)
                cons, _ = compute_constraint_and_gradient(plan.to_params, sol, x, fe_solver, KE, MaterialModel.SIMP)
                lines.append(f"At a uniform density {vf0:g}: objective {display_objective(spec, [obj.item()])[0]:.4g}; "
                             f"constraints (<= 0 when satisfied): {', '.join(f'{v:.3g}' for v in cons.reshape(-1).tolist()) or 'none'}")
            except Exception as e:
                lines.append(f"ERROR in the dry run: {e}")
        text = "\n".join(lines)
        self.parent.message_text.append(text)
        QtWidgets.QMessageBox.information(self, "Formulation check", text)

    def check_gradient(self):
        result = self._plan()
        if result is None:
            return
        plan, fe_solver, physics, spec = result
        if plan.to_params is None:
            QtWidgets.QMessageBox.warning(self, "Gradient check", "\n".join(str(i) for i in plan.errors))
            return
        from pyto.topopt.capabilities import check_gradient
        results = check_gradient(plan.to_params, fe_solver, n_elements=5)
        source = "Analytical" if gradient_method(plan.to_params.Gradient) is GRADIENT_METHOD.ANALYTICAL else "Autograd"
        text = f"{source} vs central finite differences at a random design (5 elements):\n" + "\n".join(
            f"  {r.name}: max relative error {r.max_rel_error:.2e}  {'OK' if r.ok else 'MISMATCH'}" for r in results)
        self.parent.message_text.append(text)
        QtWidgets.QMessageBox.information(self, "Gradient check", text)

    def start_optimization(self):
        """Validate and compile the formulation, then run the chosen method in a worker thread."""
        if not self.check_prerequisites():
            return
        result = self._plan()
        if result is None:
            return
        plan, fe_solver, physics, spec = result
        if plan.errors:
            QtWidgets.QMessageBox.warning(self, "Cannot optimize", "\n".join(str(i) for i in plan.issues))
            return
        for i in plan.issues:
            self.parent.message_text.append(str(i))
        self.spec, self.physics = spec, physics
        self.history_figure.clear()
        self.history_canvas.draw_idle()
        self.formulation.log_view.clear()
        self.formulation.setCurrentWidget(self.formulation.log_view)

        self.optimization_running = True
        self.optimize_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.parent.message_text.append(
            f"Starting {physics} topology optimization: {spec.objective.sense} {spec.objective.expression}; "
            f"constraints: {', '.join(f'{c.expression} {c.op} {c.bound:g}' for c in spec.active_constraints) or 'none'}")
        self.parent.message_text.append(f"Method: {spec.method.name}. Gradient: {plan.gradient}. {plan.cost}")

        import threading
        self.optimization_thread = threading.Thread(target=self.run_optimization, args=(plan, fe_solver, spec))
        self.optimization_thread.daemon = True
        self.optimization_thread.start()
        return True

    @QtCore.pyqtSlot(object)
    def update_history_plot(self, history):
        """Objective and constraints against iteration (constraints in the normalized <= 0 form)."""
        self.history_figure.clear()
        ax = self.history_figure.add_subplot(111)
        J = display_objective(self.spec, history.get("objective", []))
        ax.plot(J, color="tab:blue", label="objective")
        ax.set_xlabel("iteration"); ax.set_ylabel("objective", color="tab:blue")
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))     # iterations are whole numbers
        cons = [k for k in history if k.startswith("constraint_")]
        if cons:
            ax2 = ax.twinx()
            for k in cons:
                ax2.plot([float(np.ravel(v)[0]) for v in history[k]], linestyle=":", label=k.replace("_", " "))
            ax2.axhline(0.0, color="grey", linewidth=0.8)
            ax2.set_ylabel("constraint (<= 0)")
            ax2.legend(loc="center right")                # default size, as the tick labels; clear of both curves once converged
        self.history_canvas.draw_idle()

    def run_optimization(self, plan, fe_solver, spec):
        """Run the chosen method (worker thread); what the optimizer prints goes to the Iterations tab."""
        import contextlib
        with contextlib.redirect_stdout(_SignalLineStream(self.log_line.emit)):
            self._run_optimization(plan, fe_solver, spec)

    def _run_optimization(self, plan, fe_solver, spec):
        to_params = plan.to_params
        method = spec.method.name
        self.to_params = to_params
        self.fe_solver = fe_solver

        success = False
        error_msg = ""
        u, history, n_feas = None, {"objective": [], "volfrac": []}, 0

        def iteration_callback(h):
            self.history_update.emit({k: list(v) for k, v in h.items() if k != "change"})
            if self.show_each_iteration:
                try:
                    self.optimization_update.emit(density_snapshot(fe_solver.mesh), len(h.get("objective", [])) - 1)
                except Exception as e:
                    print(f"[live topology] {e}")

        def progress_callback(*args):
            """Driver log messages go to the GUI log (MMA draws the design from iteration_callback)."""
            if len(args) == 1 and isinstance(args[0], str):
                self.optimization_progress.emit(str(args[0]))

        try:
            if method == "MMA":
                u, history, success, error_msg, n_feas = topopt_mma(
                    fe_solver,
                    to_params=to_params,
                    maxMMAIterations=to_params.MaxIterations,
                    move_limit=spec.method.move_limit,
                    print_progress=True,
                    plot_progress=False,
                    binarize_topology=False,   # the GUI shows the continuous density (as before this change)
                    progress_callback=progress_callback,
                    iteration_callback=iteration_callback,
                    plotter=None,
                )
                if success:
                    self.parent.optimized_x = u

            else:
                error_msg = f"Method {method} not implemented"
                success = False
                
        except Exception as e:
            error_msg = f"Optimization failed with error: {str(e)}"
            success = False
        
        self.optimization_done.emit(success, error_msg, history, u, fe_solver)

    @QtCore.pyqtSlot(object, int)
    def update_visualization(self, pv_mesh, iteration):
        """
        Update visualization during optimization.
        Runs on MAIN THREAD - safe for OpenGL operations.
        
        Args:
            pv_mesh: PyVista mesh with density data (created in worker thread)
            iteration: Current iteration number
        """
    
        try:
            # Store geometry_info actor if it exists
            geometry_info_actor = None
            if 'geometry_info' in self.parent.plotter.actors:
                geometry_info_actor = self.parent.plotter.actors['geometry_info']
            
            # Clear all actors except geometry_info
            for name in list(self.parent.plotter.actors.keys()):
                if name != 'geometry_info':
                    self.parent.plotter.remove_actor(name, reset_camera=False)
            
            # Add the mesh directly (no FEA solver calls - pure OpenGL)
            self.parent.plotter.add_mesh(
                pv_mesh,
                scalars='density',
                cmap=DENSITY_CMAP,
                show_edges=pv_mesh.n_cells <= 20000,       # edges would hide the density on fine meshes
                edge_color='black',
                line_width=0.5,
                clim=[0, 1],
                show_scalar_bar=True,
                scalar_bar_args={
                    'title': 'Density',
                    'vertical': True,
                    'height': 0.25,
                    'width': 0.05,
                    'position_x': 0.9,
                    'position_y': 0.02
                }
            )
            
            # Add iteration text
            self.parent.plotter.add_text(
                f"Iteration {iteration + 1}",
                position='upper_edge',
                font_size=12,
                color='black',
                font='arial'
            )
            
            # Restore geometry_info actor if it existed
            if geometry_info_actor:
                if 'geometry_info' in self.parent.plotter.actors and self.parent.plotter.actors['geometry_info'] != geometry_info_actor:
                    self.parent.plotter.remove_actor('geometry_info', reset_camera=False)
                self.parent.plotter.add_actor(geometry_info_actor, name='geometry_info')
            
            # Render the updated scene
            self.parent.plotter.render()
            #QtCore.QCoreApplication.processEvents()
       
            
        except Exception as e:
            print(f"Error updating visualization: {e}")
            import traceback
            traceback.print_exc()


    @QtCore.pyqtSlot(bool, str, object, object, object)
    def optimization_completed(self, success, error_msg, history, u, fe_solver):
        """Handle optimization completion"""
        self.optimization_running = False
        self.optimize_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # A run that stopped at its iteration limit (e.g. 'Maximum iterations reached') still produced a
        # design; keep it viewable instead of discarding it.
        has_design = history is not None and bool(history.get('objective') if isinstance(history, dict) else False)
        if (success or has_design) and history is not None:
            if not success:
                self.parent.message_text.append(
                    f"The optimization did not converge ({error_msg}); showing the design it reached.")
            # Handle different history formats from different optimization methods
            final_objective = 0.0
            final_volume = 0.0
            n_iterations = 0
            
            # Check which keys are available in history
            if isinstance(history, dict):
                # Try to get objective value
                if 'objective' in history and history['objective']:
                    final_objective = history['objective'][-1]
                    n_iterations = len(history['objective'])
                elif 'compliance' in history and history['compliance']:
                    final_objective = history['compliance'][-1]
                    n_iterations = len(history['compliance'])
                
                if history.get('volfrac'):
                    final_volume = float(np.ravel(history['volfrac'][-1])[0])
                if history.get('objective'):
                    final_objective = display_objective(self.spec, [history['objective'][-1]])[0]
                self.history_update.emit({k: list(v) for k, v in history.items() if k != "change"})
            
            self.parent.message_text.append(f"Topology optimization ({self.physics}) finished")
            self.parent.message_text.append(f"Method: {self.spec.method.name}")
            self.parent.message_text.append(f"Final objective: {final_objective:.3g}, Volume fraction: {final_volume:.3f}")
            
            self.parent.update_LivVar('topopt.structural_performed', True)
            self.parent.set_sidebar_icon("TopOpt Execute", "check")
            self.parent.set_sidebar_icon("TopOpt Postprocess", "arrow")
            
            self.parent.topopt_results = {
                'method': self.spec.method.name,
                'spec': self.spec.to_dict(),
                'volume_fraction': final_volume,
                'objective': final_objective,
                'iterations': n_iterations,
                'converged': bool(success),
                'density': np.asarray(self.parent.hex_mesh.elemPseudoDensity).copy(),
                'history': history,
                'displacement': u,
                'fe_solver': fe_solver
            }
            
            self.visualize_optimized_topology(fe_solver)

            # Evaluate the optimized design (stresses / temperatures on the solid region) for the log and for
            # Display Options -> Geometry "TopOpt" (Deformation, Von Mises stress, Temperature).
            try:
                summary = final_design_summary(fe_solver, self.physics)
                self.parent.topopt_results['design_summary'] = summary
                self.parent.topopt_results['physics'] = self.physics
                for line in summary['lines']:
                    self.parent.message_text.append(line)
                self.parent.message_text.append(
                    "Use the 'Optimized design' buttons (here or in TopOpt Postprocess) to show its density, "
                    "deformation, von Mises stress or temperature.")
            except Exception as e:
                self.parent.message_text.append(f"Could not evaluate the final design: {e}")

            stl_path = self.parent.stl_geom.file_path
            vtu_path = os.path.splitext(stl_path)[0] + ".vtu"
            mesh = self.parent.hex_mesh
            mesh.export_vtu_mesh(
                elem_field=mesh.elemPseudoDensity,
                mask_low_pseudodensity=False,
                density_field='density',
                file_name=vtu_path
            )
            self.parent.message_text.append(f"VTU exported: {vtu_path}")
            refresh_topopt_field_buttons(self.parent, self.field_buttons)
            for win in self.parent.findChildren(AnalysisWindow):    # an Analysis window left open
                win.refresh_result_buttons()
            
        else:
            self.parent.message_text.append(f"Structural topology optimization failed: {error_msg}")

    def build_problem(self):
        """FE solver for the current loads (structural, thermal or coupled) and the named selections.

        Selections: SupportN / Supports (fixed faces), LoadN / Loads (loaded faces), FixedTempN, HeatSourceN
        (thermal faces), FixedFaces (elements of the fixed faces, for 'keep fixed faces'). Node sets are the surface
        nodes of the picked triangles, mapped exactly as for the analysis."""
        mesh = self.parent.hex_mesh
        analysis_window = AnalysisWindow(self.parent)
        analysis_window.prepare_mesh_for_analysis(mesh, "structural")
        boundary_nodes, boundary_points, tolerance = analysis_window.get_boundary_mapping_data()
        to_nodes = lambda tris: sorted(analysis_window.map_triangles_to_surface_nodes(
            tris, boundary_nodes, boundary_points, tolerance))
        selections = {}
        physics = self.problem_physics()

        structural_bc, mesh_processed = None, mesh
        if physics in ("structural", "thermo-structural"):
            fixed_nodes = {'xyz': set(), 'x': set(), 'y': set(), 'z': set()}
            all_support = set()
            for i, constraint in enumerate(self.parent.constraint_data):
                surface_nodes = sorted(analysis_window.map_bc_entry_to_nodes(
                    constraint, boundary_nodes, boundary_points, tolerance))
                selections[f"Support{i + 1}"] = {"nodes": np.array(surface_nodes, dtype=int)}
                all_support.update(surface_nodes)
                key = {'Fixed XYZ': 'xyz', 'Fixed X': 'x', 'Fixed Y': 'y', 'Fixed Z': 'z'}.get(constraint['type'])
                if key:
                    fixed_nodes[key].update(surface_nodes)
            if all_support:
                selections["Supports"] = {"nodes": np.array(sorted(all_support), dtype=int)}

            load_nodes_groups, load_forces, all_loaded = [], [], set()
            for i, force_info in enumerate(getattr(self.parent, "force_data", None) or []):
                surface_nodes = sorted(analysis_window.map_bc_entry_to_nodes(
                    force_info, boundary_nodes, boundary_points, tolerance))
                selections[f"Load{i + 1}"] = {"nodes": np.array(surface_nodes, dtype=int)}
                all_loaded.update(surface_nodes)
                if not surface_nodes:
                    continue
                if force_info.get('type') == 'torque':
                    # Distribute torque into tangential forces (same logic as run_structural_analysis)
                    axis_point = np.array(force_info.get('axis_point', [0, 0, 0]), dtype=float)
                    direction = np.array(force_info.get('direction', [0, 0, 1]), dtype=float)
                    torque_value = force_info.get('torque', 0.0)
                    if abs(torque_value) < 1e-12 or np.linalg.norm(direction) < 1e-12:
                        continue
                    direction /= np.linalg.norm(direction)
                    nodes = list(surface_nodes)
                    node_xyz = mesh.node_xyz[nodes]
                    r_vecs = node_xyz - np.mean(node_xyz, axis=0)
                    r_proj = r_vecs - np.outer(r_vecs @ direction, direction)
                    r_norm = np.linalg.norm(r_proj, axis=1)
                    r_norm[r_norm < 1e-12] = 1e-12
                    tangent_dirs = np.cross(direction, r_proj)
                    tangent_dirs /= np.linalg.norm(tangent_dirs, axis=1)[:, None]
                    raw_force = tangent_dirs * r_norm[:, None]
                    torque_actual = np.sum(np.cross(r_proj, raw_force), axis=0)
                    force_vecs = raw_force * (torque_value / (torque_actual @ direction + 1e-12))
                    for node_id, fvec in zip(nodes, force_vecs):
                        load_nodes_groups.append([node_id])
                        load_forces.append(fvec.tolist())
                else:
                    fx, fy, fz = force_info.get('force_x', 0.0), force_info.get('force_y', 0.0), force_info.get('force_z', 0.0)
                    if abs(fx) + abs(fy) + abs(fz) < 1e-12:
                        continue
                    load_nodes_groups.append(list(surface_nodes))
                    load_forces.append([fx, fy, fz])
            if all_loaded:
                selections["Loads"] = {"nodes": np.array(sorted(all_loaded), dtype=int)}
            if not load_forces and not self.has_body_force():
                self.parent.message_text.append("Warning: No effective loads (torque/forces) mapped for TopOpt.")
            mesh_processed, _mat_struct, structural_bc = analysis_window.process_data_for_solver(
                mesh, fixed_nodes, {'load_nodes_groups': load_nodes_groups, 'load_forces': load_forces},
                self.parent.applied_material['properties'])

        thermal_bc = None
        if physics in ("thermal", "thermo-structural"):
            thermal_loads = self.parent.thermal_loads_window.thermal_loads
            thermal_bc = analysis_window.process_thermal_boundary_conditions(
                mesh, thermal_loads, boundary_nodes, boundary_points, tolerance)
            for i, t in enumerate(thermal_loads.get('fixed_temps', [])):
                selections[f"FixedTemp{i + 1}"] = {"nodes": np.array(to_nodes(t['triangles']), dtype=int)}
            sources = thermal_loads.get('heat_sources', []) + thermal_loads.get('total_heat_sources', [])
            for i, t in enumerate(sources):
                selections[f"HeatSource{i + 1}"] = {"nodes": np.array(to_nodes(t['triangles']), dtype=int)}

        if (getattr(self.parent, "topopt_options", None) or {}).get("other", {}).get("keep_fixed_faces"):
            elems = analysis_window.map_triangles_to_elements(
                list(self.parent.constrained_triangles), boundary_nodes, boundary_points, tolerance)
            selections["FixedFaces"] = {"elements": np.array(sorted(elems), dtype=int)}

        mat_prop = analysis_window.create_material_properties()      # all properties (thermal too)
        solver = analysis_window.get_solver()
        body_force = gui_elem_body_force(self.parent, mesh_processed, mat_prop)
        if physics == "structural":
            fe_solver = hex_structural_fea.HexStructuralFEA(mesh=mesh_processed, mat_prop=mat_prop, bc=structural_bc,
                                                            solver=solver, rtol=1e-8, elem_body_force=body_force)
        elif physics == "thermal":
            fe_solver = HexThermalFEA(mesh=mesh, mat_prop=mat_prop, bc=thermal_bc, solver=solver, rtol=1e-8)
        else:
            fe_solver = ThermoStructuralFEA(mesh=mesh_processed, mat_prop=mat_prop, structural_bc=structural_bc,
                                            thermal_bc=thermal_bc, solver=solver, rtol=1e-8,
                                            elem_body_force=body_force,
                                            thermoElasticReferenceTemperature=self.formulation.reference_temperature)
        self.physics = physics
        return fe_solver, physics, selections

    def check_prerequisites(self):
        """Check if all prerequisites for optimization are met"""
        if self.parent.hex_mesh is None:
            QtWidgets.QMessageBox.warning(self, "No Mesh", "Please generate mesh first.")
            return False
        
        if self.parent.applied_material is None:
            QtWidgets.QMessageBox.warning(self, "No Material", "Please define material properties first.")
            return False
        
        # Supports in any form (faces, nodes, boxes): constraint_data, not only the support faces
        # (constrained_triangles), which made node/box supports look missing.
        physics = self.problem_physics()
        tw = getattr(self.parent, "thermal_loads_window", None)
        thermal_loads = getattr(tw, "thermal_loads", None) or {}
        missing = topopt_load_problems(
            physics, has_supports=bool(self.parent.constraint_data), has_forces=bool(self.parent.force_data),
            has_body_force=self.has_body_force(),
            has_thermal=any(thermal_loads.get(k) for k in ("fixed_temps", "heat_sources", "total_heat_sources")))
        if missing:
            QtWidgets.QMessageBox.warning(self, "Incomplete Loads",
                                          f"For a {physics} optimization, please apply: " + "; ".join(missing) + ".")
            return False
        
        if self.parent.topopt_options is None:
            QtWidgets.QMessageBox.warning(self, "No Constraints", "Please define topology optimization constraints first.")
            return False
        
        return True
    
    def visualize_optimized_topology(self, fe_solver):
        """Visualize the optimized topology"""
        # Store geometry_info if exists
        geometry_info_actor = None
        if 'geometry_info' in self.parent.plotter.actors:
            geometry_info_actor = self.parent.plotter.actors['geometry_info']
        
        # Clear all actors except geometry_info
        for name in list(self.parent.plotter.actors.keys()):
            if name != 'geometry_info':
                self.parent.plotter.remove_actor(name, reset_camera=False)
        
        # Plot final optimized mesh
        fe_solver.plot_mesh(plotter=self.parent.plotter)
        
        # Force restore geometry_info by re-adding it
        if geometry_info_actor:
            # Remove any conflicting actor with same name
            if 'geometry_info' in self.parent.plotter.actors and self.parent.plotter.actors['geometry_info'] != geometry_info_actor:
                self.parent.plotter.remove_actor('geometry_info', reset_camera=False)
            # Re-add the original geometry_info actor
            self.parent.plotter.add_actor(geometry_info_actor, name='geometry_info')
        
        self.parent.plotter.render()
        self.parent.message_text.append("Optimized topology visualized")
    
    def stop_optimization(self):
        """Stop the optimization process"""
        if self.optimization_running:
            self.optimization_running = False
            self.parent.message_text.append("Structural topology optimization stopped by user")
            
            self.optimize_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        """Handle window close event"""
        if self.optimization_running:
            reply = QtWidgets.QMessageBox.question(
                self, 
                'Optimization Running', 
                'Optimization is still running. Do you want to stop it and close?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            
            if reply == QtWidgets.QMessageBox.Yes:
                self.stop_optimization()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
#----------------------------------------------------------------------------
class TopOptResultsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("TopOpt Postprocess")
        self.setFixedSize(320, 280)
        self.parent = parent

        # Default values for tet mesh
        self.default_tet_elems = 10000

        # Layout
        layout = QtWidgets.QVBoxLayout(self)

        # Smoothing of the density before the isosurface (Results/SmoothingComparison): the CNN (default) or a
        # Gaussian filter, deterministic and much faster; sigma in element sizes (wider erodes thin members)
        smooth_layout = QtWidgets.QHBoxLayout()
        smooth_layout.addWidget(QtWidgets.QLabel("Smoothing"))
        self.smoothing_combo = QtWidgets.QComboBox()
        self.smoothing_combo.addItem("CNN", "cnn")
        self.smoothing_combo.addItem("Gaussian filter", "gaussian")
        self.smoothing_combo.setCurrentIndex(self.smoothing_combo.findData(DEFAULT_SMOOTHING))
        smooth_layout.addWidget(self.smoothing_combo)
        layout.addLayout(smooth_layout)
        sigma_layout = QtWidgets.QHBoxLayout()
        sigma_layout.addWidget(QtWidgets.QLabel("Gaussian width (elements)"))
        self.sigma_spin = QtWidgets.QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 3.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(DEFAULT_GAUSSIAN_SIGMA)
        sigma_layout.addWidget(self.sigma_spin)
        layout.addLayout(sigma_layout)
        self.smoothing_combo.currentIndexChanged.connect(
            lambda _: self.sigma_spin.setEnabled(self.smoothing_combo.currentData() == "gaussian"))
        self.sigma_spin.setEnabled(self.smoothing_combo.currentData() == "gaussian")

        # Apply Postprocess button
        self.apply_btn = QtWidgets.QPushButton("Apply Postprocess")
        self.apply_btn.clicked.connect(self.apply_recovery)
        layout.addWidget(self.apply_btn)

        # Tet mesh controls
        tet_layout = QtWidgets.QHBoxLayout()
        tet_layout.addWidget(QtWidgets.QLabel("Tet elements"))
        self.tet_elem_spin = QtWidgets.QSpinBox()
        self.tet_elem_spin.setMinimum(100)
        self.tet_elem_spin.setMaximum(10000000)
        self.tet_elem_spin.setSingleStep(1000)
        self.tet_elem_spin.setValue(self.default_tet_elems)
        tet_layout.addWidget(self.tet_elem_spin)
        layout.addLayout(tet_layout)

        self.tetmesh_btn = QtWidgets.QPushButton("Generate TetMesh")
        self.tetmesh_btn.clicked.connect(self.generate_tetmesh)
        layout.addWidget(self.tetmesh_btn)


    def apply_recovery(self):
        """Apply STL recovery postprocessing"""
        topopt_results = getattr(self.parent, "topopt_results", None)
        if not topopt_results or not hasattr(self.parent, "stl_geom"):
            QtWidgets.QMessageBox.warning(self, "No TopOpt Results", "Please run topology optimization first.")
            return

        stl_path = self.parent.stl_geom.file_path
        vtu_path = os.path.splitext(stl_path)[0] + ".vtu"

        try:
            # clean(): merge the STL's per-triangle duplicate vertices, so the Boolean operation sees one closed volume
            design_domain_stl = pv.read(stl_path).clean().triangulate().compute_normals()
            vtu = pv.read(vtu_path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "File Error", f"Could not load STL/VTU: {e}")
            return

        # Disable button during processing
        self.apply_btn.setEnabled(False)
        self.apply_btn.setText("Processing...")
        QtWidgets.QApplication.processEvents()

        self.parent.message_text.append("Starting STL recovery postprocessing...")
        QtWidgets.QApplication.processEvents()

        try:
            method, sigma = self.smoothing_combo.currentData(), self.sigma_spin.value()
            self.parent.message_text.append(
                "Extracting void isosurface, smoothing: " + ("CNN..." if method == "cnn" else f"Gaussian filter, width {sigma:g} elements..."))
            QtWidgets.QApplication.processEvents()

            void_region_stl = extract_isosurface(vtu, isovalue=0.5, method=method, sigma=sigma)
            
            self.parent.message_text.append("Performing boolean subtraction...")
            QtWidgets.QApplication.processEvents()
            
            optimized_topology_stl = subtract_voids_from_stl(design_domain_stl, void_region_stl)
            
            self.parent.message_text.append("STL recovery completed successfully.")
            
        except Exception as e:
            self.parent.message_text.append(f"STL recovery failed: {e}")
            QtWidgets.QMessageBox.warning(self, "Recovery Failed", f"Could not generate optimized STL: {e}")
            self.apply_btn.setEnabled(True)
            self.apply_btn.setText("Apply Postprocess")
            return

        self.parent.optimized_topology_stl = optimized_topology_stl

        # Sanity check: the recovered volume against the voxel design (elements with density >= 0.5)
        density = np.asarray(topopt_results.get("density", []), dtype=float)
        if density.size and getattr(self.parent, "hex_mesh", None) is not None:
            voxel_volume = float((density >= 0.5).sum() * np.prod(self.parent.hex_mesh.elem_size))
            ok, ratio = recovered_volume_check(optimized_topology_stl, voxel_volume)
            self.parent.message_text.append(f"Recovered volume / voxel design volume: {ratio:.3f}")
            if not ok:
                QtWidgets.QMessageBox.warning(
                    self, "Check the recovered part",
                    f"The recovered part's volume is {ratio:.2f} times the voxel design's (expected about 1). "
                    "The part is saved, but check it before use: the design-domain STL may not match the mesh, or "
                    "the recovery removed or kept too much material.")

        # Remove only mesh actors, keep geometry info
        for name in list(self.parent.plotter.actors.keys()):
            if name != "geometry_info":
                self.parent.plotter.remove_actor(name, reset_camera=False)

        self.parent.plotter.add_mesh(optimized_topology_stl, color=MATERIAL_COLOR, opacity=1.0, name="Optimized STL")
        self.parent.plotter.reset_camera()
        self.parent.plotter.render()
        self.parent.message_text.append("Optimized STL visualized.")

        # Save STL
        output_path = os.path.splitext(stl_path)[0] + "_optimized.stl"
        try:
            optimized_topology_stl.save(output_path)
            self.parent.message_text.append(f"Optimized STL saved: {output_path}")
            self.parent.set_sidebar_icon("TopOpt Postprocess", "check")
        except Exception as e:
            self.parent.message_text.append(f"Failed to save optimized STL: {e}")

        # Re-enable buttons
        self.apply_btn.setEnabled(True)
        self.apply_btn.setText("Apply Postprocess")


    def generate_tetmesh(self):
        if not getattr(self.parent, "stl_geom", None):
            QtWidgets.QMessageBox.warning(self, "No Geometry", "Please load a geometry first.")
            return
        stl_path = os.path.splitext(self.parent.stl_geom.file_path)[0] + "_optimized.stl"
        if not os.path.exists(stl_path):
            QtWidgets.QMessageBox.warning(
                self, "No Optimized STL",
                "Run 'Apply Postprocess' first to generate the optimized STL."
            )
            return
        n_elems = self.tet_elem_spin.value()
        
        # Check STL manifoldness and watertightness
        import trimesh
        mesh = trimesh.load(stl_path)
        if not mesh.is_watertight:
            trimesh.repair.fill_holes(mesh)
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.update_faces(mesh.unique_faces())
            mesh.remove_unreferenced_vertices()
            mesh.export(stl_path)
            mesh = trimesh.load(stl_path)

        # Robust non-manifold edge check
        non_manifold_edges = getattr(mesh, "edges_non_manifold", [])
        if non_manifold_edges is None or isinstance(non_manifold_edges, bool):
            # fallback for older trimesh
            non_manifold_edges = mesh.edges_unique[mesh.edges_unique_length != 2]

        if not mesh.is_watertight or len(non_manifold_edges) > 0:
            QtWidgets.QMessageBox.warning(
                self, "STL Error",
                f"STL is not watertight or has non-manifold edges.\n"
                f"Watertight: {mesh.is_watertight}\n"
                f"Non-manifold edges: {len(non_manifold_edges)}"
            )
            return
            
        tetmesh = TetMesher()
        tetmesh.createTetMeshFromSTLFile(stl_path, nElemsDesired=n_elems)
        self.parent.tetmesh = tetmesh
        
        # Visualize tetmesh and keep geometry info
        for name in list(self.parent.plotter.actors.keys()):
            if name != "geometry_info":
                self.parent.plotter.remove_actor(name, reset_camera=False)
        tetmesh.plot(plotter=self.parent.plotter)
        self.parent.update_geometry_info_text()
#----------------------------------------------------------------------------
class DisplayOptionsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Display Options")
        self.setFixedSize(300, 680)
        self.parent = parent

        self.setup_ui()
        self.connect_signals()

    def setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        geometry_layout = QtWidgets.QHBoxLayout()
        geometry_layout.addWidget(QtWidgets.QLabel("Geometry"))
        self.geometry_combo = QtWidgets.QComboBox()
        self.geometry_combo.addItems(["Initial Design"])
        geometry_layout.addWidget(self.geometry_combo)
        layout.addLayout(geometry_layout)

        field_layout = QtWidgets.QHBoxLayout()
        field_layout.addWidget(QtWidgets.QLabel("Field"))
        self.field_combo = QtWidgets.QComboBox()
        self.field_combo.addItems(["None"])
        field_layout.addWidget(self.field_combo)
        layout.addLayout(field_layout)

        display_layout = QtWidgets.QHBoxLayout()
        display_layout.addWidget(QtWidgets.QLabel("Display On"))
        self.display_combo = QtWidgets.QComboBox()
        self.display_combo.addItems(["Geometry"])
        display_layout.addWidget(self.display_combo)
        layout.addLayout(display_layout)

        self.x_cutting_spin = self.create_cutting_control(layout, "XCuttingPercent", 0)
        self.y_cutting_spin = self.create_cutting_control(layout, "YCuttingPercent", 0)
        self.z_cutting_spin = self.create_cutting_control(layout, "ZCuttingPercent", 0)

        eigen_layout = QtWidgets.QHBoxLayout()
        eigen_layout.addWidget(QtWidgets.QLabel("EigenNumber"))
        self.eigen_spin = QtWidgets.QSpinBox()
        self.eigen_spin.setMinimum(1)
        self.eigen_spin.setMaximum(10)
        self.eigen_spin.setValue(1)
        eigen_layout.addWidget(self.eigen_spin)
        layout.addLayout(eigen_layout)

        self.show_bounding_box_checkbox = QtWidgets.QCheckBox("Show bounding box")
        layout.addWidget(self.show_bounding_box_checkbox)

        self.show_triangles = QtWidgets.QCheckBox("Show triangles")
        layout.addWidget(self.show_triangles)

        self.show_text = QtWidgets.QCheckBox("Show text")
        self.show_text.setChecked(True)
        layout.addWidget(self.show_text)

        self.show_transparent_geometry = QtWidgets.QCheckBox("Show transparent geometry")
        layout.addWidget(self.show_transparent_geometry)


        self.show_structural_loads_checkbox = QtWidgets.QCheckBox("Show structural loads")
        layout.addWidget(self.show_structural_loads_checkbox)

        self.show_thermal_loads_checkbox = QtWidgets.QCheckBox("Show thermal loads")
        layout.addWidget(self.show_thermal_loads_checkbox)

        self.show_topopt_options_checkbox = QtWidgets.QCheckBox("Show TopOpt Options")
        layout.addWidget(self.show_topopt_options_checkbox)

        animate_layout = QtWidgets.QHBoxLayout()
        animate_btn = QtWidgets.QPushButton("Animate for 3 cycles")
        animate_btn.clicked.connect(self.animate_view)
        animate_layout.addWidget(animate_btn)
        layout.addLayout(animate_layout)

        self.create_control_buttons(layout)

    def create_cutting_control(self, layout, label, default_value):
        cutting_layout = QtWidgets.QHBoxLayout()
        cutting_layout.addWidget(QtWidgets.QLabel(label))
        cutting_spin = QtWidgets.QSpinBox()
        cutting_spin.setMinimum(0)
        cutting_spin.setMaximum(100)
        cutting_spin.setValue(default_value)
        cutting_layout.addWidget(cutting_spin)
        layout.addLayout(cutting_layout)
        return cutting_spin

    def create_control_buttons(self, layout):
        button_layout = QtWidgets.QVBoxLayout()
        hide_selected_btn = QtWidgets.QPushButton("Hide Selected Part")
        hide_selected_btn.clicked.connect(self.hide_selected_part)
        button_layout.addWidget(hide_selected_btn)
        save_image_btn = QtWidgets.QPushButton("Save Image")
        save_image_btn.clicked.connect(self.save_image)
        button_layout.addWidget(save_image_btn)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)

    def connect_signals(self):
        self.geometry_combo.currentTextChanged.connect(self.update_field_options)   # fields depend on the geometry
        self.geometry_combo.currentTextChanged.connect(self.update_display)
        self.field_combo.currentTextChanged.connect(self.update_display)
        self.display_combo.currentTextChanged.connect(self.update_display)
        self.x_cutting_spin.valueChanged.connect(self.update_cutting)
        self.y_cutting_spin.valueChanged.connect(self.update_cutting)
        self.z_cutting_spin.valueChanged.connect(self.update_cutting)
        for checkbox in [self.show_bounding_box_checkbox, self.show_triangles, self.show_text,
                         self.show_transparent_geometry,
                         self.show_structural_loads_checkbox, self.show_thermal_loads_checkbox,
                         self.show_topopt_options_checkbox]:
            checkbox.stateChanged.connect(self.update_display)
        self.update_checkbox_states()

    def update_checkbox_states(self):
        has_structural = (
            (hasattr(self.parent, 'force_data') and bool(self.parent.force_data)) or
            bool(getattr(self.parent, 'constraint_data', None))          # face and node supports
        )
        self.show_structural_loads_checkbox.setEnabled(bool(has_structural))
        self.show_structural_loads_checkbox.setChecked(bool(has_structural))

        has_thermal = (
            hasattr(self.parent, 'thermal_loads_window') and
            self.parent.thermal_loads_window and
            getattr(self.parent.thermal_loads_window, 'thermal_loads', None) and
            any(self.parent.thermal_loads_window.thermal_loads.get(key, [])
                for key in ['fixed_temps', 'heat_sources', 'total_heat_sources'])
        )
        self.show_thermal_loads_checkbox.setEnabled(bool(has_thermal))
        self.show_thermal_loads_checkbox.setChecked(bool(has_thermal))

        has_constraints = (
            hasattr(self.parent, 'topopt_option_actors') and
            bool(self.parent.topopt_option_actors)
        )
        self.show_topopt_options_checkbox.setEnabled(bool(has_constraints))
        self.show_topopt_options_checkbox.setChecked(bool(has_constraints))

    def showEvent(self, event):
        super().showEvent(event)
        self.update_geometry_options()
        self.update_field_options()
        self.update_checkbox_states()

    def update_display(self):
        geometry_choice = self.geometry_combo.currentText()
        field_choice = self.field_combo.currentText()
        self.remove_scalar_bar()

        should_plot = (
            (geometry_choice == "Initial Design" and self.parent.stl_geom) or
            (geometry_choice == "Mesh" and hasattr(self.parent, "hex_mesh") and self.parent.hex_mesh) or
            (geometry_choice == "TopOpt" and hasattr(self.parent, "topopt_results") and self.parent.topopt_results) or
            (geometry_choice == "Final Design" and hasattr(self.parent, "optimized_topology_stl") and self.parent.optimized_topology_stl is not None)
        )
        if should_plot:
            for name in list(self.parent.plotter.actors.keys()):
                if name != "geometry_info":
                    self.parent.plotter.remove_actor(name, reset_camera=False)

        # Visualization logic
        if geometry_choice == "Mesh":
            if field_choice == "Deformation":
                if hasattr(self.parent, "fe_solver") and self.parent.fe_solver:
                    self.parent.fe_solver.plot_deformation(plotter=self.parent.plotter)
            elif field_choice == "Von Mises stress":
                if hasattr(self.parent, "fe_solver") and self.parent.fe_solver:
                    self.parent.fe_solver.plot_vonMisesStress(plotter=self.parent.plotter)
            elif field_choice == "Temperature":
                if hasattr(self.parent, "thermal_fe_solver") and self.parent.thermal_fe_solver:
                    self.parent.thermal_fe_solver.plot_temperature(plotter=self.parent.plotter)
                else:
                    QtWidgets.QMessageBox.warning(self, "Thermal Results Missing", "Please solve for thermal analysis first.")
            elif field_choice == "None":
                AnalysisWindow(self.parent).visualize_colored_mesh("structural")
            self.toggle_mesh_edges(self.show_triangles.isChecked())
        elif geometry_choice == "Initial Design" and self.parent.stl_geom:
            self.parent.stl_geom.plotGeometry(
                show_edges=self.show_triangles.isChecked(),
                show_bounding_box=self.show_bounding_box_checkbox.isChecked(),
                plotter=self.parent.plotter,
                color=MATERIAL_COLOR
            )
        elif geometry_choice == "TopOpt" and hasattr(self.parent, "topopt_results") and self.parent.topopt_results:
            msg = show_topopt_field(self.parent, "Density" if field_choice == "None" else field_choice)
            if msg:
                QtWidgets.QMessageBox.information(self, "Field not available", msg)
        elif geometry_choice == "Final Design" and hasattr(self.parent, "optimized_topology_stl") and self.parent.optimized_topology_stl is not None:
            self.parent.plotter.add_mesh(
                self.parent.optimized_topology_stl,
                color=MATERIAL_COLOR,
                opacity=1.0,
                name="Optimized STL"
            )

        # Robust toggles for triangles and axes
        self.toggle_mesh_edges(self.show_triangles.isChecked())

        if self.show_bounding_box_checkbox.isChecked():
            self.show_bounding_box(True)
        else:
            self.show_bounding_box(False)
        if self.show_text.isChecked():
            self.show_geometry_text(True)
        else:
            self.show_geometry_text(False)
        if self.show_transparent_geometry.isChecked():
            self.set_geometry_transparency(0.5)
        else:
            self.set_geometry_transparency(1.0)
        if self.show_structural_loads_checkbox.isChecked():
            self.show_structural_loads(True)
        else:
            self.show_structural_loads(False)
        if self.show_thermal_loads_checkbox.isChecked():
            self.show_thermal_loads(True)
        else:
            self.show_thermal_loads(False)
        if self.show_topopt_options_checkbox.isChecked():
            self.show_topopt_options(True)
        else:
            self.show_topopt_options(False)

        self.parent.plotter.render()

    def toggle_mesh_edges(self, show):
        """Show/hide triangle edges for mesh wireframe."""
        if self.geometry_combo.currentText() != "Mesh":
            return  # Only toggle edges if Mesh is selected
        mesh_actor_name = "colored_mesh"
        if mesh_actor_name in self.parent.plotter.actors:
            self.parent.plotter.remove_actor(mesh_actor_name, reset_camera=False)
            mesh = getattr(self.parent, "hex_mesh", None)
            if mesh:
                mesh_polydata = AnalysisWindow(self.parent).create_mesh_polydata(
                    np.full(mesh.num_elems, 0.65)
                )
                self.parent.plotter.add_mesh(
                    mesh_polydata,
                    color='gray',
                    show_edges=show,
                    edge_color='black',
                    line_width=1,
                    show_scalar_bar=False,
                    name=mesh_actor_name
                )

    def toggle_axes(self, show):
        """Show/hide coordinate axes."""
        pass

    def show_bounding_box(self, show):
        # Remove any actor whose name contains "bounding_box" (case-insensitive)
        for name in list(self.parent.plotter.actors.keys()):
            if "bounding_box" in name.lower():
                self.parent.plotter.remove_actor(name, reset_camera=False)
        # Also remove any actor that is a PyVista Box (to catch unnamed bounding boxes)
        for name, actor in list(self.parent.plotter.actors.items()):
            try:
                # Check if actor is a PyVista Box mesh
                if hasattr(actor, "GetMapper") and hasattr(actor.GetMapper(), "GetInput"):
                    mesh = actor.GetMapper().GetInput()
                    # PyVista Box mesh has 8 points and 6 faces
                    if hasattr(mesh, "GetNumberOfPoints") and hasattr(mesh, "GetNumberOfCells"):
                        if mesh.GetNumberOfPoints() == 8 and mesh.GetNumberOfCells() == 6:
                            self.parent.plotter.remove_actor(name, reset_camera=False)
            except Exception:
                pass
        if show:
            if self.parent.stl_geom:
                bounds = self.parent.stl_geom.get_bounding_box()
                if bounds:
                    bbox_mesh = pv.Box(bounds=bounds)
                    self.parent.plotter.add_mesh(
                        bbox_mesh,
                        style='wireframe',
                        color='black',
                        line_width=2,
                        name="bounding_box_display"
                    )

    def show_geometry_text(self, show):
        if show:
            if "geometry_info" not in self.parent.plotter.actors:
                self.add_geometry_info_text()
        else:
            if "geometry_info" in self.parent.plotter.actors:
                self.parent.plotter.remove_actor("geometry_info")

    def add_geometry_info_text(self):
        if not self.parent.stl_geom:
            return
        area, volume, _, _ = self.parent.stl_geom.compute_mass_properties()
        bounds = self.parent.stl_geom.get_bounding_box()
        length_unit = self.parent.settings.get_length_unit_string()
        info_lines = [
            f"Model: {os.path.basename(self.parent.stl_geom.file_path)}",
            f"Volume: {volume:.2e} {length_unit}³",
            f"Length: {bounds[1] - bounds[0]:.2e} {length_unit}" if bounds else "Length: N/A",
            f"Surface Area: {area:.2e} {length_unit}²"
        ]
        self.parent.plotter.add_text(
            "\n".join(info_lines),
            position="upper_left",
            font_size=12,
            color="black",
            name="geometry_info",
            font="arial"
        )

    def set_geometry_transparency(self, opacity):
        for name, actor in self.parent.plotter.actors.items():
            if name == "geometry_info" or not hasattr(actor, 'GetProperty'):
                continue
            prop = actor.GetProperty()
            if hasattr(prop, 'SetOpacity'):
                prop.SetOpacity(opacity)

    def show_structural_loads(self, show):
        if hasattr(self.parent, 'force_actors'):
            for actor in self.parent.force_actors:
                if show:
                    self.parent.plotter.add_actor(actor)
                else:
                    self.parent.plotter.remove_actor(actor, reset_camera=False)
        if hasattr(self.parent, 'constraint_actors'):
            for actor in self.parent.constraint_actors:
                if show:
                    self.parent.plotter.add_actor(actor)
                else:
                    self.parent.plotter.remove_actor(actor, reset_camera=False)
        if hasattr(self.parent, 'torque_actors'):
            for actor in self.parent.torque_actors:
                if show:
                    self.parent.plotter.add_actor(actor)
                else:
                    self.parent.plotter.remove_actor(actor, reset_camera=False)

    def show_thermal_loads(self, show):
        if hasattr(self.parent, 'thermal_loads_window') and self.parent.thermal_loads_window:
            thermal_actor_lists = ['fixed_temp_actors', 'heat_source_actors', 'total_heat_actors']
            for actor_list_name in thermal_actor_lists:
                if hasattr(self.parent.thermal_loads_window, actor_list_name):
                    actor_list = getattr(self.parent.thermal_loads_window, actor_list_name)
                    for actor in actor_list:
                        if show:
                            self.parent.plotter.add_actor(actor)
                        else:
                            self.parent.plotter.remove_actor(actor, reset_camera=False)

    def show_topopt_options(self, show):
        if hasattr(self.parent, 'topopt_option_actors'):
            for actor_key, actor in self.parent.topopt_option_actors.items():
                if actor:
                    if isinstance(actor, list):
                        for a in actor:
                            if show:
                                if a not in self.parent.plotter.actors.values():
                                    self.parent.plotter.add_actor(a)
                            else:
                                self.parent.plotter.remove_actor(a, reset_camera=False)
                    else:
                        if show:
                            if actor not in self.parent.plotter.actors.values():
                                self.parent.plotter.add_actor(actor)
                        else:
                            self.parent.plotter.remove_actor(actor, reset_camera=False)

    def update_cutting(self):
        x_percent = self.x_cutting_spin.value() / 100.0
        y_percent = self.y_cutting_spin.value() / 100.0
        z_percent = self.z_cutting_spin.value() / 100.0
        if any([x_percent > 0, y_percent > 0, z_percent > 0]):
            self.apply_cutting_planes(x_percent, y_percent, z_percent)
        else:
            self.remove_cutting_planes()

    def apply_cutting_planes(self, x_percent, y_percent, z_percent):
        if not self.parent.stl_geom:
            return
        bounds = self.parent.stl_geom.get_bounding_box()
        if not bounds:
            return
        x_cut = bounds[0] + (bounds[1] - bounds[0]) * x_percent
        y_cut = bounds[2] + (bounds[3] - bounds[2]) * y_percent
        z_cut = bounds[4] + (bounds[5] - bounds[4]) * z_percent
        for name, actor in list(self.parent.plotter.actors.items()):
            if name not in ["geometry_info"] and hasattr(actor, 'GetMapper'):
                if x_percent > 0:
                    self.parent.plotter.add_mesh_clip_plane(actor, normal=(-1, 0, 0), origin=(x_cut, 0, 0))
                if y_percent > 0:
                    self.parent.plotter.add_mesh_clip_plane(actor, normal=(0, -1, 0), origin=(0, y_cut, 0))
                if z_percent > 0:
                    self.parent.plotter.add_mesh_clip_plane(actor, normal=(0, 0, -1), origin=(0, 0, z_cut))

    def remove_cutting_planes(self):
        self.update_display()

    def animate_view(self):
        self.parent.plotter.orbit_on_path(n_cycles=3, progress_bar=False)

    def hide_selected_part(self):
        if hasattr(self.parent, 'highlight_actor') and self.parent.highlight_actor:
            self.parent.plotter.remove_actor(self.parent.highlight_actor, reset_camera=False)
            self.parent.highlight_actor = None
            self.parent.plotter.render()

    def save_image(self):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
        )
        if filename:
            self.parent.plotter.screenshot(filename)
            self.parent.message_text.append(f"Image saved: {os.path.basename(filename)}")

    def reset_view(self):
        self.parent.plotter.reset_camera()
        self.parent.plotter.render()

    def on_geometry_changed(self, text):
        if text in ["Initial Design", "TopOpt"]:
            self.field_combo.blockSignals(True)
            self.field_combo.setCurrentText("None")
            self.field_combo.blockSignals(False)
        self.update_display()

    def on_field_changed(self, text):
        if self.geometry_combo.currentText() in ["Initial Design", "TopOpt"] and text != "None":
            QtWidgets.QMessageBox.warning(
                self, "Invalid Selection",
                "To display Deformation, Von Mises stress, or Temperature, switch to Mesh."
            )
            idx = self.geometry_combo.findText("Mesh")
            if idx != -1:
                self.geometry_combo.setCurrentIndex(idx)
            else:
                self.field_combo.blockSignals(True)
                self.field_combo.setCurrentText("None")
                self.field_combo.blockSignals(False)
        self.update_display()

    def update_geometry_options(self):
        items = ["Initial Design"]
        current_index = 0
        has_mesh = hasattr(self.parent, "hex_mesh") and self.parent.hex_mesh is not None
        if has_mesh:
            items.append("Mesh")
            if self.geometry_combo.currentText() == "Mesh":
                current_index = len(items) - 1
        has_topopt = (
            hasattr(self.parent, "topopt_results")
            and self.parent.topopt_results is not None
            and self.parent.topopt_results.get("fe_solver") is not None
        )
        if has_topopt:
            items.append("TopOpt")
            if self.geometry_combo.currentText() in ["TopOpt"]:
                current_index = len(items) - 1
        if hasattr(self.parent, "optimized_topology_stl") and self.parent.optimized_topology_stl is not None:
            items.append("Final Design")
            if self.geometry_combo.currentText() == "Final Design":
                current_index = len(items) - 1
        self.geometry_combo.blockSignals(True)
        self.geometry_combo.clear()
        self.geometry_combo.addItems(items)
        self.geometry_combo.setCurrentIndex(current_index)
        self.geometry_combo.blockSignals(False)

    def update_field_options(self):
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        fields = ["None", "Deformation", "Von Mises stress", "Temperature"]
        results = getattr(self.parent, "topopt_results", None) or {}
        if self.geometry_combo.currentText() == "TopOpt" and results:
            physics = results.get("physics", "structural")      # offer only the fields of that optimization
            fields = ["None"] + [f for f in topopt_fields_available(results) if f != "Density"]
        self.field_combo.addItems(fields)
        if self.geometry_combo.currentText() in ["Initial Design", "TopOpt"]:
            self.field_combo.setCurrentText("None")
        self.field_combo.blockSignals(False)

    def remove_scalar_bar(self):
        if hasattr(self.parent.plotter, 'scalar_bars'):
            for name in list(self.parent.plotter.scalar_bars.keys()):
                self.parent.plotter.remove_scalar_bar(name)

    # def remove_scalar_bar(self):
    #     """Remove all scalar bars from the plotter."""
    #     if hasattr(self.parent.plotter, 'scalar_bars'):
    #         for name in list(self.parent.plotter.scalar_bars.keys()):
    #             self.parent.plotter.remove_scalar_bar(name)
                    
    def showEvent(self, event):
        """Update options when window is shown"""
        super().showEvent(event)
        self.update_geometry_options()
        self.update_field_options() 
#----------------------------------------------------------------------------
class ProjectsWindow(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Projects")
        self.setFixedSize(200, 120)
        self.parent = parent
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # Buttons
        for text, method in [("Save Project", self.save_project), 
                           ("Load Project", self.load_project),
                           ("Close", self.close)]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(method)
            layout.addWidget(btn)

    def save_project(self):
        if not self.parent.stl_geom:
            QtWidgets.QMessageBox.warning(self, "No Data", "No geometry loaded to save.")
            return
                
        # Generate default filename from STL geometry name
        stl_basename = os.path.splitext(os.path.basename(self.parent.stl_geom.file_path))[0]
        default_filename = f"{stl_basename}.pyto"
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Project", default_filename, "PyTO Files (*.pyto)")
        
        if not filename:
            return

        # Store STL path as relative to project file location
        stl_abs_path = os.path.abspath(self.parent.stl_geom.file_path)
        project_dir = os.path.dirname(os.path.abspath(filename))
        try:
            stl_rel_path = os.path.relpath(stl_abs_path, project_dir)
        except Exception:
            stl_rel_path = self.parent.stl_geom.file_path
                
        # Get number of mesh elements - either from existing mesh or default
        # We DO NOT load the mesh, just get the element count if it was already generated
        mesh_elements = 10000  # Default value
        if hasattr(self.parent, 'hex_mesh') and self.parent.hex_mesh:
            mesh_elements = self.parent.hex_mesh.num_elems
        elif hasattr(self.parent, 'analysis_settings') and 'n_elements' in self.parent.analysis_settings:
            mesh_elements = self.parent.analysis_settings['n_elements']
        
        # Get solver type from analysis window if available
        solver_type = "PARDISO"  # Default value
        for child in self.parent.findChildren(QtWidgets.QDialog):
            if isinstance(child, AnalysisWindow) and child.isVisible():
                solver_type = child.solver_combo.currentText()
                break
        
        # Create analysis settings dictionary
        analysis_settings = {
            "n_elements": mesh_elements,
            "solver_type": solver_type
        }
        
        # Create structuralBC section
        structuralBC = {}
        
        # Add fixed faces indices
        fixed_faces_indices = []
        if hasattr(self.parent, 'constraint_data'):
            for constraint in self.parent.constraint_data:
                if constraint.get('type') == 'Fixed XYZ' and not (constraint.get('points') or constraint.get('regions')):
                    fixed_faces_indices.extend(constraint.get('triangles', []))
        
        structuralBC['fixed_faces_indices'] = fixed_faces_indices
        
        # Add load faces indices and forces
        load_faces_indices = []
        load_forces = []
        
        for force_info in self.parent.force_data:
            if force_info.get('type') == 'force_xyz' and not (force_info.get('points') or force_info.get('regions')):
                load_faces_indices.append(force_info.get('triangles', []))
                load_forces.append([
                    force_info.get('force_x', 0.0),
                    force_info.get('force_y', 0.0),
                    force_info.get('force_z', 0.0)
                ])
        
        structuralBC['load_faces_indices'] = load_faces_indices
        structuralBC['load_forces'] = load_forces
        # Node (point) supports and loads, as coordinates (independent of the mesh)
        node_entry = lambda e: bool(e.get('points') or e.get('regions'))
        structuralBC['point_supports'] = [{'type': c['type'], 'points': c.get('points') or [],
                                           'regions': c.get('regions') or []}
                                          for c in getattr(self.parent, 'constraint_data', []) if node_entry(c)]
        structuralBC['point_loads'] = [{'points': f.get('points') or [], 'regions': f.get('regions') or [],
                                        'force': [f.get('force_x', 0.0), f.get('force_y', 0.0), f.get('force_z', 0.0)]}
                                       for f in self.parent.force_data if node_entry(f)]
        
        # Add counts
        structuralBC['constraint_counts'] = {
            'fixed_triangles': len(fixed_faces_indices),
            'loaded_triangles': sum(len(faces) for faces in load_faces_indices)
        }
        
        # Create thermalBC section
        thermalBC = {
            'fixed_temps': [],
            'heat_sources': [],
            'total_heat_sources': [],
            'thermal_counts': {
                'fixed_temps': 0,
                'heat_sources': 0,
                'total_heat_sources': 0,
                'convection': 0
            }
        }
        
        # Populate thermal loads if available
        if getattr(self.parent, 'thermal_loads_window', None):
            thermal_loads = self.parent.thermal_loads_window.thermal_loads
            
            for load_type in ['fixed_temps', 'heat_sources', 'total_heat_sources']:
                for load in thermal_loads.get(load_type, []):
                    cleaned_load = {
                        'triangles': load.get('triangles', [])
                    }
                    
                    # Add value based on type
                    if load_type == 'fixed_temps' and 'temperature' in load:
                        cleaned_load['temperature'] = load['temperature']
                    elif load_type == 'heat_sources' and 'heat_flux' in load:
                        cleaned_load['heat_flux'] = load['heat_flux']
                    elif load_type == 'total_heat_sources' and 'total_heat' in load:
                        cleaned_load['total_heat'] = load['total_heat']
                        
                    thermalBC[load_type].append(cleaned_load)
                    
            # Update counts
            thermalBC['thermal_counts'] = {
                'fixed_temps': sum(len(load.get('triangles', [])) for load in thermalBC['fixed_temps']),
                'heat_sources': sum(len(load.get('triangles', [])) for load in thermalBC['heat_sources']),
                'total_heat_sources': sum(len(load.get('triangles', [])) for load in thermalBC['total_heat_sources']),
                'convection': 0  # Set to actual count if you add convection loads
            }
        
        # Build the complete project data
        project_data = {
            'version': '2025.01',
            'stl_file_path': stl_rel_path,
            'settings': {
                'unit_system': self.parent.settings.unit_system,
                'temperature_unit': self.parent.settings.temperature_unit,
                'angle_unit': self.parent.settings.angle_unit
            },
            'analysis_settings': analysis_settings,
            'material_data': getattr(self.parent, 'applied_material', None),
            'structuralBC': structuralBC,
            'thermalBC': thermalBC,
            'topopt_options': getattr(self.parent, 'topopt_options', None),
            **project_entries(getattr(self.parent, 'optimization_spec', None), getattr(self.parent, 'body_force', None)),
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(project_data, f, indent=2, default=str)
            self.parent.message_text.append(f"Project saved: {os.path.basename(filename)}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save Error", f"Error saving project: {str(e)}")
            self.parent.message_text.append(f"Save failed: {str(e)}")

    def load_project(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Project", "", "PyTO Files (*.pyto)")
        
        if not filename:
            return
        try:
            with open(filename, 'r') as f:
                project_data = json.load(f)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Error loading project: {str(e)}")
            return
        
        # Clear all existing data before loading project
        self.parent.plotter.clear_actors()
        self.parent.plotter.reset_camera()
        self.parent.constrained_triangles.clear()
        self.parent.constraint_actors.clear()
        self.parent.constraint_data.clear()
        self.parent.force_data.clear()
        self.parent.force_actors.clear()
        self.parent.material_data = None
        self.parent.applied_material = None
        self.parent.highlight_actor = None
        self.parent.LivVar = self.parent.create_initial_state()

        # Restore settings
        if 'settings' in project_data:
            s = project_data['settings']
            self.parent.settings.update_settings(
                s.get('unit_system', 'MKS'),
                s.get('temperature_unit', 'Kelvin'),
                s.get('angle_unit', 'Degree'))
        
        # Store mesh elements count and solver (DON'T LOAD MESH)
        # Just save the settings for later use when the user decides to generate a mesh
        mesh_elements = 10000  # Default
        solver_type = "PARDISO"  # Default
        
        if 'analysis_settings' in project_data:
            analysis_settings = project_data['analysis_settings']
            mesh_elements = analysis_settings.get('n_elements', 10000)
            solver_type = analysis_settings.get('solver_type', 'PARDISO')
            self.parent.analysis_settings = analysis_settings
            self.parent.message_text.append(
                f"Loaded analysis settings: {mesh_elements} elements, solver: {solver_type}"
            )
        elif 'mesh_elements' in project_data:
            # Legacy support for older projects
            mesh_elements = project_data.get('mesh_elements', 10000)
            self.parent.analysis_settings = {'n_elements': mesh_elements, 'solver_type': solver_type}
            self.parent.message_text.append(f"Loaded mesh settings: {mesh_elements} elements")
        else:
            # If no settings found, use defaults and store them
            self.parent.analysis_settings = {'n_elements': mesh_elements, 'solver_type': solver_type}
        
        # Store for later mesh generation
        self.parent.mesh_elements = mesh_elements
        
        # Update Analysis Window element count if it exists
        for child in self.parent.findChildren(QtWidgets.QDialog):
            if isinstance(child, AnalysisWindow) and child.isVisible():
                child.elements_spin.setValue(mesh_elements)
                solver_idx = child.solver_combo.findText(solver_type)
                if solver_idx >= 0:
                    child.solver_combo.setCurrentIndex(solver_idx)
        
        # Load geometry
        stl_path = project_data.get('stl_file_path')
        if stl_path:
            # Resolve relative to project file location if not absolute
            if not os.path.isabs(stl_path):
                stl_path_full = os.path.normpath(os.path.join(os.path.dirname(filename), stl_path))
            else:
                stl_path_full = stl_path
            # If not found, try current directory
            if not os.path.exists(stl_path_full):
                stl_basename = os.path.basename(stl_path)
                stl_path_full = os.path.join(os.getcwd(), stl_basename)
            # If still not found, warn
            if os.path.exists(stl_path_full):
                self.load_geometry(stl_path_full)
            else:
                self.parent.message_text.append(f"Warning: STL file not found: {stl_path_full}")
        
        # Restore material
        if project_data.get('material_data'):
            self.parent.applied_material = project_data['material_data']
            self.parent.update_LivVar('material_defined', True)
            self.parent.set_sidebar_icon("Material", "check")
            self.parent.set_sidebar_icon("Structural Loads", "arrow")
            # Update geometry info text with correct material after restoring material
            if getattr(self.parent, "stl_geom", None):
                area, volume, _, _ = self.parent.stl_geom.compute_mass_properties()
                bounds = self.parent.stl_geom.get_bounding_box()
                length_unit = self.parent.settings.get_length_unit_string()
                material_name = self.parent.applied_material.get("name", "None")
                info_lines = [
                    f"Model: {os.path.basename(self.parent.stl_geom.file_path)}",
                    f"Volume: {volume:.2e} {length_unit}³",
                    f"Length: {bounds[1] - bounds[0]:.2e} {length_unit}" if bounds else "Length: N/A",
                    f"Material: {material_name}"
                ]
                self.parent.plotter.remove_actor("geometry_info")
                self.parent.plotter.add_text(
                    "\n".join(info_lines),
                    position="upper_left",
                    font_size=12,
                    color="black",
                    name="geometry_info",
                    font="arial"
                )
        
        # Restore structural loads from structuralBC or legacy format
        if 'structuralBC' in project_data:
            self.restore_structural_loads(project_data['structuralBC'])
        else:
            # Legacy support for older projects
            self.parent.force_data = project_data.get('force_data', [])
            self.parent.constraint_data = project_data.get('constraint_data', [])
            self.parent.constrained_triangles = set(project_data.get('constrained_triangles', []))
            self.restore_structural_loads()
        
        # Restore thermal loads from thermalBC or legacy format
        if 'thermalBC' in project_data:
            self.restore_thermal_loads(project_data['thermalBC'])
        elif 'thermal_loads' in project_data:
            # Legacy support for older projects
            self.restore_thermal_loads(project_data['thermal_loads'])
        
        # Restore TopOpt options
        if project_data.get('topopt_options'):
            self.restore_topopt_options(project_data['topopt_options'])

        # Restore the optimization formulation and the body force (older projects: migrated, see restore_from_project)
        spec, body_force = restore_from_project(project_data)
        self.parent.optimization_spec = spec
        if body_force:
            self.parent.body_force = body_force
            self.parent.message_text.append(f"Body force restored: {body_force} m/s^2")
        if spec is not None and not project_data.get('optimization'):
            self.parent.message_text.append(
                "Note: this project's TopOpt performance limits (stress safety / max displacement / max temperature) "
                "were never applied by earlier versions. They are now constraint rows in TopOpt Execute, DISABLED "
                "until you review and enable them.")
        
        self.parent.message_text.append(f"Project loaded: {os.path.basename(filename)}")
        self.close()

    def load_geometry(self, file_path):
        """Load geometry using the same approach as GeometryWindow"""
        self.parent.plotter.clear_actors()
        self.parent.plotter.reset_camera()
        stl_geom = STLGeom(file_path)
        stl_geom.plotGeometry(show_edges=False, show_axes=True, show_bounding_box=False, plotter=self.parent.plotter,
                               color=MATERIAL_COLOR)
        
        area, volume, _, _ = stl_geom.compute_mass_properties()
        bounds = stl_geom.get_bounding_box()
        length_unit = self.parent.settings.get_length_unit_string()

        # Get material name or 'None'
        material_name = "None"
        if getattr(self.parent, "applied_material", None):
            material_name = self.parent.applied_material.get("name", "None")

        info_lines = [
            f"Model: {os.path.basename(file_path)}",
            f"Volume: {volume:.2e} {length_unit}³",
            f"Length: {bounds[1] - bounds[0]:.2e} {length_unit}" if bounds else "Length: N/A",
            f"Material: {material_name}"
        ]

        self.parent.plotter.remove_actor("geometry_info")
        self.parent.plotter.add_text("\n".join(info_lines), position="upper_left", font_size=12, 
                                   color="black", name="geometry_info", font="arial")

        self.parent.stl_geom = stl_geom
        self.parent.plotter.disable_picking()
        self.parent.plotter.enable_point_picking(callback=self.parent.on_left_button_press, use_picker=True, 
                                                picker='cell', show_message=False, left_clicking=True, show_point=False)
        self.parent.plotter.iren.add_observer("RightButtonPressEvent", self.parent.on_right_button_press)
        
        self.parent.update_LivVar('geometry_loaded', True)
        self.parent.set_sidebar_icon("Geometry", "check")
        self.parent.set_sidebar_icon("Material", "arrow")
        self.parent.message_text.append(f"Geometry loaded from project: {os.path.basename(file_path)}")

    def restore_structural_loads(self, structuralBC=None):
        """Restore structural loads and visualizations"""
        # Check if geometry is loaded
        if not self.parent.stl_geom:
            self.parent.message_text.append("Warning: Cannot restore structural loads - no geometry loaded.")
            return
        
        # Process structuralBC data if provided (new format)
        if structuralBC is not None:
            # Extract constraint data
            fixed_faces_indices = structuralBC.get('fixed_faces_indices', [])
            if fixed_faces_indices:
                # Create constraint data entry
                constraint_info = {
                    'type': 'Fixed XYZ',
                    'triangles': fixed_faces_indices
                }
                self.parent.constraint_data = [constraint_info]
                self.parent.constrained_triangles = set(fixed_faces_indices)
            
            # Extract force data
            load_faces_indices = structuralBC.get('load_faces_indices', [])
            load_forces = structuralBC.get('load_forces', [])
            
            self.parent.force_data = []
            for triangles, force in zip(load_faces_indices, load_forces):
                if len(force) >= 3:
                    force_info = {
                        'type': 'force_xyz',
                        'triangles': triangles,
                        'force_x': force[0],
                        'force_y': force[1],
                        'force_z': force[2]
                    }
                    self.parent.force_data.append(force_info)

            # Node (point) supports and loads
            for ps in structuralBC.get('point_supports', []):
                self.parent.constraint_data.append({'type': ps['type'], 'triangles': [], 'points': ps.get('points', []),
                                                    'regions': ps.get('regions', [])})
            for pl in structuralBC.get('point_loads', []):
                self.parent.force_data.append({'type': 'force_xyz', 'triangles': [], 'points': pl.get('points', []),
                                               'regions': pl.get('regions', []), 'force_x': pl['force'][0],
                                               'force_y': pl['force'][1], 'force_z': pl['force'][2]})
        
        # Create structural loads window if needed
        if not getattr(self.parent, 'structural_loads_window', None):
            self.parent.structural_loads_window = StructuralLoadsWindow(self.parent)
        
        # Recreate constraint visualizations (faces and point supports)
        if self.parent.constraint_data:
            self.parent.structural_loads_window.visualize_constraints()
        
        # Recreate force visualizations
        for force_info in self.parent.force_data:
            # Add triangle_data if not already present
            if force_info.get('points') or force_info.get('regions'):
                self.parent.structural_loads_window._rebuild_force_visualization(force_info)
                continue
            if 'triangle_data' not in force_info:
                force_info['triangle_data'] = self.recreate_triangle_data(force_info['triangles'])
                
            if force_info.get('type') == 'torque':
                # Visualize torque
                axis_point = force_info.get('axis_point', [0, 0, 0])
                direction = force_info.get('direction', [0, 0, 1])
                torque_value = force_info.get('torque', 0.0)
                surface_type = force_info.get('surface_type', 'PLANAR')
                cyl_radius = force_info.get('cyl_radius', None)
                self.parent.structural_loads_window.visualize_torque(
                    axis_point, direction, torque_value, surface_type, cyl_radius=cyl_radius
                )
            else:
                config = self.parent.structural_loads_window.LOAD_TYPES["Force"]
                for axis in ['X', 'Y', 'Z']:
                    force_value = force_info.get(f'force_{axis.lower()}', 0)
                    if force_value != 0:
                        direction = config["directions"][axis]
                        self.parent.structural_loads_window.visualize_force_arrows(
                            force_info['triangle_data'], force_value, direction, config["color"], axis)
        
        # Update structural loads state
        if self.parent.force_data:
            self.parent.update_LivVar('structural_loads.forces_applied', True)
        if self.parent.constraint_data:            # face and node supports
            self.parent.update_LivVar('structural_loads.fixed_constraints', True)
        
        if (self.parent.LivVar['structural_loads']['forces_applied'] and 
            self.parent.LivVar['structural_loads']['fixed_constraints']):
            self.parent.LivVar['structural_loads']['applied'] = True
            self.parent.set_sidebar_icon("Structural Loads", "check")
            self.parent.set_sidebar_icon("Analysis", "arrow")
            self.parent.set_sidebar_icon("TopOpt Options", "arrow")

    def restore_thermal_loads(self, thermal_data):
        """Restore thermal loads and visualizations from either format"""
        # Check if geometry is loaded
        if not self.parent.stl_geom:
            self.parent.message_text.append("Warning: Cannot restore thermal loads - no geometry loaded.")
            return
        
        # Create thermal loads window if needed
        if not getattr(self.parent, 'thermal_loads_window', None):
            self.parent.thermal_loads_window = ThermalLoadsWindow(self.parent)
        
        # Initialize thermal_loads structure if needed
        thermal_loads = {
            'fixed_temps': [],
            'heat_sources': [],
            'total_heat_sources': []
        }
        
        # Handle thermalBC format (new format)
        if isinstance(thermal_data, dict) and any(key in thermal_data for key in ['fixed_temps', 'heat_sources', 'total_heat_sources']):
            # Just copy the thermal data directly
            for key in ['fixed_temps', 'heat_sources', 'total_heat_sources']:
                if key in thermal_data:
                    thermal_loads[key] = thermal_data[key]
        
        # Store the thermal loads in the thermal_loads_window
        self.parent.thermal_loads_window.thermal_loads = thermal_loads
        
        # Recreate visualizations for each thermal load type
        thermal_types = {
            'fixed_temps': ('Temperature', 'temperature'),
            'heat_sources': ('Heat Flux', 'heat_flux'),
            'total_heat_sources': ('Total Heat', 'total_heat')
        }
        
        for load_key, (type_name, value_key) in thermal_types.items():
            for load in thermal_loads.get(load_key, []):
                # Add triangle_data if not already present
                if 'triangle_data' not in load:
                    load['triangle_data'] = self.recreate_triangle_data(load['triangles'])
                
                # Get thermal load configuration and create visualization
                if hasattr(self.parent.thermal_loads_window, 'THERMAL_TYPES'):
                    config = self.parent.thermal_loads_window.THERMAL_TYPES[type_name]
                    self.parent.thermal_loads_window.create_thermal_visualization(
                        load['triangle_data'], load[value_key], config)
        
        # Update thermal state
        if any(thermal_loads.get(key, []) for key in thermal_types.keys()):
            self.parent.update_LivVar('thermal_loads.applied', True)
            self.parent.set_sidebar_icon("Thermal Loads", "check")
            self.parent.set_sidebar_icon("Analysis", "arrow")
            self.parent.set_sidebar_icon("TopOpt Options", "arrow")

    def restore_topopt_options(self, options):
        """Restore TopOpt constraints and visualizations"""
        self.parent.topopt_options = options
        
        # Create TopOpt options window if needed
        if not getattr(self.parent, 'topopt_options_window', None):
            self.parent.topopt_options_window = TopOptOptionsWindow(self.parent)
        
        # Update widgets to match saved options
        for category, option_data in options.items():
            for option_key, option_value in option_data.items():
                if isinstance(option_value, bool):
                    widget_name = f'{option_key}_check'
                    if widget_name in self.parent.topopt_options_window.widgets:
                        self.parent.topopt_options_window.widgets[widget_name].setChecked(option_value)
                elif isinstance(option_value, dict):
                    check_widget = f'{option_key}_check'
                    if check_widget in self.parent.topopt_options_window.widgets:
                        self.parent.topopt_options_window.widgets[check_widget].setChecked(
                            option_value.get('enabled', False))
                    
                    if 'value' in option_value:
                        for widget_type in ['combo', 'spin']:
                            value_widget = f'{option_key}_{widget_type}'
                            if value_widget in self.parent.topopt_options_window.widgets:
                                widget = self.parent.topopt_options_window.widgets[value_widget]
                                if widget_type == 'combo':
                                    widget.setCurrentText(str(option_value['value']))
                                else:
                                    widget.setValue(option_value['value'])
                                break
        
        self.parent.topopt_options_window.update_visualizations()
        self.parent.update_LivVar('topopt.constraints_defined', True)
        self.parent.set_sidebar_icon("TopOpt Options", "check")
        self.parent.set_sidebar_icon("TopOpt Execute", "arrow")

    def recreate_triangle_data(self, triangle_indices):
        """Recreate triangle data structure from saved indices"""
        if not self.parent.stl_geom:
            self.parent.message_text.append("Warning: Cannot recreate triangle data - no geometry loaded.")
            return []
        
        triangle_data = []
        for tri_idx in triangle_indices:
            try:
                triangle_vertices = self.parent.stl_geom.mesh.vectors[tri_idx]
                center = np.mean(triangle_vertices, axis=0)
                v1, v2 = triangle_vertices[1] - triangle_vertices[0], triangle_vertices[2] - triangle_vertices[0]
                normal = np.cross(v1, v2)
                norm_length = np.linalg.norm(normal)
                if norm_length > 0:
                    normal = normal / norm_length
                
                triangle_data.append({
                    'index': tri_idx,
                    'center': center,
                    'normal': normal,
                    'vertices': triangle_vertices
                })
            except (IndexError, KeyError) as e:
                self.parent.message_text.append(f"Warning: Could not recreate triangle {tri_idx}: {e}")
                continue
        return triangle_data
#----------------------------------------------------------------------------
def main():
    """Start the PyTO GUI (used by `python -m pyto.gui.PyTOGUI` and by run_gui.py in the repository root)."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("* { font-size: 10pt; }")
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
