from __future__ import annotations

import fnmatch
import gzip
import json
import os
import posixpath
import tempfile
import time
import traceback
from pathlib import Path
import shutil
import h5py
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.figure import Figure
from qtpy.QtCore import QSettings, QSize, Qt, QModelIndex
from qtpy.QtGui import QAction, QColor, QFont, QFontDatabase, QIcon, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QStatusBar,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTableView,
    QToolBar,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from .hdf5_tree_model import HDF5TreeModel
from .syntax_highlighter import SyntaxHighlighter, get_language_from_path
from .utilities import (
    excluded_dirs,
    excluded_files,
    indices_to_ranges,
    ranges_to_indices,
    sanitize_hdf5_name,
    dataset_to_text,
)
from .csv_table_model import CSVTableModel
from .draggable_plot_list_widget import DraggablePlotListWidget
from .scaled_image_label import ScaledImageLabel
from .drop_tree_view import DropTreeView
from .column_statistics_dialog import ColumnStatisticsDialog
from .column_sort_dialog import ColumnSortDialog
from .column_filter_dialog import ColumnFilterDialog
from .column_visibility_dialog import ColumnVisibilityDialog
from .unique_values_dialog import UniqueValuesDialog
from .plot_options_dialog import PlotOptionsDialog


class HDF5Viewer(QMainWindow):
    """Main window for the HDF5 viewer application."""

    def __init__(self, parent=None):
        """Initialize the HDF5 viewer main window.

        Args:
            parent: Parent widget, or None for top-level window
        """
        super().__init__(parent)
        self.setWindowTitle("VibeHDF5")
        self.resize(900, 600)
        self._original_pixmap = None
        self._current_highlighter = None  # Track current syntax highlighter

        # Initialize QSettings for storing recent files
        self.settings = QSettings("VibeHDF5", "VibeHDF5")
        self.max_recent_files = 10
        self.recent_file_actions = []

        # Central widget: splitter with tree (left) and preview (right)
        central = QWidget(self)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central)

        splitter = QSplitter(self)
        splitter.setHandleWidth(8)  # Make handle wider for easier grabbing
        splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing completely
        splitter.splitterMoved.connect(self._on_splitter_moved)
        central_layout.addWidget(splitter)

        # Tree view + model (left)
        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Search bar for filtering tree items
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(4, 4, 4, 4)
        search_label = QLabel("Filter:")
        search_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Enter glob pattern (e.g., *.csv, data*, *test*)")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.textChanged.connect(self._on_search_text_changed)
        search_layout.addWidget(self.search_input)

        left_layout.addLayout(search_layout)

        # Create a vertical splitter for tree view and saved plots list
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setHandleWidth(8)
        left_splitter.setChildrenCollapsible(False)

        # Tree view widget
        tree_widget = QWidget()
        tree_layout = QVBoxLayout(tree_widget)
        tree_layout.setContentsMargins(0, 0, 0, 0)

        self.tree = DropTreeView(tree_widget, viewer=self)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setSelectionBehavior(QTreeView.SelectRows)
        self.tree.setHeaderHidden(False)
        tree_layout.addWidget(self.tree)

        left_splitter.addWidget(tree_widget)

        # Saved plots widget (below tree)
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)

        saved_plots_label = QLabel("Saved Plots:")
        saved_plots_label.setStyleSheet("font-weight: bold; padding: 4px;")
        plots_layout.addWidget(saved_plots_label)

        self.saved_plots_list = DraggablePlotListWidget(self)
        self.saved_plots_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.saved_plots_list.customContextMenuRequested.connect(self._on_saved_plots_context_menu)
        self.saved_plots_list.setToolTip(
            "Drag and drop to filesystem to export plot. Double-click to rename."
        )
        self.saved_plots_list.setEditTriggers(QListWidget.DoubleClicked)
        plots_layout.addWidget(self.saved_plots_list)

        # Buttons for plot management
        plot_buttons_layout = QHBoxLayout()
        self.btn_save_plot = QPushButton("Save Plot")
        self.btn_save_plot.setToolTip(
            "Save current table column selection as a named plot configuration"
        )
        self.btn_save_plot.clicked.connect(self._save_plot_config_dialog)
        self.btn_save_plot.setEnabled(False)
        plot_buttons_layout.addWidget(self.btn_save_plot)

        self.btn_edit_plot_options = QPushButton("Edit Options")
        self.btn_edit_plot_options.setToolTip(
            "Customize appearance, styling, and export settings for the selected plot"
        )
        self.btn_edit_plot_options.clicked.connect(self._edit_plot_options_dialog)
        self.btn_edit_plot_options.setEnabled(False)
        plot_buttons_layout.addWidget(self.btn_edit_plot_options)

        self.btn_delete_plot = QPushButton("Delete")
        self.btn_delete_plot.setToolTip("Delete the selected plot configuration permanently")
        self.btn_delete_plot.clicked.connect(self._delete_plot_config)
        self.btn_delete_plot.setEnabled(False)
        plot_buttons_layout.addWidget(self.btn_delete_plot)

        plots_layout.addLayout(plot_buttons_layout)

        left_splitter.addWidget(plots_widget)

        # Set initial sizes for the splitter (tree gets more space)
        left_splitter.setStretchFactor(0, 3)
        left_splitter.setStretchFactor(1, 1)

        left_layout.addWidget(left_splitter)

        splitter.addWidget(left)

        self.model = HDF5TreeModel(self)
        self.tree.setModel(self.model)
        self.tree.header().setStretchLastSection(True)
        self.tree.header().setDefaultSectionSize(350)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, Qt.AscendingOrder)

        # Enable editing via double-click for renaming
        self.tree.setEditTriggers(QAbstractItemView.DoubleClicked)

        # Enable drag-and-drop (both export and external import)
        self.tree.setDragEnabled(True)  # allow dragging out
        self.tree.setAcceptDrops(True)  # allow external drops
        self.tree.setDragDropMode(QAbstractItemView.DragDrop)
        self.tree.setDefaultDropAction(Qt.MoveAction)  # move for internal, copy for external

        # Context menu on tree
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.on_tree_context_menu)

        # Connect to model data changes to handle renames
        self.model.dataChanged.connect(self._on_tree_item_renamed)

        # Preview panel (right)
        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)

        # Create a vertical splitter for content and attributes
        right_splitter = QSplitter(Qt.Vertical, right)
        right_splitter.setHandleWidth(8)  # Make handle wider for easier grabbing
        right_splitter.setChildrenCollapsible(False)  # Prevent panels from collapsing completely
        right_splitter.splitterMoved.connect(self._on_splitter_moved)
        right_layout.addWidget(right_splitter)

        # Top section: main content preview
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_label = QLabel("No selection")
        self.preview_edit = QPlainTextEdit(self)
        self.preview_edit.setReadOnly(True)
        # Use a fixed-width font for better alignment of file contents
        try:
            fixed = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        except Exception:
            fixed = QFont("Courier New")
        self.preview_edit.setFont(fixed)
        # Avoid wrapping so columns/bytes stay aligned
        try:
            self.preview_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        except Exception:
            pass
        content_layout.addWidget(self.preview_label)
        content_layout.addWidget(self.preview_edit)

        # hide this so only one is visible when application starts up
        self.preview_edit.setVisible(False)

        # Filter panel for CSV tables (hidden by default)
        self.filter_panel = QWidget()
        filter_panel_layout = QHBoxLayout(self.filter_panel)
        filter_panel_layout.setContentsMargins(5, 5, 5, 5)

        filter_label = QLabel("Filters:")
        filter_panel_layout.addWidget(filter_label)

        self.filter_status_label = QLabel("No filters applied")
        filter_panel_layout.addWidget(self.filter_status_label)

        self.btn_configure_filters = QPushButton("Configure Filters...")
        self.btn_configure_filters.setToolTip(
            "Add or modify filter conditions to show only specific rows (filters are saved with the file)"
        )
        self.btn_configure_filters.clicked.connect(self._configure_filters_dialog)
        filter_panel_layout.addWidget(self.btn_configure_filters)

        self.btn_clear_filters = QPushButton("Clear Filters")
        self.btn_clear_filters.setToolTip("Remove all active filters and show all rows")
        self.btn_clear_filters.clicked.connect(self._clear_filters)
        self.btn_clear_filters.setEnabled(False)
        filter_panel_layout.addWidget(self.btn_clear_filters)

        self.btn_show_statistics = QPushButton("Statistics...")
        self.btn_show_statistics.setToolTip(
            "View statistical summaries (min, max, mean, median, etc.) for each column using filtered data"
        )
        self.btn_show_statistics.clicked.connect(self._show_statistics_dialog)
        filter_panel_layout.addWidget(self.btn_show_statistics)

        self.btn_configure_sort = QPushButton("Sort...")
        self.btn_configure_sort.setToolTip(
            "Configure multi-column sorting with ascending/descending order (sort settings are saved with the file)"
        )
        self.btn_configure_sort.clicked.connect(self._configure_sort_dialog)
        filter_panel_layout.addWidget(self.btn_configure_sort)

        self.btn_clear_sort = QPushButton("Clear Sort")
        self.btn_clear_sort.setToolTip("Remove all sorting and display rows in original order")
        self.btn_clear_sort.clicked.connect(self._clear_sort)
        self.btn_clear_sort.setEnabled(False)
        filter_panel_layout.addWidget(self.btn_clear_sort)

        self.btn_configure_columns = QPushButton("Columns...")
        self.btn_configure_columns.setToolTip("Select which columns to display in the table")
        self.btn_configure_columns.clicked.connect(self._configure_columns_dialog)
        filter_panel_layout.addWidget(self.btn_configure_columns)

        filter_panel_layout.addStretch()

        self.filter_panel.setVisible(False)
        content_layout.addWidget(self.filter_panel)

        # Table widget for CSV/tabular data (hidden by default)
        # Replace QTableWidget with QTableView for CSV preview
        self.preview_table = QTableView()
        self.preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.preview_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.preview_table.setSortingEnabled(False)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setContextMenuPolicy(Qt.CustomContextMenu)
        # self.preview_table.customContextMenuRequested.connect(self._on_table_context_menu)
        content_layout.addWidget(self.preview_table)

        # Lazy loading state variables
        self._table_batch_size = 1000  # Load rows in batches
        self._table_loaded_rows = 0  # Track how many rows are loaded
        self._table_is_loading = False  # Prevent concurrent loads

        # Image preview label (hidden by default)
        self.preview_image = ScaledImageLabel(self, rescale_callback=self._show_scaled_image)
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setVisible(False)
        self.preview_image.setScaledContents(False)  # We'll scale manually for aspect ratio
        content_layout.addWidget(self.preview_image)

        right_splitter.addWidget(content_widget)

        # Bottom section: tabbed widget for Attributes and Plot
        self.bottom_tabs = QTabWidget()

        # Attributes tab
        attrs_widget = QWidget()
        attrs_layout = QVBoxLayout(attrs_widget)
        attrs_layout.setContentsMargins(0, 0, 0, 0)

        self.attrs_label = QLabel("Attributes")
        self.attrs_label.setVisible(False)
        self.attrs_table = QTableWidget(self)
        self.attrs_table.setVisible(True)  # Always visible when tab is active
        self.attrs_table.setColumnCount(2)
        self.attrs_table.setHorizontalHeaderLabels(["Name", "Value"])
        self.attrs_table.horizontalHeader().setStretchLastSection(True)
        self.attrs_table.setAlternatingRowColors(True)
        self.attrs_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.attrs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        # Enable context menu for attributes table
        self.attrs_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.attrs_table.customContextMenuRequested.connect(self._on_attrs_context_menu)
        attrs_layout.addWidget(self.attrs_table)

        self.bottom_tabs.addTab(attrs_widget, "Attributes")

        # Plot tab
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.plot_figure = Figure(figsize=(8, 6))
        self.plot_canvas = FigureCanvas(self.plot_figure)
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, plot_widget)

        # Add toolbar first, then canvas
        plot_layout.addWidget(self.plot_toolbar)
        plot_layout.addWidget(self.plot_canvas)

        self.bottom_tabs.addTab(plot_widget, "Plot")

        # Start with Attributes tab visible
        self.bottom_tabs.setCurrentIndex(0)

        right_splitter.addWidget(self.bottom_tabs)

        # Set initial sizes: main content gets most of the space
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # Tool bar / actions
        self._create_actions()
        self._create_toolbar()
        self._create_menu_bar()
        self.setStatusBar(QStatusBar(self))

        # Add permanent file size label to right side of status bar
        self.file_size_label = QLabel()
        self.file_size_label.setStyleSheet("QLabel { padding: 0 5px; }")
        self.statusBar().addPermanentWidget(self.file_size_label)

        self.tree.selectionModel().selectionChanged.connect(self.on_selection_changed)

        # Track currently previewed CSV group (for plotting)
        self._current_csv_group_path: str | None = None

        # Track CSV data and filters
        self._csv_data_dict: dict[str, np.ndarray] = {}  # Cached column data (lazy loaded)
        self._csv_dataset_info: dict = {}  # Metadata for lazy loading (column -> (ds_key, th_grp_path, row_count, dtype))
        self._csv_column_names: list[str] = []
        self._csv_filters: list[tuple[str, str, str]] = []  # (column, operator, value)
        self._csv_filtered_indices: np.ndarray | None = None  # Indices of visible rows
        self._csv_visible_columns: list[str] = []  # Columns to show in table

        # Track saved plot configurations for current CSV group
        self._saved_plots: list[dict] = []  # List of plot config dictionaries

        # Connect saved plots list selection changed
        self.saved_plots_list.itemSelectionChanged.connect(self._on_saved_plot_selection_changed)
        self.saved_plots_list.itemClicked.connect(self._on_saved_plot_clicked)
        self.saved_plots_list.itemChanged.connect(self._on_plot_item_renamed)

        # Track current search pattern
        self._search_pattern: str = ""

    def _is_csv_group(self, path: str) -> bool:
        """Return True if the group at the given path is a CSV-derived group."""
        # This checks for the presence of 'column_names' attribute in the group
        if not self.model or not self.model.filepath or not path:
            return False
        try:
            with h5py.File(self.model.filepath, "r") as h5:
                if path in h5:
                    grp = h5[path]
                    return isinstance(grp, h5py.Group) and "column_names" in grp.attrs
        except Exception:
            pass
        return False

    def _get_filtered_sorted_indices(self, data_dict, filters, sort_specs):
        """Return filtered and sorted row indices for given data, filters, and sort specs."""
        if not data_dict:
            return np.array([], dtype=int), 0, 0
        max_rows = max(len(data_dict[col]) for col in data_dict)
        if max_rows == 0:
            return np.array([], dtype=int), 0, 0
        valid_rows = np.ones(max_rows, dtype=bool)
        # Filtering
        if filters:
            for col_name, operator, value_str in filters:
                if col_name not in data_dict:
                    continue
                col_data = data_dict[col_name]
                try:
                    if operator in ("=", "==", "!=", "<", "<=", ">", ">="):
                        mask = self._evaluate_filter(col_data, operator, value_str)
                    elif operator == "contains":
                        mask = (
                            np.char.find(np.char.lower(col_data.astype(str)), value_str.lower())
                            >= 0
                        )
                    elif operator == "starts with":
                        mask = np.char.startswith(
                            np.char.lower(col_data.astype(str)), value_str.lower()
                        )
                    elif operator == "ends with":
                        mask = np.char.endswith(
                            np.char.lower(col_data.astype(str)), value_str.lower()
                        )
                    else:
                        mask = np.ones_like(valid_rows, dtype=bool)
                except Exception:
                    mask = np.zeros_like(valid_rows, dtype=bool)
                # Ensure mask is same length as valid_rows
                if len(mask) != len(valid_rows):
                    mask = np.resize(mask, len(valid_rows))
                valid_rows &= mask
        filtered_indices = np.where(valid_rows)[0]
        # Sorting
        if sort_specs and len(filtered_indices) > 0:
            sort_columns = []
            sort_orders = []
            for col_name, ascending in sort_specs:
                if col_name in data_dict:
                    sort_columns.append(col_name)
                    sort_orders.append(ascending)
            if sort_columns:
                try:
                    sort_data = {}
                    for col_name in sort_columns:
                        col_data = data_dict[col_name][filtered_indices]
                        sort_data[col_name] = col_data
                    df = pd.DataFrame(sort_data)
                    df_sorted = df.sort_values(
                        by=sort_columns, ascending=sort_orders, na_position="last"
                    )
                    # Map sorted indices back to original data indices
                    filtered_indices = filtered_indices[df_sorted.index.values]
                except Exception as e:
                    print(f"Warning: Could not sort data: {e}")
        if len(filtered_indices) > 0:
            start_row = int(filtered_indices[0])
            end_row = int(filtered_indices[-1])
        else:
            start_row = 0
            end_row = 0
        return filtered_indices, start_row, end_row

    def _create_progress_dialog(
        self, title: str, max_value: int = 100, min_duration: int = 500
    ) -> QProgressDialog:
        """Create a standard progress dialog with consistent settings.

        Args:
            title: Dialog title/label text
            max_value: Maximum progress value (default 100)
            min_duration: Minimum duration in ms before showing (default 500)

        Returns:
            Configured QProgressDialog instance
        """
        progress = QProgressDialog(title, "Cancel", 0, max_value, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(min_duration)
        progress.setValue(0)
        QApplication.processEvents()
        return progress

    def _save_csv_attr_to_hdf5(
        self, attr_name: str, value, success_msg: str, clear_msg: str = None
    ):
        """Generic helper to save a CSV group attribute to HDF5.

        Args:
            attr_name: Name of the attribute to save
            value: Value to save (will be JSON encoded if not empty)
            success_msg: Status bar message on successful save
            clear_msg: Status bar message when clearing attribute (optional)
        """
        if not self._current_csv_group_path or not self.model or not self.model.filepath:
            return

        try:
            with h5py.File(self.model.filepath, "r+") as h5:
                if self._current_csv_group_path in h5:
                    grp = h5[self._current_csv_group_path]
                    if isinstance(grp, h5py.Group):
                        if value:
                            # Save the value as JSON
                            json_str = json.dumps(value)
                            grp.attrs[attr_name] = json_str
                            self.statusBar().showMessage(success_msg, 3000)
                        else:
                            # Remove attribute if value is empty
                            if attr_name in grp.attrs:
                                del grp.attrs[attr_name]
                            if clear_msg:
                                self.statusBar().showMessage(clear_msg, 3000)

            # Update file size display after modification
            self._update_file_size_display()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not save {attr_name} to HDF5: {exc}")

    def _load_csv_attr_from_hdf5(self, grp: h5py.Group, attr_name: str, validator=None):
        """Generic helper to load a CSV group attribute from HDF5.

        Args:
            grp: HDF5 group to load from
            attr_name: Name of the attribute to load
            validator: Optional function to validate/transform loaded value

        Returns:
            Loaded value or None if not found/invalid
        """
        try:
            if attr_name in grp.attrs:
                json_str = grp.attrs[attr_name]
                if isinstance(json_str, bytes):
                    json_str = json_str.decode("utf-8")
                value = json.loads(json_str)
                if validator:
                    return validator(value)
                return value
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not load {attr_name} from HDF5: {exc}")
        return None

    def _apply_plot_style(self, fig_or_ax, ax, use_dark: bool) -> None:
        """Apply dark or light background styling to a plot figure and axes.

        Args:
            fig_or_ax: Figure object (for setting figure background)
            ax: Axes object (for setting axes colors)
            use_dark: True for dark background, False for light background
        """
        if use_dark:
            fig_or_ax.set_facecolor("#1e1e1e")
            ax.set_facecolor("#2e2e2e")
            color = "white"
        else:
            fig_or_ax.set_facecolor("white")
            ax.set_facecolor("white")
            color = "black"

        # Set spine colors
        for spine in ax.spines.values():
            spine.set_color(color)

        # Set label and tick colors
        ax.xaxis.label.set_color(color)
        ax.yaxis.label.set_color(color)
        ax.tick_params(axis="x", colors=color)
        ax.tick_params(axis="y", colors=color)

    def _parse_datetime_column(
        self, x_arr: np.ndarray, min_len: int, datetime_format: str = ""
    ) -> tuple[np.ndarray, bool]:
        """Parse a column as datetime and convert to matplotlib date numbers.

        Args:
            x_arr: Array of values to parse as dates
            min_len: Maximum length to process
            datetime_format: Optional strptime format string

        Returns:
            Tuple of (numeric_array, success_flag)
        """
        try:
            if datetime_format:
                x_data = pd.to_datetime(
                    pd.Series(x_arr[:min_len]), format=datetime_format, errors="coerce"
                )
            else:
                x_data = pd.to_datetime(pd.Series(x_arr[:min_len]), errors="coerce")

            valid_dates = x_data.notna()
            if valid_dates.sum() > 0:
                x_num = np.array([mdates.date2num(d) if pd.notna(d) else np.nan for d in x_data])
                return x_num, True
        except Exception:
            pass

        return np.arange(min_len, dtype=float), False

    def _make_legend_interactive(self, ax: plt.Axes, legend) -> None:
        """Make the legend interactive so clicking on labels toggles line visibility.

        Args:
            ax: Matplotlib axes object
            legend: Legend object to make interactive
        """
        if not legend:
            return

        # Get the figure from the axes
        fig = ax.get_figure()
        if not fig:
            return

        # Map legend lines to plot lines
        line_dict = {}
        for legend_line, orig_line in zip(legend.get_lines(), ax.get_lines()):
            legend_line.set_picker(True)
            legend_line.set_pickradius(5)
            line_dict[legend_line] = orig_line

        # Connect pick event
        def on_pick(event):
            legend_line = event.artist
            if legend_line not in line_dict:
                return

            orig_line = line_dict[legend_line]
            visible = not orig_line.get_visible()
            orig_line.set_visible(visible)

            # Update legend line appearance
            if visible:
                legend_line.set_alpha(1.0)
            else:
                legend_line.set_alpha(0.2)

            # Use the figure's canvas to redraw
            if fig.canvas:
                fig.canvas.draw_idle()

        # Store the connection ID to avoid multiple connections
        if not hasattr(self, "_legend_pick_cid"):
            self._legend_pick_cid = None

        # Disconnect previous connection if exists
        if self._legend_pick_cid is not None:
            try:
                self.plot_figure.canvas.mpl_disconnect(self._legend_pick_cid)
            except Exception:
                pass

        # Connect new pick event to the current figure
        self._legend_pick_cid = fig.canvas.mpl_connect("pick_event", on_pick)

    def _capture_plot_visibility_state(self) -> dict[str, bool]:
        """Capture current visibility state of all plot lines.

        Returns:
            Dictionary mapping series label to visibility state
        """
        series_visibility = {}
        if hasattr(self, "plot_figure") and self.plot_figure.axes:
            ax = self.plot_figure.axes[0]
            for line in ax.get_lines():
                label = line.get_label()
                if label and not label.startswith("_"):  # Ignore internal matplotlib labels
                    series_visibility[label] = line.get_visible()
        return series_visibility

    def _apply_plot_visibility_state(
        self, ax: plt.Axes, y_names: list[str], series_visibility: dict[str, bool]
    ) -> None:
        """Apply visibility state to plot lines and update legend appearance.

        Args:
            ax: Matplotlib axes object
            y_names: List of Y column names
            series_visibility: Dictionary mapping series label to visibility state
        """
        # Apply visibility to plot lines
        for line in ax.get_lines():
            label = line.get_label()
            if label in series_visibility:
                line.set_visible(series_visibility[label])

        # Update legend appearance to reflect hidden series
        if series_visibility and ax.get_legend():
            legend = ax.get_legend()
            for legend_line, orig_line in zip(legend.get_lines(), ax.get_lines()):
                if not orig_line.get_visible():
                    legend_line.set_alpha(0.2)

    def _process_x_axis_data(
        self, x_idx: int | None, col_data: dict, y_names: list[str], x_name: str, plot_options: dict
    ) -> tuple[np.ndarray, np.ndarray, bool, bool, int]:
        """Process X-axis data for plotting, handling single-column, datetime, and string data.

        Args:
            x_idx: Column index for X axis, or None for point count
            col_data: Dictionary of column name -> array data
            y_names: List of Y column names
            x_name: X column name (or "Point" if x_idx is None)
            plot_options: Dictionary of plot options

        Returns:
            Tuple of (x_arr, x_num, x_is_string, xaxis_datetime, min_len)
        """
        # Calculate minimum data length
        min_len = min(len(col_data.get(n, [])) for n in y_names if n in col_data)

        # Handle single-column mode (x_idx is None)
        if x_idx is None:
            x_arr = np.arange(min_len)
            x_num = x_arr.astype(float)
            return x_arr, x_num, False, False, min_len

        # Multi-column mode - process X data
        x_arr = col_data[x_name].ravel()
        min_len = min(len(x_arr), min_len)

        xaxis_datetime = plot_options.get("xaxis_datetime", False)
        datetime_format = plot_options.get("datetime_format", "").strip()

        # Check if x_arr contains strings
        x_is_string = False
        if len(x_arr) > 0:
            first_val = x_arr[0]
            x_is_string = isinstance(first_val, str) or (
                hasattr(first_val, "dtype") and first_val.dtype.kind in ("U", "O")
            )

        # Parse X data based on type
        if x_is_string:
            # Try datetime parsing with various strategies
            if xaxis_datetime and not datetime_format:
                # Auto-detect datetime format
                x_num, success = self._parse_datetime_column(x_arr, min_len)
                if success:
                    xaxis_datetime = True
                else:
                    x_num = np.arange(min_len, dtype=float)
                    xaxis_datetime = False
            elif not xaxis_datetime:
                # Auto-detect datetime even if not explicitly requested
                x_num, success = self._parse_datetime_column(x_arr, min_len)
                xaxis_datetime = success
            elif xaxis_datetime and datetime_format:
                # Use specified format
                x_num, success = self._parse_datetime_column(x_arr, min_len, datetime_format)
                if not success:
                    # Fall back to numeric conversion
                    x_num = (
                        pd.to_numeric(pd.Series(x_arr[:min_len]), errors="coerce")
                        .astype(float)
                        .to_numpy()
                    )
                    xaxis_datetime = False
            else:
                x_num = np.arange(min_len, dtype=float)
        else:
            # Non-string data - convert to numeric
            x_num = (
                pd.to_numeric(pd.Series(x_arr[:min_len]), errors="coerce").astype(float).to_numpy()
            )
            xaxis_datetime = False

        return x_arr, x_num, x_is_string, xaxis_datetime, min_len

    def _format_xaxis(
        self,
        ax,
        fig,
        xaxis_datetime: bool,
        x_is_string: bool,
        x_arr: np.ndarray,
        min_len: int,
        plot_options: dict | None = None,
    ) -> None:
        """Format x-axis for datetime or categorical string data.

        Args:
            ax: Matplotlib axes object
            fig: Matplotlib figure object (for autofmt_xdate)
            xaxis_datetime: True if x-axis contains datetime data
            x_is_string: True if x-axis contains string data
            x_arr: Original x-axis array (for string labels)
            min_len: Length of data
            plot_options: Optional plot options dict (for custom datetime format)
        """

        if xaxis_datetime:
            # Format datetime x-axis
            if plot_options:
                datetime_display_format = plot_options.get("datetime_display_format", "").strip()
                if datetime_display_format:
                    ax.xaxis.set_major_formatter(DateFormatter(datetime_display_format))
                else:
                    ax.xaxis.set_major_locator(AutoDateLocator())
            else:
                ax.xaxis.set_major_locator(AutoDateLocator())
            # Rotate labels for better readability
            fig.autofmt_xdate()
        elif x_is_string and not xaxis_datetime:
            # X-axis is categorical strings - set string labels on integer positions
            num_points = min(min_len, len(x_arr))
            if num_points <= 50:
                # Show all labels if not too many
                ax.set_xticks(np.arange(num_points))
                ax.set_xticklabels(x_arr[:num_points], rotation=45, ha="right")
            else:
                # Show subset of labels to avoid overcrowding
                step = max(1, num_points // 20)  # Show ~20 labels
                indices = np.arange(0, num_points, step)
                ax.set_xticks(indices)
                ax.set_xticklabels([x_arr[i] for i in indices], rotation=45, ha="right")

    def _apply_axis_limits(self, ax, plot_options: dict) -> None:
        """Apply axis limit settings from plot options.

        Args:
            ax: Matplotlib axes object
            plot_options: Plot options dict containing xlim/ylim settings
        """
        # Apply x-axis limits
        xlim_min = plot_options.get("xlim_min")
        xlim_max = plot_options.get("xlim_max")
        if xlim_min is not None or xlim_max is not None:
            current_xlim = ax.get_xlim()
            new_xlim = (
                xlim_min if xlim_min is not None else current_xlim[0],
                xlim_max if xlim_max is not None else current_xlim[1],
            )
            ax.set_xlim(new_xlim)

        # Apply y-axis limits
        ylim_min = plot_options.get("ylim_min")
        ylim_max = plot_options.get("ylim_max")
        if ylim_min is not None or ylim_max is not None:
            current_ylim = ax.get_ylim()
            new_ylim = (
                ylim_min if ylim_min is not None else current_ylim[0],
                ylim_max if ylim_max is not None else current_ylim[1],
            )
            ax.set_ylim(new_ylim)

    def _plot_series_with_options(
        self,
        ax: plt.Axes,
        x_num: np.ndarray,
        y_num: np.ndarray,
        valid: np.ndarray,
        y_name: str,
        series_opts: dict,
        any_plotted: bool,
    ) -> bool:
        """Plot a single series with smoothing and trendline options applied.

        Args:
            ax: Matplotlib axes to plot on
            x_num: X-axis data (numeric)
            y_num: Y-axis data (numeric)
            valid: Boolean mask for valid data points
            y_name: Name of the series
            series_opts: Dictionary of series options (color, smoothing, trendlines, etc.)
            any_plotted: Whether any data has been plotted yet

        Returns:
            Updated any_plotted flag
        """
        label = series_opts.get("label", "").strip() or y_name
        plot_type = series_opts.get("plot_type", "line")

        # Smoothing logic
        apply_smooth = series_opts.get("smooth", False)
        smooth_mode = series_opts.get("smooth_mode", "smoothed")
        smooth_window = series_opts.get("smooth_window", 5)

        # Plot original if requested
        if not apply_smooth or smooth_mode in ("original", "both"):
            plot_kwargs = {
                "label": label
                if not (apply_smooth and smooth_mode == "both")
                else f"{label} (original)"
            }
            if "color" in series_opts and series_opts["color"]:
                plot_kwargs["color"] = series_opts["color"]
            if "linestyle" in series_opts and series_opts["linestyle"]:
                plot_kwargs["linestyle"] = series_opts["linestyle"]
            if "marker" in series_opts and series_opts["marker"]:
                plot_kwargs["marker"] = series_opts["marker"]
            if "linewidth" in series_opts:
                plot_kwargs["linewidth"] = series_opts["linewidth"]
            if "markersize" in series_opts:
                plot_kwargs["markersize"] = series_opts["markersize"]

            if apply_smooth and smooth_mode == "both":
                plot_kwargs["alpha"] = 0.3
                plot_kwargs["linewidth"] = plot_kwargs.get("linewidth", 1.5) * 0.7

            # Use bar or plot depending on plot_type
            if plot_type == "bar":
                # For bar charts, remove line/marker specific options
                bar_kwargs = {"label": plot_kwargs.get("label")}
                if "color" in plot_kwargs:
                    bar_kwargs["color"] = plot_kwargs["color"]
                if "alpha" in plot_kwargs:
                    bar_kwargs["alpha"] = plot_kwargs["alpha"]
                ax.bar(x_num[valid], y_num[valid], **bar_kwargs)
            else:
                ax.plot(x_num[valid], y_num[valid], **plot_kwargs)
            any_plotted = True

        # Plot smoothed if requested
        if apply_smooth and smooth_mode in ("smoothed", "both"):
            try:
                window = max(2, int(smooth_window))
                y_series = pd.Series(y_num[valid])
                y_smooth = (
                    y_series.rolling(window=window, center=True, min_periods=1).mean().to_numpy()
                )

                smooth_kwargs = {
                    "label": f"{label} (MA-{window})" if smooth_mode == "both" else label
                }
                if "color" in series_opts and series_opts["color"]:
                    smooth_kwargs["color"] = series_opts["color"]
                if "linestyle" in series_opts and series_opts["linestyle"]:
                    smooth_kwargs["linestyle"] = series_opts["linestyle"]
                else:
                    smooth_kwargs["linestyle"] = "-"
                if "linewidth" in series_opts:
                    smooth_kwargs["linewidth"] = series_opts["linewidth"]
                else:
                    smooth_kwargs["linewidth"] = 2.0

                # Use bar or plot depending on plot_type
                if plot_type == "bar":
                    bar_kwargs = {"label": smooth_kwargs.get("label")}
                    if "color" in smooth_kwargs:
                        bar_kwargs["color"] = smooth_kwargs["color"]
                    ax.bar(x_num[valid], y_smooth, **bar_kwargs)
                else:
                    ax.plot(x_num[valid], y_smooth, **smooth_kwargs)
                any_plotted = True
            except Exception:
                # Smoothing failed, already plotted original if needed
                pass

        # Plot trend line if requested
        apply_trend = series_opts.get("trendline", False)
        if apply_trend:
            try:
                trend_type = series_opts.get("trendline_type", "linear")

                # Calculate trend line using numpy polyfit
                if trend_type == "linear":
                    degree = 1
                elif trend_type == "poly2":
                    degree = 2
                elif trend_type == "poly3":
                    degree = 3
                elif trend_type == "poly4":
                    degree = 4
                else:
                    degree = 1

                # Fit polynomial to the data
                coeffs = np.polyfit(x_num[valid], y_num[valid], degree)
                poly = np.poly1d(coeffs)
                y_trend = poly(x_num[valid])

                # Prepare trend line label
                if degree == 1:
                    trend_label = f"{label} (linear trend)"
                else:
                    trend_label = f"{label} (poly{degree} trend)"

                # Plot trend line
                trend_kwargs = {"label": trend_label}
                if "color" in series_opts and series_opts["color"]:
                    trend_kwargs["color"] = series_opts["color"]
                trend_kwargs["linestyle"] = "--"
                trend_kwargs["linewidth"] = 2.0
                trend_kwargs["alpha"] = 0.8

                ax.plot(x_num[valid], y_trend, **trend_kwargs)
                any_plotted = True
            except Exception:
                # Trend line calculation failed, silently continue
                pass

        return any_plotted

    def _apply_plot_labels_and_formatting(
        self,
        ax: plt.Axes,
        fig,
        x_name: str,
        y_names: list,
        plot_config: dict,
        plot_options: dict,
        use_dark: bool,
        interactive_legend: bool = True,
    ) -> None:
        """Apply labels, fonts, grid, legend, log scale, and reference lines to a plot.

        Args:
            ax: Matplotlib axes object
            fig: Matplotlib figure object
            x_name: Name of x-axis data
            y_names: List of y-axis data names
            plot_config: Plot configuration dictionary
            plot_options: Plot options dictionary
            use_dark: Whether dark background is enabled
            interactive_legend: Whether to make legend interactive (only for UI plots)
        """

        plot_type = plot_options.get("type", "line")

        # Apply custom labels or use defaults
        xlabel = plot_options.get("xlabel", "").strip() or x_name
        if plot_type == "contourf":
            # for these, there is only one y label [z is the color value]
            ylabel = plot_options.get("ylabel", "").strip() or y_names[0].strip()
        else:
            ylabel = plot_options.get("ylabel", "").strip() or ", ".join(y_names)
        custom_title = plot_options.get("title", "").strip()
        title_text = custom_title if custom_title else plot_config.get("name", "Plot")

        # Apply font family if specified
        font_family = plot_options.get("font_family", "serif")

        # Apply labels with font sizes and family
        title_obj = ax.set_title(
            title_text, fontsize=plot_options.get("title_fontsize", 12), family=font_family
        )
        title_obj.set_color("white" if use_dark else "black")
        ax.set_xlabel(
            xlabel, fontsize=plot_options.get("axis_label_fontsize", 10), family=font_family
        )
        ax.set_ylabel(
            ylabel, fontsize=plot_options.get("axis_label_fontsize", 10), family=font_family
        )
        ax.tick_params(axis="both", which="major", labelsize=plot_options.get("tick_fontsize", 9))

        # Apply font family to tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)

        # Apply grid and legend
        grid_alpha = 0.3 if plot_options.get("grid", True) else None
        if grid_alpha:
            ax.grid(True, alpha=grid_alpha)
        if plot_options.get("legend", True) and plot_type != "contourf":
            # for now, contour plots don't have a legend
            legend_loc = plot_options.get("legend_loc", "best")
            legend = ax.legend(fontsize=plot_options.get("legend_fontsize", 9), loc=legend_loc)
            # Apply font family to legend text
            for text in legend.get_texts():
                text.set_fontfamily(font_family)
            # Make legend interactive (only for UI plots, not exports)
            if interactive_legend:
                self._make_legend_interactive(ax, legend)

        # Apply log scale
        if plot_options.get("xlog", False):
            ax.set_xscale("log")
        if plot_options.get("ylog", False):
            ax.set_yscale("log")

        # Reference lines
        ref_lines = plot_options.get("reference_lines", [])
        for refline in ref_lines:
            try:
                line_type = refline.get("type")
                value = refline.get("value")
                if line_type == "horizontal" and value is not None:
                    ax.axhline(
                        y=value,
                        color=refline.get("color", "red"),
                        linestyle=refline.get("linestyle", "--"),
                        linewidth=refline.get("linewidth", 1.0),
                        label=refline.get("label"),
                    )
                elif line_type == "vertical" and value is not None:
                    ax.axvline(
                        x=value,
                        color=refline.get("color", "red"),
                        linestyle=refline.get("linestyle", "--"),
                        linewidth=refline.get("linewidth", 1.0),
                        label=refline.get("label"),
                    )
            except Exception:
                pass

        fig.tight_layout()

        # Apply colorbar font and label styling for contourf plots
        if plot_options.get("type", "line") == "contourf":
            # if the figure has a colorbar attached:
            if self.cbar:
                cax = self.cbar.ax
                font_family = plot_options.get("font_family", "serif")
                font_size = plot_options.get("axis_label_fontsize", 10)
                tick_font_size = plot_options.get("tick_fontsize", 10)
                font_color = "white" if use_dark else "black"
                # Set colorbar label font
                cax.yaxis.label.set_fontfamily(font_family)
                cax.yaxis.label.set_fontsize(font_size)
                cax.yaxis.label.set_color(font_color)
                # Set colorbar tick font
                for tick in cax.get_yticklabels():
                    tick.set_fontfamily(font_family)
                    tick.set_fontsize(tick_font_size)
                    tick.set_color(font_color)

    def _create_actions(self) -> None:
        """Create all QAction objects for menu and toolbar items."""
        # Get standard icon theme
        style = self.style()

        # Create a custom icon with "H5" text on file icon for HDF5-specific actions
        def create_h5_file_icon():
            """Create an icon with 'H5' drawn using lines on a standard file icon."""
            from qtpy.QtCore import QPoint, QRect
            from qtpy.QtGui import QPainter, QPen

            base_icon = style.standardIcon(QStyle.SP_FileIcon)
            pixmap = base_icon.pixmap(48, 48)

            painter = QPainter(pixmap)
            # painter.setRenderHint(QPainter.Antialiasing)

            # Calculate center position - adjust to be more centered on the icon
            center_x = pixmap.width() // 2 - 24  # Shift slightly left
            center_y = pixmap.height() // 2 - 15  # Shift slightly up

            # Draw white background rectangle for visibility
            bg_rect = QRect(center_x - 14, center_y - 10, 28, 20)
            painter.fillRect(bg_rect, QColor(255, 255, 255, 220))

            # Set up pen for drawing lines
            pen = QPen(QColor(0, 100, 200), 3)  # Blue, 3px wide
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)

            # Draw "H" using lines (left vertical, horizontal bar, right vertical)
            h_left_x = center_x - 12
            h_right_x = center_x - 3
            h_top_y = center_y - 8
            h_bottom_y = center_y + 8
            h_mid_y = center_y

            # H - left vertical line
            painter.drawLine(h_left_x, h_top_y, h_left_x, h_bottom_y)
            # H - horizontal bar
            painter.drawLine(h_left_x, h_mid_y, h_right_x, h_mid_y)
            # H - right vertical line
            painter.drawLine(h_right_x, h_top_y, h_right_x, h_bottom_y)

            # Draw "5" using lines (top, curve, bottom)
            five_left_x = center_x + 3
            five_right_x = center_x + 12
            five_top_y = center_y - 8
            five_mid_y = center_y - 1
            five_bottom_y = center_y + 8

            # 5 - top horizontal line
            painter.drawLine(five_left_x, five_top_y, five_right_x, five_top_y)
            # 5 - left vertical line (top to middle)
            painter.drawLine(five_left_x, five_top_y, five_left_x, five_mid_y)
            # 5 - middle horizontal line
            painter.drawLine(five_left_x, five_mid_y, five_right_x, five_mid_y)
            # 5 - right vertical line (middle to bottom)
            painter.drawLine(five_right_x, five_mid_y, five_right_x, five_bottom_y)
            # 5 - bottom horizontal line
            painter.drawLine(five_left_x, five_bottom_y, five_right_x, five_bottom_y)

            painter.end()
            return QIcon(pixmap)

        self.act_new = QAction("New HDF5 File…", self)
        self.act_new.setIcon(style.standardIcon(QStyle.SP_FileIcon))
        self.act_new.setShortcut("Ctrl+N")
        self.act_new.setToolTip("Create a new empty HDF5 file (Ctrl+N)")
        self.act_new.triggered.connect(self.new_file_dialog)

        self.act_open = QAction("Open HDF5…", self)
        self.act_open.setIcon(style.standardIcon(QStyle.SP_DirOpenIcon))
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.setToolTip("Open an existing HDF5 file for browsing and editing (Ctrl+O)")
        self.act_open.triggered.connect(self.open_file_dialog)

        # Add files/folder actions
        self.act_add_files = QAction("Add Files…", self)
        self.act_add_files.setIcon(style.standardIcon(QStyle.SP_FileDialogNewFolder))
        self.act_add_files.setShortcut("Ctrl+Shift+F")
        self.act_add_files.setToolTip(
            "Import one or more files into the HDF5 archive (Ctrl+Shift+F)"
        )
        self.act_add_files.triggered.connect(self.add_files_dialog)

        self.act_add_folder = QAction("Add Folder…", self)
        self.act_add_folder.setIcon(style.standardIcon(QStyle.SP_FileDialogNewFolder))
        self.act_add_folder.setShortcut("Ctrl+Shift+D")
        self.act_add_folder.setToolTip(
            "Import an entire folder structure recursively into the HDF5 archive (Ctrl+Shift+D)"
        )
        self.act_add_folder.triggered.connect(self.add_folder_dialog)

        self.act_new_folder = QAction("New Folder…", self)
        self.act_new_folder.setIcon(style.standardIcon(QStyle.SP_FileDialogNewFolder))
        self.act_new_folder.setShortcut("Ctrl+Shift+N")
        self.act_new_folder.setToolTip(
            "Create a new empty group (folder) in the selected location (Ctrl+Shift+N)"
        )
        self.act_new_folder.triggered.connect(self.new_folder_dialog)

        self.act_expand = QAction("Expand All", self)
        self.act_expand.setIcon(style.standardIcon(QStyle.SP_ArrowDown))
        self.act_expand.setToolTip("Expand all groups in the tree view to show full hierarchy")
        self.act_expand.triggered.connect(self.tree.expandAll)

        self.act_collapse = QAction("Collapse All", self)
        self.act_collapse.setIcon(style.standardIcon(QStyle.SP_ArrowUp))
        self.act_collapse.setToolTip("Collapse all groups in the tree view to show only top level")
        self.act_collapse.triggered.connect(self.tree.collapseAll)

        self.act_quit = QAction("Quit", self)
        self.act_quit.setIcon(style.standardIcon(QStyle.SP_DialogCloseButton))
        self.act_quit.setShortcut("Ctrl+Q")
        self.act_quit.setToolTip("Close the application (Ctrl+Q)")
        self.act_quit.triggered.connect(self.close)

        # About action
        self.act_about = QAction("About VibeHDF5", self)
        self.act_about.setIcon(style.standardIcon(QStyle.SP_MessageBoxInformation))
        self.act_about.setToolTip("Show information about VibeHDF5")
        self.act_about.triggered.connect(self.show_about_dialog)

        # Plotting action for CSV tables
        self.act_plot_selected = QAction("Plot Selected Columns (2D Lines)", self)
        self.act_plot_selected.setIcon(style.standardIcon(QStyle.SP_FileDialogContentsView))
        self.act_plot_selected.setToolTip(
            "Plot selected table columns (1 column: Y vs point count; 2+ columns: first is X, others are Y)"
        )
        self.act_plot_selected.triggered.connect(self.plot_selected_columns)
        self.act_plot_selected.setEnabled(False)

        self.act_contourf_selected = QAction("Plot Selected Columns (Contourf)", self)
        self.act_contourf_selected.setIcon(style.standardIcon(QStyle.SP_FileDialogContentsView))
        self.act_contourf_selected.setToolTip(
            "Plot selected table columns (3D contourf plot: X, Y, Z columns required)"
        )
        self.act_contourf_selected.triggered.connect(self.plot_selected_columns_contourf)
        self.act_contourf_selected.setEnabled(False)

        # DAG actions
        self.act_show_dag_pyqt = QAction("Use Pyqtgraph...", self)
        self.act_show_dag_pyqt.setToolTip(
            "Show DAG representation of the HDF5 file structure (using Pyqtgraph)"
        )
        self.act_show_dag_pyqt.triggered.connect(self._show_dag_visualization_pyqtgraph)

        self.act_show_dag = QAction("Use Graphviz...", self)
        self.act_show_dag.setToolTip(
            "Show DAG representation of the HDF5 file structure (using Graphviz)"
        )
        self.act_show_dag.triggered.connect(self._show_dag_visualization)

        # Font size actions
        self.act_increase_font = QAction("Increase Font Size", self)
        self.act_increase_font.setShortcut("Ctrl++")
        self.act_increase_font.setToolTip("Increase GUI font size (Ctrl++)")
        self.act_increase_font.triggered.connect(self._increase_font_size)

        self.act_decrease_font = QAction("Decrease Font Size", self)
        self.act_decrease_font.setShortcut("Ctrl+-")
        self.act_decrease_font.setToolTip("Decrease GUI font size (Ctrl+-)")
        self.act_decrease_font.triggered.connect(self._decrease_font_size)

        self.act_reset_font = QAction("Reset Font Size", self)
        self.act_reset_font.setShortcut("Ctrl+0")
        self.act_reset_font.setToolTip("Reset GUI font size to default (Ctrl+0)")
        self.act_reset_font.triggered.connect(self._reset_font_size)

        # Repack file action
        self.act_repack = QAction("Repack File...", self)
        self.act_repack.setIcon(style.standardIcon(QStyle.SP_BrowserReload))
        self.act_repack.setToolTip("Reclaim space from deleted items by repacking the HDF5 file")
        self.act_repack.triggered.connect(self._repack_file_dialog)

        # File properties action
        self.act_file_properties = QAction("File Properties...", self)
        self.act_file_properties.setIcon(style.standardIcon(QStyle.SP_MessageBoxInformation))
        self.act_file_properties.setToolTip("View detailed information about the HDF5 file")
        self.act_file_properties.triggered.connect(self._show_file_properties_dialog)

        # Merge file action
        self.act_merge_file = QAction("Merge File...", self)
        self.act_merge_file.setIcon(create_h5_file_icon())
        self.act_merge_file.setToolTip(
            "Import contents from another HDF5 file into the current file"
        )
        self.act_merge_file.triggered.connect(self._merge_file_dialog)

        # Create recent file actions (will be populated dynamically)
        for i in range(self.max_recent_files):
            action = QAction(self)
            action.setVisible(False)
            action.triggered.connect(self._open_recent_file)
            self.recent_file_actions.append(action)

    def _create_toolbar(self) -> None:
        """Create and populate the main toolbar."""
        tb = QToolBar("Main", self)
        tb.setIconSize(QSize(16, 16))  # Smaller icon size
        self.addToolBar(tb)
        tb.addAction(self.act_new)
        tb.addAction(self.act_open)
        tb.addSeparator()
        tb.addAction(self.act_add_files)
        tb.addAction(self.act_add_folder)
        tb.addAction(self.act_new_folder)
        tb.addSeparator()
        tb.addAction(self.act_merge_file)
        tb.addSeparator()
        tb.addAction(self.act_file_properties)
        tb.addAction(self.act_repack)
        tb.addSeparator()
        tb.addAction(self.act_expand)
        tb.addAction(self.act_collapse)
        tb.addSeparator()
        tb.addAction(self.act_plot_selected)
        tb.addAction(self.act_contourf_selected)

    def _create_menu_bar(self) -> None:
        """Create and populate the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.act_new)
        file_menu.addAction(self.act_open)
        file_menu.addSeparator()

        # Recent files submenu
        self.recent_files_menu = file_menu.addMenu("Open Recent")
        for action in self.recent_file_actions:
            self.recent_files_menu.addAction(action)
        self.recent_files_menu.addSeparator()
        self.act_clear_recent = QAction("Clear Recent Files", self)
        self.act_clear_recent.triggered.connect(self._clear_recent_files)
        self.recent_files_menu.addAction(self.act_clear_recent)

        file_menu.addSeparator()
        file_menu.addAction(self.act_add_files)
        file_menu.addAction(self.act_add_folder)
        file_menu.addAction(self.act_new_folder)
        file_menu.addSeparator()
        file_menu.addAction(self.act_merge_file)
        file_menu.addSeparator()
        file_menu.addAction(self.act_file_properties)
        file_menu.addAction(self.act_repack)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        # Update recent files menu after creation
        self._update_recent_files_menu()

        # View menu

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.act_expand)
        view_menu.addAction(self.act_collapse)
        view_menu.addSeparator()
        # Add plot actions to a submenu
        plot_menu = view_menu.addMenu("Plot Selected Columns")
        plot_menu.addAction(self.act_plot_selected)
        plot_menu.addAction(self.act_contourf_selected)
        view_menu.addSeparator()
        dag_menu = view_menu.addMenu("Visualize HDF5 File DAG")
        dag_menu.addAction(self.act_show_dag)
        dag_menu.addAction(self.act_show_dag_pyqt)
        view_menu.addSeparator()
        view_menu.addAction(self.act_increase_font)
        view_menu.addAction(self.act_decrease_font)
        view_menu.addAction(self.act_reset_font)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction(self.act_about)

    # Determine where to add new content in the HDF5 file
    def _get_target_group_path(self) -> str:
        """Determine the target HDF5 group path for adding new content.

        Returns:
            String path to target group, or "/" for root
        """
        sel = self.tree.selectionModel().selectedIndexes()
        if not sel:
            return "/"
        index = sel[0].sibling(sel[0].row(), 0)
        item = self.model.itemFromIndex(index)
        if item is None:
            return "/"
        kind = item.data(self.model.ROLE_KIND)
        path = item.data(self.model.ROLE_PATH) or "/"

        candidate = "/"
        if kind == "group":
            candidate = path
        elif kind in ("attr", "attrs-folder"):
            candidate = path  # path points to the owner group/dataset for attrs-folder and attr
        elif kind == "dataset":
            # parent group of dataset
            try:
                candidate = posixpath.dirname(path) or "/"
            except Exception:
                candidate = "/"
        elif kind == "file":
            candidate = "/"

        # Safety: if candidate is a CSV-derived group, drop into its parent instead
        try:
            fpath = self.model.filepath
            if fpath and candidate and candidate != "/":
                with h5py.File(fpath, "r") as h5:
                    try:
                        obj = h5[candidate]
                    except Exception:  # noqa: BLE001
                        obj = None
                    if obj is not None and isinstance(obj, h5py.Group):
                        try:
                            is_csv = (
                                "source_type" in obj.attrs and obj.attrs["source_type"] == "csv"
                            )
                        except Exception:  # noqa: BLE001
                            is_csv = False
                        if is_csv:
                            return posixpath.dirname(candidate) or "/"
        except Exception:  # noqa: BLE001
            pass
        return candidate

    def _get_target_group_path_for_index(self, index) -> str:
        """Determine the target HDF5 group path for a given tree index.

        Args:
            index: QModelIndex of the tree item

        Returns:
            String path to target group, or "/" for root
        """
        if not index or not index.isValid():
            return self._get_target_group_path()
        index = index.sibling(index.row(), 0)
        item = self.model.itemFromIndex(index)
        if item is None:
            return self._get_target_group_path()
        kind = item.data(self.model.ROLE_KIND)
        path = item.data(self.model.ROLE_PATH) or "/"
        # Compute the default candidate target path based on item kind
        if kind == "group":
            candidate = path
        elif kind in ("attr", "attrs-folder"):
            candidate = path  # owner group path
        elif kind == "dataset":
            try:
                candidate = posixpath.dirname(path) or "/"
            except Exception:  # noqa: BLE001
                candidate = "/"
        elif kind == "file":
            candidate = "/"
        else:
            candidate = self._get_target_group_path()

        # If the candidate is a CSV-derived group, redirect the drop to its parent group
        # to avoid placing files inside the CSV group (regardless of expansion state).
        try:
            fpath = self.model.filepath
            if fpath and candidate and candidate != "/" and kind == "group":
                with h5py.File(fpath, "r") as h5:
                    try:
                        obj = h5[candidate]
                    except Exception:  # noqa: BLE001
                        obj = None
                    if obj is not None and isinstance(obj, h5py.Group):
                        try:
                            is_csv = (
                                "source_type" in obj.attrs and obj.attrs["source_type"] == "csv"
                            )
                        except Exception:  # noqa: BLE001
                            is_csv = False
                        if is_csv:
                            parent = posixpath.dirname(candidate) or "/"
                            return parent
        except Exception:  # noqa: BLE001
            pass

        return candidate

    def add_files_dialog(self) -> None:
        """Open a file selection dialog and add selected files to the HDF5 archive."""
        fpath = self.model.filepath
        if not fpath:
            QMessageBox.information(self, "No file", "Open an HDF5 file first.")
            return
        files, _ = QFileDialog.getOpenFileNames(self, "Select files to add")
        if not files:
            return
        target_group = self._get_target_group_path()
        added, errors = self._add_items_batch(files, [], target_group)
        self.model.load_file(fpath)
        self.tree.expandToDepth(1)
        if errors:
            QMessageBox.warning(self, "Completed with errors", "\n".join(errors))
        elif added:
            self.statusBar().showMessage(f"Added {added} file(s) to {target_group}", 5000)

        # Update file size display after adding files
        self._update_file_size_display()

    def add_folder_dialog(self) -> None:
        """Open a folder selection dialog and add folder contents recursively to HDF5."""
        fpath = self.model.filepath
        if not fpath:
            QMessageBox.information(self, "No file", "Open an HDF5 file first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select folder to add")
        if not directory:
            return
        target_group = self._get_target_group_path()
        added, errors = self._add_items_batch([], [directory], target_group)
        self.model.load_file(fpath)
        self.tree.expandToDepth(2)
        if errors:
            QMessageBox.warning(self, "Completed with errors", "\n".join(errors))
        elif added:
            self.statusBar().showMessage(f"Added {added} item(s) under {target_group}", 5000)

        # Update file size display after adding folder
        self._update_file_size_display()

    def new_folder_dialog(self) -> None:
        """Create a new empty group (folder) in the HDF5 file."""
        fpath = self.model.filepath
        if not fpath:
            QMessageBox.information(self, "No file", "Open an HDF5 file first.")
            return

        target_group = self._get_target_group_path()

        # Get folder name from user

        folder_name, ok = QInputDialog.getText(
            self, "New Folder", f"Enter folder name to create in {target_group}:"
        )

        if not ok or not folder_name:
            return

        # Sanitize the folder name
        folder_name = sanitize_hdf5_name(folder_name)
        if not folder_name:
            QMessageBox.warning(self, "Invalid Name", "Folder name cannot be empty.")
            return

        # Create the full path for the new group
        if target_group == "/":
            new_group_path = "/" + folder_name
        else:
            new_group_path = posixpath.join(target_group, folder_name)

        # Create the group in the HDF5 file
        try:
            with h5py.File(fpath, "r+") as h5:
                if new_group_path in h5:
                    QMessageBox.warning(
                        self, "Already Exists", f"Group '{new_group_path}' already exists."
                    )
                    return
                h5.create_group(new_group_path)

            # Reload the tree and expand to show the new folder
            self.model.load_file(fpath)
            self.tree.expandToDepth(2)
            self.statusBar().showMessage(f"Created folder: {new_group_path}", 5000)

            # Update file size display after creating folder
            self._update_file_size_display()

        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to create folder: {exc}")

    def _move_hdf5_item(self, source_path: str, target_group: str) -> bool:
        """Move an HDF5 dataset or group from source_path to target_group.

        Returns True if successful, False otherwise.
        """
        fpath = self.model.filepath
        if not fpath:
            QMessageBox.warning(self, "No file", "No HDF5 file loaded.")
            return False

        # Can't move to itself
        if source_path == target_group:
            QMessageBox.warning(self, "Invalid Move", "Cannot move item to itself.")
            return False

        # Can't move into its own child
        if target_group.startswith(source_path + "/"):
            QMessageBox.warning(self, "Invalid Move", "Cannot move item into its own child.")
            return False

        try:
            with h5py.File(fpath, "r+") as h5:
                # Check if source exists
                if source_path not in h5:
                    QMessageBox.warning(self, "Not Found", f"Source '{source_path}' not found.")
                    return False

                # Prevent moving items out of CSV groups
                source_parent = posixpath.dirname(source_path) or "/"
                if (
                    source_parent
                    and source_parent != "/"
                    and isinstance(h5[source_parent], h5py.Group)
                ):
                    parent_grp = h5[source_parent]
                    if (
                        "source_type" in parent_grp.attrs
                        and parent_grp.attrs["source_type"] == "csv"
                    ):
                        QMessageBox.warning(
                            self, "Invalid Move", "Cannot move items out of CSV groups."
                        )
                        return False

                # Prevent moving into CSV groups
                if (
                    target_group
                    and target_group != "/"
                    and isinstance(h5[target_group], h5py.Group)
                ):
                    grp = h5[target_group]
                    if "source_type" in grp.attrs and grp.attrs["source_type"] == "csv":
                        QMessageBox.warning(
                            self, "Invalid Target", "Cannot move items into CSV groups."
                        )
                        return False

                # Construct new path
                item_name = os.path.basename(source_path)
                if target_group == "/":
                    new_path = "/" + item_name
                else:
                    new_path = posixpath.join(target_group, item_name)

                # If already at destination, treat as no-op
                if source_path == new_path:
                    return True

                # Check if destination exists
                if new_path in h5:
                    resp = QMessageBox.question(
                        self,
                        "Overwrite?",
                        f"'{new_path}' already exists. Overwrite?",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No,
                    )
                    if resp != QMessageBox.Yes:
                        return False
                    del h5[new_path]

                # Perform the move (copy + delete)
                h5.copy(source_path, new_path)
                del h5[source_path]

                return True

        except Exception as exc:
            QMessageBox.critical(self, "Move Failed", f"Failed to move item: {exc}")
            return False

    def _add_items_batch(
        self, files: list[str], folders: list[str], target_group: str
    ) -> tuple[int, list[str]]:
        """Add multiple files and folders to the HDF5 archive in batch.

        Args:
            files: List of file paths to add
            folders: List of folder paths to add recursively
            target_group: HDF5 group path where items should be added

        Returns:
            Tuple of (added_count, error_list) where added_count is number of successfully added items
            and error_list contains error messages for failed items
        """
        fpath = self.model.filepath
        if not fpath:
            return 0, ["No HDF5 file loaded"]
        errors: list[str] = []
        added = 0
        try:
            with h5py.File(fpath, "r+") as h5:
                # Final safety: never allow writing into a CSV-derived group
                try:
                    if (
                        target_group
                        and target_group != "/"
                        and isinstance(h5[target_group], h5py.Group)
                    ):
                        grp = h5[target_group]
                        if "source_type" in grp.attrs and grp.attrs["source_type"] == "csv":
                            target_group = posixpath.dirname(target_group) or "/"
                except Exception:  # noqa: BLE001
                    pass

                if target_group == "/":
                    base_grp = h5
                else:
                    base_grp = h5.require_group(target_group)
                for path_on_disk in files:
                    name = os.path.basename(path_on_disk)
                    if name in excluded_files:
                        continue
                    h5_path = (
                        posixpath.join(target_group, name) if target_group != "/" else "/" + name
                    )
                    try:
                        self._create_dataset_from_file(base_grp, h5_path, path_on_disk, np)
                        added += 1
                    except FileExistsError:
                        resp = QMessageBox.question(
                            self,
                            "Overwrite?",
                            f"'{h5_path}' exists. Overwrite?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.No,
                        )
                        if resp == QMessageBox.Yes:
                            del h5[h5_path]
                            self._create_dataset_from_file(base_grp, h5_path, path_on_disk, np)
                            added += 1
                    except Exception as exc:  # noqa: BLE001
                        errors.append(f"{name}: {exc}")
                for directory in folders:
                    base_name = os.path.basename(os.path.normpath(directory))
                    if target_group == "/":
                        root_h5_group = "/" + base_name
                    else:
                        root_h5_group = posixpath.join(target_group, base_name)
                    for dirpath, dirnames, filenames in os.walk(directory):
                        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
                        rel = os.path.relpath(dirpath, directory)
                        rel = "." if rel == "." else rel.replace("\\", "/")
                        current_group_path = (
                            root_h5_group if rel == "." else posixpath.join(root_h5_group, rel)
                        )
                        grp = h5.require_group(current_group_path)
                        for filename in filenames:
                            if filename in excluded_files:
                                continue
                            file_on_disk = os.path.join(dirpath, filename)
                            h5_path = posixpath.join(current_group_path, filename)
                            try:
                                self._create_dataset_from_file(grp, h5_path, file_on_disk, np)
                                added += 1
                            except FileExistsError:
                                resp = QMessageBox.question(
                                    self,
                                    "Overwrite?",
                                    f"'{h5_path}' exists. Overwrite?",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No,
                                )
                                if resp == QMessageBox.Yes:
                                    del h5[h5_path]
                                    self._create_dataset_from_file(grp, h5_path, file_on_disk, np)
                                    added += 1
                            except Exception as exc:  # noqa: BLE001
                                errors.append(f"{h5_path}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return 0, [str(exc)]
        return added, errors

    def _create_dataset_from_file(self, grp, h5_path: str, disk_path: str, np) -> None:
        """Create a dataset at h5_path from a file on disk under the given group (or file root).

        For CSV files, creates a group with individual datasets for each column.
        For other files, stores as text or binary data.

        Args:
            grp: HDF5 group or file object where the dataset should be created
            h5_path: Full HDF5 path for the new dataset
            disk_path: Path to the file on disk
            np: numpy module reference for array creation

        Raises:
            FileExistsError: If the path already exists in the HDF5 file
        """
        # Check existence
        f = grp.file
        if h5_path in f:
            raise FileExistsError(h5_path)

        # Special handling for CSV files
        if disk_path.lower().endswith(".csv"):
            self._create_datasets_from_csv(f, h5_path, disk_path)
            return

        # Ensure parent groups exist
        parent = os.path.dirname(h5_path).replace("\\", "/")
        if parent and parent != "/":
            f.require_group(parent)
        # Try text then binary
        try:
            with open(disk_path, "r", encoding="utf-8") as fin:
                data = fin.read()
            # Compress text data with gzip (level 9 for maximum compression)
            compressed = gzip.compress(data.encode("utf-8"), compresslevel=9)
            ds = f.create_dataset(h5_path, data=np.frombuffer(compressed, dtype="uint8"))
            # Mark as compressed text so we can decompress on read
            ds.attrs["compressed"] = "gzip"
            ds.attrs["original_encoding"] = "utf-8"
            return
        except Exception:  # noqa: BLE001
            pass
        # Read as binary and compress
        with open(disk_path, "rb") as fin:
            bdata = fin.read()
        # Compress binary data with gzip (level 9 for maximum compression)
        compressed = gzip.compress(bdata, compresslevel=9)
        ds = f.create_dataset(h5_path, data=np.frombuffer(compressed, dtype="uint8"))
        # Mark as compressed binary so we can decompress on read
        ds.attrs["compressed"] = "gzip"
        ds.attrs["original_encoding"] = "binary"

    def _create_datasets_from_csv(self, f: h5py.File, h5_path: str, disk_path: str) -> None:
        """Convert a CSV file to HDF5 datasets.

        Creates a group at h5_path (without .csv extension) containing one dataset per column.
        Each dataset contains the column data with appropriate dtype.

        Args:
            f: Open HDF5 file object
            h5_path: Desired HDF5 path for the CSV group (will have .csv extension removed)
            disk_path: Path to the CSV file on disk
        """
        # Create progress dialog
        progress = self._create_progress_dialog("Reading CSV file...")

        # Read CSV with pandas
        try:
            df = pd.read_csv(disk_path)
            progress.setValue(20)
            QApplication.processEvents()
            if progress.wasCanceled():
                raise ValueError("CSV import cancelled by user")
        except Exception as exc:  # noqa: BLE001
            progress.close()
            raise ValueError(f"Failed to read CSV file: {exc}") from exc

        # Remove .csv extension from group name
        group_path = h5_path
        if group_path.lower().endswith(".csv"):
            group_path = group_path[:-4]

        # Ensure parent groups exist
        parent = os.path.dirname(group_path).replace("\\", "/")
        if parent and parent != "/":
            f.require_group(parent)

        # Create a group for the CSV data
        grp = f.create_group(group_path)

        # Add metadata about the source file
        grp.attrs["source_file"] = os.path.basename(disk_path)
        grp.attrs["source_type"] = "csv"
        grp.attrs["column_names"] = list(df.columns)

        progress.setLabelText(f"Creating datasets for {len(df.columns)} columns...")
        progress.setValue(30)
        QApplication.processEvents()

        # Create a dataset for each column
        used_names: set[str] = set()
        column_dataset_names: list[str] = []
        total_cols = len(df.columns)
        for idx, col in enumerate(df.columns):
            if progress.wasCanceled():
                # Clean up partial group
                try:
                    del f[group_path]
                except Exception:  # noqa: BLE001
                    pass
                progress.close()
                raise ValueError("CSV import cancelled by user")

            # Update progress (30-90% range for column processing)
            progress_val = 30 + int((idx / total_cols) * 60)
            progress.setValue(progress_val)
            progress.setLabelText(f"Creating dataset {idx + 1}/{total_cols}: {col}")
            QApplication.processEvents()

            col_data = df[col]

            # Clean column name for use as dataset name
            base = sanitize_hdf5_name(str(col))
            ds_name = base if base else "unnamed_column"
            # Ensure uniqueness within the group
            if ds_name in used_names:
                i = 2
                while f"{ds_name}_{i}" in used_names or f"{ds_name}_{i}" in grp:
                    i += 1
                ds_name = f"{ds_name}_{i}"
            used_names.add(ds_name)
            column_dataset_names.append(ds_name)

            # Convert pandas Series to numpy array with appropriate dtype
            # Determine optimal chunk size for compression
            data_len = len(col_data)
            chunk_size = min(10000, data_len) if data_len > 1000 else None

            if col_data.dtype == "object":
                # For object dtype, convert to Python list then create dataset
                # This avoids numpy unicode string issues
                try:
                    # Convert to Python strings
                    str_list = [str(x) for x in col_data.values]
                    # Use gzip compression for string columns
                    grp.create_dataset(
                        ds_name,
                        data=str_list,
                        dtype=h5py.string_dtype(encoding="utf-8"),
                        compression="gzip",
                        compression_opts=6,  # Higher compression for text (1-9, 6 is good balance)
                        chunks=(chunk_size,)
                        if chunk_size
                        else True,  # Enable chunking for compression
                    )
                except Exception:  # noqa: BLE001
                    # Fallback: convert to bytes with compression
                    str_list = [str(x) for x in col_data.values]
                    grp.create_dataset(
                        ds_name,
                        data=str_list,
                        dtype=h5py.string_dtype(encoding="utf-8"),
                        compression="gzip",
                        compression_opts=6,
                        chunks=(chunk_size,) if chunk_size else True,
                    )
            else:
                # Numeric or other numpy-supported dtypes with compression
                # Use chunking to enable compression and improve I/O for partial reads
                grp.create_dataset(
                    ds_name,
                    data=col_data.values,
                    compression="gzip",
                    compression_opts=4,  # Moderate compression for numeric data (1-9)
                    chunks=(chunk_size,) if chunk_size else True,
                )

        # Persist the actual dataset names used for each column (same order as column_names)
        progress.setLabelText("Finalizing CSV import...")
        progress.setValue(95)
        QApplication.processEvents()

        try:
            grp.attrs["column_dataset_names"] = np.array(column_dataset_names, dtype=object)
        except Exception:  # noqa: BLE001
            # Fallback to list assignment if dtype=object attr not permitted
            grp.attrs["column_dataset_names"] = column_dataset_names

        progress.setValue(100)
        progress.close()

    def new_file_dialog(self) -> None:
        """Create a new HDF5 file."""
        last_dir = os.getcwd()
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Create New HDF5 File",
            last_dir,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if not filepath:
            return

        # Add .h5 extension if no extension provided
        if not filepath.endswith((".h5", ".hdf5")):
            filepath += ".h5"

        # Check if file already exists
        if os.path.exists(filepath):
            resp = QMessageBox.question(
                self,
                "File exists",
                f"File '{filepath}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return

        try:
            # Create a new empty HDF5 file
            with h5py.File(filepath, "w"):
                # Create an empty file with a root group
                pass

            # Load the newly created file
            self.load_hdf5(filepath)
            self.statusBar().showMessage(f"Created new HDF5 file: {filepath}", 5000)
            self._update_file_size_display()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Failed to create file",
                f"Could not create HDF5 file:\n{filepath}\n\n{exc}",
            )

    def open_file_dialog(self) -> None:
        """Open a file selection dialog to open an existing HDF5 file."""
        last_dir = os.getcwd()
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open HDF5 File",
            last_dir,
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )
        if filepath:
            self.load_hdf5(filepath)

    def load_hdf5(self, path: str | Path) -> None:
        """Load an HDF5 file for viewing and editing.

        Args:
            path: Path to the HDF5 file to load
        """
        path = str(path)
        try:
            self.model.load_file(path)
        except Exception as exc:  # show friendly error dialog
            QMessageBox.critical(
                self,
                "Failed to open HDF5",
                f"Could not open file:\n{path}\n\n{exc}",
            )
            return
        self.statusBar().showMessage(path)
        self.tree.expandToDepth(1)
        self.preview_label.setText("No selection")
        self._set_preview_text("")

        # Update file size display
        self._update_file_size_display()

        # Add to recent files
        self._add_recent_file(path)

    def _update_file_size_display(self) -> None:
        """Update the file size display in the status bar."""
        if not self.model or not self.model.filepath:
            self.file_size_label.setText("")
            return

        try:
            file_path = Path(self.model.filepath)
            if file_path.exists():
                size_bytes = file_path.stat().st_size
                size_str = self._format_file_size(size_bytes)
                self.file_size_label.setText(f"File size: {size_str}")
            else:
                self.file_size_label.setText("")
        except Exception:
            self.file_size_label.setText("")

    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format.

        Args:
            size_bytes: File size in bytes

        Returns:
            Formatted string like '1.5 MB' or '234 KB'
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                if unit == "B":
                    return f"{size_bytes:.0f} {unit}"
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    def _get_recent_files(self) -> list[str]:
        """Get the list of recent files from QSettings.

        Returns:
            List of recent file paths, most recent first
        """
        recent_files = self.settings.value("recent_files", [])
        if not isinstance(recent_files, list):
            recent_files = []
        # Filter out files that no longer exist
        return [f for f in recent_files if os.path.exists(f)]

    def _add_recent_file(self, filepath: str) -> None:
        """Add a file to the recent files list.

        Args:
            filepath: Absolute path to the file to add
        """
        filepath = os.path.abspath(filepath)
        recent_files = self._get_recent_files()

        # Remove if already in list
        if filepath in recent_files:
            recent_files.remove(filepath)

        # Add to front of list
        recent_files.insert(0, filepath)

        # Limit to max_recent_files
        recent_files = recent_files[: self.max_recent_files]

        # Save to settings
        self.settings.setValue("recent_files", recent_files)

        # Update menu
        self._update_recent_files_menu()

    def _update_recent_files_menu(self) -> None:
        """Update the recent files menu with current list."""
        recent_files = self._get_recent_files()

        for i, action in enumerate(self.recent_file_actions):
            if i < len(recent_files):
                filepath = recent_files[i]
                # Show just the filename and first part of path for readability
                display_name = f"{i + 1}. {os.path.basename(filepath)}"
                action.setText(display_name)
                action.setData(filepath)
                action.setVisible(True)
                action.setToolTip(filepath)
            else:
                action.setVisible(False)

        # Enable/disable clear action and menu
        has_files = len(recent_files) > 0
        self.act_clear_recent.setEnabled(has_files)
        self.recent_files_menu.setEnabled(has_files)

    def _open_recent_file(self) -> None:
        """Open a file from the recent files list."""
        action = self.sender()
        if action:
            filepath = action.data()
            if filepath and os.path.exists(filepath):
                self.load_hdf5(filepath)
            else:
                # File no longer exists, remove from list
                recent_files = self._get_recent_files()
                if filepath in recent_files:
                    recent_files.remove(filepath)
                    self.settings.setValue("recent_files", recent_files)
                    self._update_recent_files_menu()
                QMessageBox.warning(
                    self, "File Not Found", f"The file no longer exists:\n{filepath}"
                )

    def _clear_recent_files(self) -> None:
        """Clear the recent files list."""
        reply = QMessageBox.question(
            self,
            "Clear Recent Files",
            "Are you sure you want to clear the recent files list?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.settings.setValue("recent_files", [])
            self._update_recent_files_menu()
            self.statusBar().showMessage("Recent files list cleared", 3000)

    def show_about_dialog(self) -> None:
        """Show the About VibeHDF5 dialog."""
        try:
            from vibehdf5 import __version__

            version = __version__
        except ImportError:
            version = "unknown"

        about_text = f"""<h2>VibeHDF5</h2>
        <p><b>Version:</b> {version}</p>
        <p>A powerful, lightweight GUI application for browsing, managing, and visualizing HDF5 file structures.</p>

        <p><b>Features:</b></p>
        <ul>
        <li>Browse and explore HDF5 file hierarchies</li>
        <li>Preview datasets with syntax highlighting</li>
        <li>Import and manage CSV data with filtering and plotting</li>
        <li>Drag-and-drop file import and export</li>
        <li>Interactive matplotlib plotting</li>
        </ul>

        <p><b>Author:</b> Jacob Williams</p>
        <p><b>Repository:</b> <a href="https://github.com/jacobwilliams/vibehdf5">github.com/jacobwilliams/vibehdf5</a></p>
        <p><b>License:</b> MIT</p>
        """

        QMessageBox.about(self, "About VibeHDF5", about_text)

    def _increase_font_size(self) -> None:
        """Increase the application font size."""
        app = QApplication.instance()
        if app:
            font = app.font()
            current_size = font.pointSize()
            if current_size > 0:  # pointSize returns -1 if not set
                new_size = min(current_size + 1, 32)  # Cap at 32pt
                font.setPointSize(new_size)
            else:
                # Use pixel size as fallback
                pixel_size = font.pixelSize()
                if pixel_size > 0:
                    new_size = min(pixel_size + 1, 42)  # Cap at 42px
                    font.setPixelSize(new_size)
                else:
                    # Default starting point
                    font.setPointSize(12)
            app.setFont(font)

            # Also update the preview_edit fixed-width font
            preview_font = self.preview_edit.font()
            if current_size > 0:
                preview_font.setPointSize(new_size)
            else:
                preview_font.setPixelSize(new_size)
            self.preview_edit.setFont(preview_font)

            self.statusBar().showMessage(f"Font size increased to {font.pointSize()}pt", 2000)

    def _decrease_font_size(self) -> None:
        """Decrease the application font size."""
        app = QApplication.instance()
        if app:
            font = app.font()
            current_size = font.pointSize()
            if current_size > 0:
                new_size = max(current_size - 1, 6)  # Minimum 6pt
                font.setPointSize(new_size)
            else:
                # Use pixel size as fallback
                pixel_size = font.pixelSize()
                if pixel_size > 0:
                    new_size = max(pixel_size - 1, 8)  # Minimum 8px
                    font.setPixelSize(new_size)
                else:
                    # Default starting point
                    font.setPointSize(10)
            app.setFont(font)

            # Also update the preview_edit fixed-width font
            preview_font = self.preview_edit.font()
            if current_size > 0:
                preview_font.setPointSize(new_size)
            else:
                preview_font.setPixelSize(new_size)
            self.preview_edit.setFont(preview_font)

            self.statusBar().showMessage(f"Font size decreased to {font.pointSize()}pt", 2000)

    def _reset_font_size(self) -> None:
        """Reset the application font size to default."""
        app = QApplication.instance()
        if app:
            # Get system default font
            default_font = QApplication.font()
            app.setFont(default_font)

            # Also reset the preview_edit fixed-width font
            try:
                fixed = QFontDatabase.systemFont(QFontDatabase.FixedFont)
            except Exception:
                fixed = QFont("Courier New")
            self.preview_edit.setFont(fixed)

            self.statusBar().showMessage(
                f"Font size reset to default ({default_font.pointSize()}pt)", 2000
            )

    def _repack_file_dialog(self) -> None:
        """Dialog to repack the HDF5 file to reclaim space from deleted items."""
        if not self.model or not self.model.filepath:
            QMessageBox.information(self, "No File", "No HDF5 file is currently loaded.")
            return

        current_path = Path(self.model.filepath)
        if not current_path.exists():
            QMessageBox.warning(self, "File Not Found", "The current file no longer exists.")
            return

        # Get current file size
        current_size = current_path.stat().st_size
        current_size_str = self._format_file_size(current_size)

        # Show confirmation dialog with explanation
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Repack HDF5 File")
        msg.setText(
            "Repack the HDF5 file to reclaim space from deleted items?\n\n"
            f"Current file size: {current_size_str}\n\n"
            "Note: When you delete datasets or groups, HDF5 marks the space as unused "
            "but doesn't immediately reclaim it. Repacking creates a new optimized file "
            "that reclaims this space.\n\n"
            "This operation may take a few moments for large files."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)

        if msg.exec() != QMessageBox.Yes:
            return

        # Perform the repack
        try:
            # Create a temporary file for the repacked version
            temp_fd, temp_path = tempfile.mkstemp(suffix=".h5", prefix="repack_")
            os.close(temp_fd)  # Close the file descriptor, we'll let h5py handle it

            # Show progress dialog
            progress = QProgressDialog(
                "Repacking file...\nThis may take a moment for large files.", None, 0, 0, self
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            QApplication.processEvents()

            try:
                # Copy all data to a new file (this automatically repacks)
                with h5py.File(str(current_path), "r") as source:
                    with h5py.File(temp_path, "w") as dest:
                        # Copy all groups and datasets recursively
                        def copy_recursively(src_group, dst_group):
                            for key in src_group.keys():
                                if isinstance(src_group[key], h5py.Group):
                                    # Create group and copy attributes
                                    new_group = dst_group.create_group(key)
                                    for attr_key, attr_val in src_group[key].attrs.items():
                                        new_group.attrs[attr_key] = attr_val
                                    # Recurse into subgroup
                                    copy_recursively(src_group[key], new_group)
                                elif isinstance(src_group[key], h5py.Dataset):
                                    # Copy dataset with compression if available
                                    src_ds = src_group[key]
                                    dst_group.create_dataset(
                                        key,
                                        data=src_ds[()],
                                        dtype=src_ds.dtype,
                                        compression=src_ds.compression,
                                        compression_opts=src_ds.compression_opts,
                                    )
                                    # Copy attributes
                                    for attr_key, attr_val in src_ds.attrs.items():
                                        dst_group[key].attrs[attr_key] = attr_val

                        # Copy root attributes
                        for attr_key, attr_val in source.attrs.items():
                            dest.attrs[attr_key] = attr_val

                        # Copy all contents
                        copy_recursively(source, dest)

                progress.close()

                # Get new file size
                new_size = Path(temp_path).stat().st_size
                new_size_str = self._format_file_size(new_size)
                saved_bytes = current_size - new_size
                saved_str = self._format_file_size(saved_bytes) if saved_bytes > 0 else "0 B"

                # Replace original file with repacked version
                shutil.move(temp_path, str(current_path))

                # Reload the file
                self.load_hdf5(str(current_path))

                # Show results
                result_msg = (
                    f"File successfully repacked!\n\n"
                    f"Original size: {current_size_str}\n"
                    f"New size: {new_size_str}\n"
                    f"Space reclaimed: {saved_str}"
                )

                if saved_bytes <= 0:
                    result_msg += (
                        "\n\nNo space was reclaimed. The file was already optimally packed."
                    )

                QMessageBox.information(self, "Repack Complete", result_msg)

            finally:
                # Clean up temp file if it still exists
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        except Exception as exc:
            QMessageBox.critical(
                self,
                "Repack Failed",
                f"Failed to repack the file:\n\n{exc}\n\nThe original file has not been modified.",
            )

    def _show_file_properties_dialog(self) -> None:
        """Show a dialog with detailed file properties and metadata."""
        if not self.model or not self.model.filepath:
            QMessageBox.information(self, "No File", "No HDF5 file is currently loaded.")
            return

        file_path = Path(self.model.filepath)
        if not file_path.exists():
            QMessageBox.warning(self, "File Not Found", "The current file no longer exists.")
            return

        try:
            # Gather file information
            info = {}

            # Basic file info
            info["File Path"] = str(file_path.absolute())
            info["File Name"] = file_path.name

            # File size
            file_size = file_path.stat().st_size
            info["File Size"] = f"{self._format_file_size(file_size)} ({file_size:,} bytes)"

            # File timestamps
            import datetime

            mtime = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)
            info["Modified"] = mtime.strftime("%Y-%m-%d %H:%M:%S")
            ctime = datetime.datetime.fromtimestamp(file_path.stat().st_ctime)
            info["Created"] = ctime.strftime("%Y-%m-%d %H:%M:%S")

            # HDF5-specific information
            with h5py.File(str(file_path), "r") as h5:
                # Count items
                num_groups = 0
                num_datasets = 0
                num_attrs = len(h5.attrs)
                total_datasets_size = 0

                def count_items(group):
                    nonlocal num_groups, num_datasets, num_attrs, total_datasets_size
                    for key in group.keys():
                        if isinstance(group[key], h5py.Group):
                            num_groups += 1
                            num_attrs += len(group[key].attrs)
                            count_items(group[key])
                        elif isinstance(group[key], h5py.Dataset):
                            num_datasets += 1
                            num_attrs += len(group[key].attrs)
                            try:
                                # Estimate dataset size
                                ds = group[key]
                                if hasattr(ds, "nbytes"):
                                    total_datasets_size += ds.nbytes
                            except Exception:
                                pass

                count_items(h5)

                info["Groups"] = f"{num_groups:,}"
                info["Datasets"] = f"{num_datasets:,}"
                info["Attributes"] = f"{num_attrs:,}"
                info["Total Items"] = f"{num_groups + num_datasets:,}"

                # Dataset storage info
                if total_datasets_size > 0:
                    info["Dataset Size"] = (
                        f"{self._format_file_size(total_datasets_size)} ({total_datasets_size:,} bytes)"
                    )
                    overhead = file_size - total_datasets_size
                    if overhead > 0:
                        overhead_pct = (overhead / file_size) * 100
                        info["Metadata Overhead"] = (
                            f"{self._format_file_size(overhead)} ({overhead_pct:.1f}%)"
                        )

                # HDF5 library version
                try:
                    info["HDF5 Library"] = h5py.version.hdf5_version
                except Exception:
                    pass

                # h5py version
                try:
                    info["h5py Version"] = h5py.version.version
                except Exception:
                    pass

                # File format version
                try:
                    # Try to get userblock size
                    fid = h5.id
                    fcpl = fid.get_create_plist()
                    userblock_size = fcpl.get_userblock()
                    if userblock_size > 0:
                        info["Userblock Size"] = f"{self._format_file_size(userblock_size)}"
                except Exception:
                    pass

                # Root attributes
                if len(h5.attrs) > 0:
                    root_attrs = []
                    for key in list(h5.attrs.keys())[:5]:  # Show first 5
                        root_attrs.append(key)
                    info["Root Attributes"] = ", ".join(root_attrs)
                    if len(h5.attrs) > 5:
                        info["Root Attributes"] += f" (+{len(h5.attrs) - 5} more)"

            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("File Properties")
            dialog.setMinimumWidth(500)

            layout = QVBoxLayout(dialog)

            # Add header label
            header = QLabel(f"<b>HDF5 File Information</b>")
            header.setStyleSheet("font-size: 14px; padding: 5px;")
            layout.addWidget(header)

            # Create table for properties
            table = QTableWidget()
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Property", "Value"])
            table.horizontalHeader().setStretchLastSection(True)
            table.setAlternatingRowColors(True)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.verticalHeader().setVisible(False)

            # Populate table
            table.setRowCount(len(info))
            for i, (key, value) in enumerate(info.items()):
                # Property name
                key_item = QTableWidgetItem(key)
                key_item.setFont(QFont(key_item.font().family(), -1, QFont.Bold))
                table.setItem(i, 0, key_item)

                # Property value
                value_item = QTableWidgetItem(str(value))
                table.setItem(i, 1, value_item)

            table.resizeColumnsToContents()
            table.setColumnWidth(0, 180)  # Fixed width for property names

            layout.addWidget(table)

            # Add buttons
            button_box = QHBoxLayout()
            button_box.addStretch()

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            close_btn.setDefault(True)
            button_box.addWidget(close_btn)

            layout.addLayout(button_box)

            dialog.exec()

        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to retrieve file properties:\n\n{exc}")

    def _get_dataset_info(self, dataset_path: str) -> dict[str, str]:
        """
        Retrieve detailed information about a dataset in the currently loaded HDF5 file.

        Args:
            dataset_path (str): The HDF5 path to the dataset to inspect.

        Returns:
            dict[str, str]: A dictionary containing dataset properties and statistics. Keys include:
                - 'Name': Full HDF5 path of the dataset
                - 'Shape': Shape of the dataset
                - 'Data Type': Data type of the dataset
                - 'Size': Number of elements
                - 'Memory Size': Size in memory (if available)
                - 'Storage Size': Actual disk space used (if available)
                - 'Compression Ratio': Ratio of memory size to storage size (if available)
                - 'Chunks': Chunk shape (if chunked)
                - 'Chunked': Whether the dataset is chunked
                - 'Compression': Compression type (if any)
                - 'Compression Options': Compression options (if any)
                - 'Scale-Offset Filter': Scale-offset filter info (if any)
                - 'Shuffle Filter': Whether shuffle filter is enabled
                - 'Fletcher32 Checksum': Whether Fletcher32 checksum is enabled
                - 'Fill Value': Fill value (if any)
                - 'Attributes': Number of attributes
                - 'Attribute Names': Names of up to 5 attributes
                - 'External Storage': Number of external storage files (if any)
                - 'Dimensions': Number of dimensions (for multidimensional datasets)
                - '  Dimension N': Size of each dimension
                - 'Min Value', 'Max Value', 'Mean Value', 'Std Dev': Statistics for numeric datasets (if small enough)

        If the dataset is not found or not a dataset, returns an empty dict.
        """

        if not self.model or not self.model.filepath:
            QMessageBox.information(self, "No File", "No HDF5 file is currently loaded.")
            return {}

        try:
            with h5py.File(self.model.filepath, "r") as h5:
                if dataset_path not in h5:
                    QMessageBox.warning(self, "Not Found", f"Dataset '{dataset_path}' not found.")
                    return

                ds = h5[dataset_path]
                if not isinstance(ds, h5py.Dataset):
                    QMessageBox.warning(
                        self, "Not a Dataset", f"'{dataset_path}' is not a dataset."
                    )
                    return

                # Gather dataset information
                info = {}

                # Basic info
                info["Name"] = ds.name
                info["Shape"] = str(ds.shape)
                info["Data Type"] = str(ds.dtype)
                info["Size"] = f"{ds.size:,} elements"

                # Memory size
                if hasattr(ds, "nbytes"):
                    info["Memory Size"] = (
                        f"{self._format_file_size(ds.nbytes)} ({ds.nbytes:,} bytes)"
                    )

                # Storage size (actual disk space used)
                try:
                    storage_size = ds.id.get_storage_size()
                    if storage_size > 0:
                        info["Storage Size"] = (
                            f"{self._format_file_size(storage_size)} ({storage_size:,} bytes)"
                        )
                        # Calculate compression ratio
                        if hasattr(ds, "nbytes") and ds.nbytes > 0:
                            ratio = ds.nbytes / storage_size
                            info["Compression Ratio"] = f"{ratio:.2f}:1"
                except Exception:
                    pass

                # Chunks
                if ds.chunks:
                    info["Chunks"] = str(ds.chunks)
                    info["Chunked"] = "Yes"
                else:
                    info["Chunked"] = "No (Contiguous)"

                # Compression
                compression = ds.compression
                if compression:
                    info["Compression"] = compression
                    if ds.compression_opts:
                        info["Compression Options"] = str(ds.compression_opts)
                else:
                    info["Compression"] = "None"

                # Filters
                try:
                    if ds.scaleoffset:
                        info["Scale-Offset Filter"] = str(ds.scaleoffset)
                except Exception:
                    pass

                try:
                    if ds.shuffle:
                        info["Shuffle Filter"] = "Enabled"
                except Exception:
                    pass

                try:
                    if ds.fletcher32:
                        info["Fletcher32 Checksum"] = "Enabled"
                except Exception:
                    pass

                # Fill value
                try:
                    fillvalue = ds.fillvalue
                    if fillvalue is not None:
                        info["Fill Value"] = str(fillvalue)
                except Exception:
                    pass

                # Attributes
                num_attrs = len(ds.attrs)
                info["Attributes"] = f"{num_attrs:,}"
                if num_attrs > 0:
                    attr_names = []
                    for key in list(ds.attrs.keys())[:5]:  # Show first 5
                        attr_names.append(key)
                    info["Attribute Names"] = ", ".join(attr_names)
                    if num_attrs > 5:
                        info["Attribute Names"] += f" (+{num_attrs - 5} more)"

                # External storage
                try:
                    external = ds.external
                    if external:
                        info["External Storage"] = f"{len(external)} file(s)"
                except Exception:
                    pass

                # Dimensions (for multidimensional datasets)
                if len(ds.shape) > 1:
                    info["Dimensions"] = f"{len(ds.shape)}D"
                    for i, dim_size in enumerate(ds.shape):
                        info[f"  Dimension {i}"] = f"{dim_size:,}"

                # For numeric data, show value range if dataset is small enough
                if ds.size > 0 and ds.size <= 1000000 and ds.dtype.kind in ("i", "u", "f"):
                    try:
                        data = ds[:]
                        if data.size > 0:
                            info["Min Value"] = str(np.min(data))
                            info["Max Value"] = str(np.max(data))
                            if ds.dtype.kind == "f":
                                info["Mean Value"] = f"{np.mean(data):.6g}"
                                info["Std Dev"] = f"{np.std(data):.6g}"
                    except Exception:
                        pass
        except:
            return {}
        return info

    def _show_dataset_info_dialog(self, dataset_path: str) -> None:
        """Show a dialog with detailed dataset information.

        Args:
            dataset_path: HDF5 path to the dataset
        """

        info = self._get_dataset_info(dataset_path)

        if info:
            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Dataset Information: {info['Name'].split('/')[-1]}")
            dialog.setMinimumWidth(500)

            layout = QVBoxLayout(dialog)

            # Add header label
            header = QLabel(f"<b>Dataset: {dataset_path}</b>")
            header.setStyleSheet("font-size: 14px; padding: 5px;")
            layout.addWidget(header)

            # Create table for properties
            table = QTableWidget()
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Property", "Value"])
            table.horizontalHeader().setStretchLastSection(True)
            table.setAlternatingRowColors(True)
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            table.setSelectionBehavior(QAbstractItemView.SelectRows)
            table.verticalHeader().setVisible(False)

            # Populate table
            table.setRowCount(len(info))
            for i, (key, value) in enumerate(info.items()):
                # Property name
                key_item = QTableWidgetItem(key)
                if not key.startswith("  "):  # Don't bold sub-items
                    key_item.setFont(QFont(key_item.font().family(), -1, QFont.Bold))
                table.setItem(i, 0, key_item)

                # Property value
                value_item = QTableWidgetItem(str(value))
                table.setItem(i, 1, value_item)

            table.resizeColumnsToContents()
            table.setColumnWidth(0, 180)  # Fixed width for property names

            layout.addWidget(table)

            # Add buttons
            button_box = QHBoxLayout()
            button_box.addStretch()

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.accept)
            close_btn.setDefault(True)
            button_box.addWidget(close_btn)

            layout.addLayout(button_box)

            dialog.exec()

        else:
            pass

    def _merge_file_dialog(self) -> None:
        """Dialog to select and merge another HDF5 file into the current file."""
        if not self.model or not self.model.filepath:
            QMessageBox.information(
                self, "No File", "No HDF5 file is currently loaded. Open or create a file first."
            )
            return

        current_path = Path(self.model.filepath)
        if not current_path.exists():
            QMessageBox.warning(self, "File Not Found", "The current file no longer exists.")
            return

        # Select source file to merge from
        source_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select HDF5 File to Merge",
            str(current_path.parent),
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )

        if not source_file:
            return  # User cancelled

        source_path = Path(source_file)

        # Don't allow merging file with itself
        if source_path.absolute() == current_path.absolute():
            QMessageBox.warning(self, "Invalid Selection", "Cannot merge a file with itself.")
            return

        # Get basic info about source file
        try:
            with h5py.File(str(source_path), "r") as src_h5:
                num_groups = 0
                num_datasets = 0

                def count_items(group):
                    nonlocal num_groups, num_datasets
                    for key in group.keys():
                        if isinstance(group[key], h5py.Group):
                            num_groups += 1
                            count_items(group[key])
                        elif isinstance(group[key], h5py.Dataset):
                            num_datasets += 1

                count_items(src_h5)
        except Exception as exc:
            QMessageBox.critical(self, "Cannot Read File", f"Failed to read source file:\n\n{exc}")
            return

        # Confirm merge operation
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Confirm Merge")
        msg.setText(
            f"Merge contents from:\n{source_path.name}\n\n"
            f"Into current file:\n{current_path.name}\n\n"
            f"Source contains: {num_groups} group(s), {num_datasets} dataset(s)\n\n"
            "Note: If items with the same path exist in both files, "
            "you will be prompted for each conflict."
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setDefaultButton(QMessageBox.Yes)

        if msg.exec() != QMessageBox.Yes:
            return

        # Perform the merge
        try:
            conflicts = []
            items_copied = 0

            # Show progress dialog
            progress = QProgressDialog(
                "Merging files...\nThis may take a moment for large files.",
                "Cancel",
                0,
                num_groups + num_datasets,
                self,
            )
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            QApplication.processEvents()

            items_processed = 0

            with h5py.File(str(source_path), "r") as src_h5:
                with h5py.File(str(current_path), "r+") as dst_h5:

                    def merge_recursively(src_group, dst_group, path="/"):
                        nonlocal items_copied, items_processed, conflicts

                        if progress.wasCanceled():
                            return False

                        for key in src_group.keys():
                            if progress.wasCanceled():
                                return False

                            src_path = posixpath.join(path, key)

                            if isinstance(src_group[key], h5py.Group):
                                items_processed += 1
                                progress.setValue(items_processed)
                                progress.setLabelText(f"Merging: {src_path}")
                                QApplication.processEvents()

                                # Check if group exists
                                if key in dst_group:
                                    if isinstance(dst_group[key], h5py.Group):
                                        # Merge into existing group
                                        if not merge_recursively(
                                            src_group[key], dst_group[key], src_path
                                        ):
                                            return False
                                    else:
                                        # Conflict: target is a dataset
                                        conflicts.append(
                                            f"{src_path} (group conflicts with existing dataset)"
                                        )
                                else:
                                    # Create new group and copy attributes
                                    new_group = dst_group.create_group(key)
                                    for attr_key, attr_val in src_group[key].attrs.items():
                                        try:
                                            new_group.attrs[attr_key] = attr_val
                                        except Exception:
                                            pass
                                    items_copied += 1
                                    # Recurse into subgroup
                                    if not merge_recursively(src_group[key], new_group, src_path):
                                        return False

                            elif isinstance(src_group[key], h5py.Dataset):
                                items_processed += 1
                                progress.setValue(items_processed)
                                progress.setLabelText(f"Merging: {src_path}")
                                QApplication.processEvents()

                                # Check if dataset exists
                                if key in dst_group:
                                    # Ask user what to do
                                    reply = QMessageBox.question(
                                        self,
                                        "Item Exists",
                                        f"Item already exists: {src_path}\n\nOverwrite?",
                                        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                                        QMessageBox.No,
                                    )

                                    if reply == QMessageBox.Cancel:
                                        return False
                                    elif reply == QMessageBox.Yes:
                                        # Delete existing and copy new
                                        del dst_group[key]
                                        src_ds = src_group[key]
                                        dst_group.create_dataset(
                                            key,
                                            data=src_ds[()],
                                            dtype=src_ds.dtype,
                                            compression=src_ds.compression,
                                            compression_opts=src_ds.compression_opts,
                                        )
                                        # Copy attributes
                                        for attr_key, attr_val in src_ds.attrs.items():
                                            try:
                                                dst_group[key].attrs[attr_key] = attr_val
                                            except Exception:
                                                pass
                                        items_copied += 1
                                    else:
                                        # Skip this item
                                        conflicts.append(f"{src_path} (skipped - already exists)")
                                else:
                                    # Copy dataset
                                    src_ds = src_group[key]
                                    dst_group.create_dataset(
                                        key,
                                        data=src_ds[()],
                                        dtype=src_ds.dtype,
                                        compression=src_ds.compression,
                                        compression_opts=src_ds.compression_opts,
                                    )
                                    # Copy attributes
                                    for attr_key, attr_val in src_ds.attrs.items():
                                        try:
                                            dst_group[key].attrs[attr_key] = attr_val
                                        except Exception:
                                            pass
                                    items_copied += 1

                        return True

                    # Copy root attributes if they don't exist
                    for attr_key, attr_val in src_h5.attrs.items():
                        if attr_key not in dst_h5.attrs:
                            try:
                                dst_h5.attrs[attr_key] = attr_val
                            except Exception:
                                pass

                    # Merge contents
                    success = merge_recursively(src_h5, dst_h5)

            progress.close()

            if not success:
                QMessageBox.information(self, "Merge Cancelled", "Merge operation was cancelled.")
                # Still reload to show any partial changes
                self.model.load_file(str(current_path))
                self.tree.expandToDepth(2)
                self._update_file_size_display()
                return

            # Reload the file
            self.model.load_file(str(current_path))
            self.tree.expandToDepth(2)
            self._update_file_size_display()

            # Show results
            result_msg = f"Merge complete!\n\nItems copied: {items_copied}"

            if conflicts:
                result_msg += f"\n\nConflicts/Skipped: {len(conflicts)}"
                if len(conflicts) <= 10:
                    result_msg += "\n" + "\n".join(conflicts)
                else:
                    result_msg += (
                        "\n" + "\n".join(conflicts[:10]) + f"\n... and {len(conflicts) - 10} more"
                    )

            QMessageBox.information(self, "Merge Complete", result_msg)

        except Exception as exc:
            QMessageBox.critical(self, "Merge Failed", f"Failed to merge files:\n\n{exc}")

    # Search/Filter handling
    def _on_search_text_changed(self, text: str) -> None:
        """Handle search text changes and filter the tree view."""
        self._search_pattern = text.strip()
        self._apply_tree_filter()

    def _apply_tree_filter(self) -> None:
        """Apply the search filter to the tree view."""
        if not self._search_pattern:
            # Show all items if search is empty
            self._set_all_items_visible(self.model.invisibleRootItem(), True)
            return

        # Hide all items first
        self._set_all_items_visible(self.model.invisibleRootItem(), False)

        # Show items matching the pattern and their parents
        self._filter_items_recursive(self.model.invisibleRootItem(), self._search_pattern)

    def _set_all_items_visible(self, parent_item, visible: bool) -> None:
        """Recursively set visibility of all items in the tree."""
        for row in range(parent_item.rowCount()):
            child_item = parent_item.child(row, 0)
            if child_item:
                # Get the index for this item
                index = child_item.index()
                self.tree.setRowHidden(index.row(), index.parent(), not visible)
                # Recursively process children
                self._set_all_items_visible(child_item, visible)

    def _filter_items_recursive(self, parent_item, pattern: str) -> bool:
        """Recursively filter items based on glob pattern.

        Returns True if this item or any of its children match the pattern.
        """
        has_visible_child = False

        for row in range(parent_item.rowCount()):
            child_item = parent_item.child(row, 0)
            if not child_item:
                continue

            # Get the item name
            item_name = child_item.text()

            # Check if any children match (recursive)
            child_has_match = self._filter_items_recursive(child_item, pattern)

            # Check if this item matches the pattern
            # If pattern contains a slash, match against the full path from root
            if "/" in pattern:
                # Build the full path for this item
                full_path = self._get_item_path(child_item)
                # Strip leading slash for matching (e.g., "/folder/file.png" -> "folder/file.png")
                if full_path.startswith("/"):
                    full_path = full_path[1:]
                item_matches = fnmatch.fnmatch(full_path.lower(), pattern.lower())
            else:
                # Match against just the item name
                item_matches = fnmatch.fnmatch(item_name.lower(), pattern.lower())

            # Show item if it matches OR if any of its children match
            should_show = item_matches or child_has_match

            # Get the index for this item
            index = child_item.index()
            self.tree.setRowHidden(index.row(), index.parent(), not should_show)

            if should_show:
                has_visible_child = True

        return has_visible_child

    def _get_item_path(self, item) -> str:
        """Build the full path of an item from root to the item."""
        path_parts = []
        current = item
        while current is not None:
            # Skip the root invisible item
            if current.parent() is None:
                break
            path_parts.append(current.text())
            current = current.parent()

        # Reverse to get path from root to item
        path_parts.reverse()
        # Skip the first element (filename) to get HDF5-like path
        if len(path_parts) > 1:
            return "/" + "/".join(path_parts[1:])
        return "/"

    # Selection handling
    def on_selection_changed(self, selected, _deselected) -> None:
        """Handle tree selection changes and update preview.

        Args:
            selected: QItemSelection of newly selected items
            _deselected: QItemSelection of deselected items (unused)
        """
        indexes = selected.indexes()
        if not indexes:
            self._hide_attributes()
            # Clear plot display when nothing is selected
            self._current_csv_group_path = None
            self._saved_plots = []
            self._refresh_saved_plots_list()
            self._clear_plot_display()
            return
        index = indexes[0]
        item = self.model.itemFromIndex(index)
        kind = item.data(self.model.ROLE_KIND)
        path = item.data(self.model.ROLE_PATH)

        if kind == "dataset":
            self.preview_dataset(path)
        elif kind == "attr":
            key = item.data(self.model.ROLE_ATTR_KEY)
            self.preview_attribute(path, key)
        elif kind == "group":
            self.preview_group(path)
        else:
            self.preview_label.setText(str(kind) if kind else "")
            self._set_preview_text("")
            self._hide_attributes()
            # Clear plot display for unknown item types
            self._current_csv_group_path = None
            self._saved_plots = []
            self._refresh_saved_plots_list()
            self._clear_plot_display()

    def _on_tree_item_renamed(
        self, topLeft: QModelIndex, bottomRight: QModelIndex, roles: list[int]
    ) -> None:
        """Handle when a tree item is renamed.

        Update internal references if the currently viewed CSV group was renamed.

        Args:
            topLeft: Top-left index of changed data
            bottomRight: Bottom-right index of changed data
            roles: List of roles that changed
        """
        # Check if this is an edit role change (rename)
        if Qt.EditRole not in roles and Qt.DisplayRole not in roles:
            return

        # Get the renamed item
        item = self.model.itemFromIndex(topLeft)
        if item is None:
            return

        new_path = item.data(self.model.ROLE_PATH)

        # If the currently viewed CSV group was renamed, update the reference
        if self._current_csv_group_path and new_path:
            # Check if we need to update the current CSV group path
            # This could be the group itself or a parent group
            old_path = self._current_csv_group_path

            # Try to determine the old path by checking against the new path
            # The model's setData already updated paths, so we need to check
            # if this rename affects our currently viewed path
            # For now, just clear it to be safe - the user can reselect
            # A more sophisticated approach would track the old path before rename

            # Actually, let's just refresh the selection to update paths correctly
            current_index = self.tree.currentIndex()
            if current_index.isValid():
                self.on_selection_changed(
                    self.tree.selectionModel().selection(), self.tree.selectionModel().selection()
                )

    def add_to_menu(self, menu: QMenu, label: str, icon: str) -> QAction:
        """Helper to add an action to the save menu with given label and icon.

        Args:
            menu: QMenu to add the action to
            label: Text label for the action [e.g., "CSV" will be shown as "CSV..."]
            icon: Name of the icon file (in the icons directory)

        Returns:
            The created QAction
        """

        # all icons are in the "icons" subdirectory
        icon_dir = os.path.join(os.path.dirname(__file__), "icons")
        action = menu.addAction(f"{label}...")
        action.setIcon(QIcon(os.path.join(icon_dir, icon)))
        return action

    # Context menu handling
    def on_tree_context_menu(self, point) -> None:
        """Handle context menu requests on tree items.

        Args:
            point: QPoint position where the context menu was requested
        """
        index = self.tree.indexAt(point)
        if not index.isValid():
            return
        # Always act on column 0 item for role data
        index0 = index.siblingAtColumn(0)
        item = self.model.itemFromIndex(index0)
        if item is None:
            return
        kind = item.data(self.model.ROLE_KIND)
        path = item.data(self.model.ROLE_PATH)
        attr_key = item.data(self.model.ROLE_ATTR_KEY)

        # Check if this is a CSV group
        is_csv_group = False
        csv_expanded = False
        if kind == "group" and path and self.model.filepath:
            try:
                with h5py.File(self.model.filepath, "r") as h5:
                    grp = h5[path]
                    if isinstance(grp, h5py.Group):
                        if "source_type" in grp.attrs and grp.attrs["source_type"] == "csv":
                            is_csv_group = True
                            csv_expanded = item.data(self.model.ROLE_CSV_EXPANDED) or False
            except Exception:  # noqa: BLE001
                pass

        # Determine if deletable and label
        deletable = False
        label = None
        if kind == "dataset":
            deletable = True
            label = f"Delete dataset '{item.text()}'"
        elif kind == "group":
            # Don't allow deleting the file root
            if path and path != "/":
                deletable = True
                label = f"Delete group '{item.text()}'"
        elif kind == "attr":
            deletable = True
            label = f"Delete attribute '{attr_key}'"

        menu = QMenu(self)
        style = self.style()

        # Add dataset information option
        act_info = None
        act_show_dag_dataset = None
        if kind == "dataset":
            act_info = menu.addAction(f"Dataset Information...")
            act_info.setIcon(style.standardIcon(QStyle.SP_MessageBoxInformation))
        elif kind == "group":
            act_show_dag_dataset = menu.addAction("Show DAG for this group...")
            act_show_dag_dataset.setIcon(style.standardIcon(QStyle.SP_FileDialogContentsView))
            act_show_dag_dataset.setIcon(style.standardIcon(QStyle.SP_FileDialogListView))

        # Add CSV group expand/collapse option
        act_toggle_csv = None
        act_save_csv = None
        act_save_excel = None
        act_save_json = None
        act_save_html = None
        act_save_latex = None
        act_save_markdown = None
        if is_csv_group:
            if csv_expanded:
                act_toggle_csv = menu.addAction("Hide Internal Structure")
                act_toggle_csv.setIcon(style.standardIcon(QStyle.SP_FileDialogDetailedView))
            else:
                act_toggle_csv = menu.addAction("Show Internal Structure")
                act_toggle_csv.setIcon(style.standardIcon(QStyle.SP_DirIcon))

            # Add "Save as..." submenu with export options
            save_menu = menu.addMenu("Save as...")
            save_menu.setIcon(style.standardIcon(QStyle.SP_DialogSaveButton))
            # file export options:
            act_save_csv = self.add_to_menu(save_menu, "CSV", "icon_csv.png")
            act_save_excel = self.add_to_menu(save_menu, "Excel", "icon_excel.png")
            act_save_json = self.add_to_menu(save_menu, "JSON", "icon_json.png")
            save_menu.addSeparator()
            act_save_html = self.add_to_menu(save_menu, "HTML", "icon_html.png")
            act_save_latex = self.add_to_menu(save_menu, "LaTeX", "icon_latex.png")
            act_save_markdown = self.add_to_menu(save_menu, "Markdown", "icon_markdown.png")
            menu.addSeparator()

        act_delete = None
        if deletable and label:
            act_delete = menu.addAction(label)
            act_delete.setIcon(style.standardIcon(QStyle.SP_TrashIcon))

        # If no actions available, don't show menu
        if (
            not act_info
            and not act_toggle_csv
            and not act_save_csv
            and not act_save_excel
            and not act_save_json
            and not act_save_html
            and not act_save_latex
            and not act_save_markdown
            and not act_delete
            and not act_show_dag_dataset
        ):
            return

        global_pos = self.tree.viewport().mapToGlobal(point)
        chosen = menu.exec(global_pos)

        # mapping of actions to format strings
        save_as_formats = {
            act_save_csv: "csv",
            act_save_excel: "xlsx",
            act_save_json: "json",
            act_save_html: "html",
            act_save_latex: "tex",
            act_save_markdown: "md",
        }

        if chosen and chosen == act_info and act_info is not None:
            self._show_dataset_info_dialog(path)
        elif chosen == act_show_dag_dataset and act_show_dag_dataset is not None:
            self._show_dag_visualization_pyqtgraph(path)
        elif chosen == act_toggle_csv:
            self.model.toggle_csv_group_expansion(item)
        elif chosen == act_delete:
            # Confirm destructive action
            target_desc = label.replace("Delete ", "") if label else "item"
            resp = QMessageBox.question(
                self,
                "Confirm delete",
                f"Are you sure you want to delete {target_desc}?\n\nThis will modify the HDF5 file and cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp == QMessageBox.Yes:
                self._perform_delete(kind, path, attr_key)
        elif chosen in save_as_formats:
            self._save_csv_group_as(path, format=save_as_formats[chosen])
        else:
            # No valid action selected (e.g., menu closed), do nothing
            pass

    def _save_csv_group_as(self, csv_group_path: str, format: str = "csv") -> None:
        """Save a CSV group to a file using a save dialog.

        Args:
            csv_group_path: HDF5 path to the CSV group
            format: Export format - "csv", "xlsx", "json", "html", "tex", or "md"
        """
        if not self.model or not self.model.filepath:
            QMessageBox.warning(self, "No file", "No HDF5 file is loaded.")
            return

        try:
            with h5py.File(self.model.filepath, "r") as h5:
                if csv_group_path not in h5:
                    QMessageBox.warning(
                        self, "Not found", f"Path '{csv_group_path}' not found in file."
                    )
                    return

                group = h5[csv_group_path]
                if not isinstance(group, h5py.Group):
                    QMessageBox.warning(self, "Invalid", "Selected item is not a group.")
                    return

                # Determine default filename
                source_file = group.attrs.get("source_file")
                if isinstance(source_file, (bytes, bytearray)):
                    try:
                        source_file = source_file.decode("utf-8")
                    except Exception:  # noqa: BLE001
                        source_file = None

                if isinstance(source_file, str) and source_file.lower().endswith(".csv"):
                    # Replace .csv extension with the target format
                    default_name = source_file[:-4] + f".{format}"
                else:
                    default_name = (os.path.basename(csv_group_path) or "export") + f".{format}"

                # Configure dialog based on format
                if format == "json":
                    dialog_title = "Save JSON File"
                    file_filter = "JSON Files (*.json);;All Files (*)"
                elif format == "html":
                    dialog_title = "Save HTML File"
                    file_filter = "HTML Files (*.html);;All Files (*)"
                elif format == "xlsx":
                    dialog_title = "Save Excel File"
                    file_filter = "Excel Files (*.xlsx);;All Files (*)"
                elif format == "tex":
                    dialog_title = "Save LaTeX File"
                    file_filter = "LaTeX Files (*.tex);;All Files (*)"
                elif format == "md":
                    dialog_title = "Save Markdown File"
                    file_filter = "Markdown Files (*.md);;All Files (*)"
                else:
                    dialog_title = "Save CSV File"
                    file_filter = "CSV Files (*.csv);;All Files (*)"

                # Show save dialog
                save_path, _ = QFileDialog.getSaveFileName(
                    self, dialog_title, default_name, file_filter
                )

                if not save_path:
                    return

                # Get filtered indices and visible columns if this is the currently displayed CSV
                filtered_indices = None
                visible_columns = None
                if csv_group_path == self._current_csv_group_path:
                    filtered_indices = self.model.get_csv_filtered_indices(csv_group_path)
                    # Get the visible columns in their current visual order
                    if hasattr(self, "_csv_visible_columns") and self._csv_visible_columns:
                        visible_columns = self._csv_visible_columns

                # Temporarily set visible columns in model for export
                if visible_columns:
                    self.model.set_csv_visible_columns(csv_group_path, visible_columns)

                # Get DataFrame directly from model (sorting is applied automatically via stored sort_specs)
                df = self.model._reconstruct_csv_tempfile(
                    group, csv_group_path, filtered_indices, return_dataframe=True
                )
                if df is None:
                    QMessageBox.warning(self, "Export Failed", "Failed to reconstruct CSV data.")
                    return

                # Export based on format
                try:
                    if format == "json":
                        df.to_json(save_path, orient="records", indent=2)
                        status_msg = f"Saved JSON to {save_path}"
                    elif format == "html":
                        df.to_html(save_path, index=False, border=1, justify="left")
                        status_msg = f"Saved HTML to {save_path}"
                    elif format == "xlsx":
                        df.to_excel(save_path, index=False)
                        status_msg = f"Saved Excel to {save_path}"
                    elif format == "tex":
                        df.to_latex(save_path, index=False)
                        status_msg = f"Saved LaTeX to {save_path}"
                    elif format == "md":
                        df.to_markdown(save_path, index=False)
                        status_msg = f"Saved Markdown to {save_path}"
                    else:
                        # Export as CSV
                        df.to_csv(save_path, index=False)
                        status_msg = f"Saved CSV to {save_path}"
                except Exception as exc:
                    QMessageBox.warning(
                        self, "Export Failed", f"Failed to export as {format.upper()}: {exc}"
                    )
                    return

                self.statusBar().showMessage(status_msg, 5000)

        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save {format.upper()}: {exc}")

    def _perform_delete(self, kind: str, path: str, attr_key: str | None) -> None:
        """Delete an HDF5 item (dataset, group, or attribute).

        Args:
            kind: Type of item to delete ('dataset', 'group', or 'attr')
            path: HDF5 path to the item or its owner (for attributes)
            attr_key: Attribute key name if kind is 'attr', None otherwise
        """
        fpath = self.model.filepath
        if not fpath:
            QMessageBox.warning(self, "No file", "No HDF5 file is loaded.")
            return
        try:
            with h5py.File(fpath, "r+") as h5:
                if kind == "dataset":
                    # Deleting a dataset link by absolute path
                    del h5[path]
                elif kind == "group":
                    if path == "/":
                        raise ValueError("Cannot delete the root group")
                    del h5[path]
                elif kind == "attr":
                    if attr_key is None:
                        raise ValueError("Missing attribute key")
                    # For attributes, 'path' is the group/dataset owner path
                    owner = h5[path]
                    del owner.attrs[attr_key]
                else:
                    raise ValueError(f"Unsupported kind: {kind}")
        except Exception as exc:
            QMessageBox.critical(self, "Delete failed", f"Could not delete: {exc}")
            return

        # Refresh model to reflect changes
        try:
            self.model.load_file(fpath)
            self.tree.expandToDepth(1)

            # Update file size display after deletion
            self._update_file_size_display()
        except Exception as exc:
            QMessageBox.warning(
                self, "Refresh failed", f"Deleted, but failed to refresh view: {exc}"
            )

    def preview_dataset(self, dspath: str) -> None:
        """Preview an HDF5 dataset in the preview pane.

        Args:
            dspath: HDF5 path to the dataset
        """
        # Clear plot display and saved plots when viewing a dataset
        self._current_csv_group_path = None
        self._saved_plots = []
        self._refresh_saved_plots_list()
        self._clear_plot_display()

        self.preview_label.setText(f"Dataset: {os.path.basename(dspath)}")
        fpath = self.model.filepath
        if not fpath:
            self._set_preview_text("No file loaded")
            self.preview_edit.setVisible(True)
            self.preview_image.setVisible(False)
            self._hide_attributes()
            return
        # If the dataset name is an image format, try to display as image
        image_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
            ".ico",
        )
        if dspath.lower().endswith(image_extensions):
            try:
                with h5py.File(fpath, "r") as h5:
                    obj = h5[dspath]
                    if not isinstance(obj, h5py.Dataset):
                        self._set_preview_text("Selected path is not a dataset.")
                        self.preview_edit.setVisible(True)
                        self.preview_image.setVisible(False)
                        self._hide_attributes()
                        return
                    # Read raw bytes from dataset
                    data = obj[()]

                    # Check if this is compressed binary data
                    if "compressed" in obj.attrs and obj.attrs["compressed"] == "gzip":
                        encoding = obj.attrs.get("original_encoding", "utf-8")
                        if isinstance(encoding, bytes):
                            encoding = encoding.decode("utf-8")
                        if (
                            encoding == "binary"
                            and isinstance(data, np.ndarray)
                            and data.dtype == np.uint8
                        ):
                            # Decompress the binary data
                            compressed_bytes = data.tobytes()
                            img_bytes = gzip.decompress(compressed_bytes)
                        elif isinstance(data, bytes):
                            img_bytes = data
                        elif hasattr(data, "tobytes"):
                            img_bytes = data.tobytes()
                        else:
                            self._set_preview_text("Dataset is not a valid image byte array.")
                            self.preview_edit.setVisible(True)
                            self.preview_image.setVisible(False)
                            self._hide_attributes()
                            return
                    elif isinstance(data, bytes):
                        img_bytes = data
                    elif hasattr(data, "tobytes"):
                        img_bytes = data.tobytes()
                    else:
                        self._set_preview_text("Dataset is not a valid image byte array.")
                        self.preview_edit.setVisible(True)
                        self.preview_image.setVisible(False)
                        self._hide_attributes()
                        return
                    pixmap = QPixmap()
                    # QPixmap.loadFromData will auto-detect the format
                    if pixmap.loadFromData(img_bytes):
                        # Scale pixmap to fit preview area, maintaining aspect ratio
                        self._show_scaled_image(pixmap)
                        self.preview_image.setVisible(True)
                        self.preview_edit.setVisible(False)
                        self.preview_table.setVisible(False)
                        self.filter_panel.setVisible(False)
                        # Show attributes for the dataset
                        self._show_attributes(obj)
                    else:
                        self._set_preview_text("Failed to load image from dataset.")
                        self.preview_edit.setVisible(True)
                        self.preview_image.setVisible(False)
                        self._hide_attributes()
            except Exception as exc:
                self._set_preview_text(f"Error reading image dataset:\n{exc}")
                self.preview_edit.setVisible(True)
                self.preview_image.setVisible(False)
                self._hide_attributes()
            return
        # Otherwise, show text preview for non-image datasets
        try:
            with h5py.File(fpath, "r") as h5:
                obj = h5[dspath]
                if not isinstance(obj, h5py.Dataset):
                    self._set_preview_text("Selected path is not a dataset.")
                    self.preview_edit.setVisible(True)
                    self.preview_image.setVisible(False)
                    self._hide_attributes()
                    return
                ds = obj
                text, note = dataset_to_text(ds, limit_bytes=1_000_000)
                if note:
                    note = note.strip("()")  # Remove parentheses from note
                    # add the note after the file name:
                    self.preview_label.setText(f"Dataset: {os.path.basename(dspath)} ({note})")
                # Apply syntax highlighting based on file extension
                language = get_language_from_path(dspath)
                self._set_preview_text(text=text, language=language)
                self.preview_edit.setVisible(True)
                self.preview_image.setVisible(False)
                # Show attributes for the dataset
                self._show_attributes(ds)
        except Exception as exc:
            self._set_preview_text(f"Error reading dataset:\n{exc}")
            self.preview_edit.setVisible(True)
            self.preview_image.setVisible(False)
            self._hide_attributes()

    def _set_preview_text(self, text: str, language: str = "plain") -> None:
        """Set preview text with optional syntax highlighting.

        Args:
            text: The text content to display
            language: Language identifier for syntax highlighting (default: "plain")
        """
        # Remove old highlighter if exists
        if self._current_highlighter is not None:
            self._current_highlighter.setDocument(None)
            self._current_highlighter = None

        # Set the text
        self.preview_edit.setPlainText(text)

        # Apply syntax highlighting if not plain text
        if language != "plain":
            try:
                self._current_highlighter = SyntaxHighlighter(
                    self.preview_edit.document(), language=language
                )
            except Exception:  # noqa: BLE001
                # If highlighting fails, just show plain text
                pass

        # Show text view, hide table and image
        self.preview_edit.setVisible(True)
        self.preview_table.setVisible(False)
        self.preview_image.setVisible(False)
        self.filter_panel.setVisible(False)

    def _show_attributes(self, h5_obj) -> None:
        """Display attributes of an HDF5 object in the attributes table.

        Args:
            h5_obj: HDF5 group or dataset object with attributes
        """
        try:
            attrs = dict(h5_obj.attrs)
            if attrs:
                self.attrs_table.setRowCount(len(attrs))
                for row, (key, value) in enumerate(attrs.items()):
                    # Attribute name
                    name_item = QTableWidgetItem(str(key))
                    self.attrs_table.setItem(row, 0, name_item)
                    # Attribute value (convert to string)
                    try:
                        if isinstance(value, (np.ndarray, list, tuple)):
                            # For arrays/lists, show truncated representation
                            value_str = repr(value)
                            if len(value_str) > 200:
                                value_str = value_str[:200] + "..."
                        else:
                            value_str = str(value)
                    except Exception:  # noqa: BLE001
                        value_str = repr(value)
                    value_item = QTableWidgetItem(value_str)
                    self.attrs_table.setItem(row, 1, value_item)
                # Resize columns to content
                self.attrs_table.resizeColumnsToContents()
            else:
                # No attributes, clear the table
                self.attrs_table.setRowCount(0)
        except Exception:  # noqa: BLE001
            # If there's an error, just clear the attributes table
            self.attrs_table.setRowCount(0)

    def _hide_attributes(self) -> None:
        """Hide the attributes table."""
        self.attrs_table.setRowCount(0)

    def _show_scaled_image(self, pixmap=None):
        """Display a scaled image in the preview pane.

        Args:
            pixmap: QPixmap to display, or None to use stored pixmap
        """
        # Use the provided pixmap or the stored one
        if pixmap is not None:
            self._original_pixmap = pixmap
        pixmap = self._original_pixmap
        if pixmap is None:
            return
        label_size = self.preview_image.size()
        if label_size.width() < 10 or label_size.height() < 10:
            label_size = self.preview_image.parentWidget().size()
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_image.setPixmap(scaled)

    def updateCanvas(self):
        """If plot canvas is visible, apply tight layout and redraw"""
        if hasattr(self, "plot_canvas") and self.plot_canvas.isVisible():
            try:
                self.plot_figure.tight_layout()
                self.plot_canvas.draw()
            except Exception:
                pass  # Ignore layout errors during resize

    def resizeEvent(self, event):
        """Handle window resize events to rescale displayed images and adjust plot layout.

        Args:
            event: QResizeEvent object
        """
        # If an image is visible, rescale it to fit the new size
        if self.preview_image.isVisible():
            self._show_scaled_image()

        self.updateCanvas()

        super().resizeEvent(event)

    def _on_splitter_moved(self, pos: int, index: int) -> None:
        """Handle splitter movement to adjust plot layout.

        Args:
            pos: New position of the splitter handle
            index: Index of the splitter handle that was moved
        """

        self.updateCanvas()

    def preview_attribute(self, grouppath: str, key: str) -> None:
        """Preview an HDF5 attribute value.

        Args:
            grouppath: HDF5 path to the group or dataset containing the attribute
            key: Attribute key name
        """
        self.preview_label.setText(f"Attribute: {grouppath}@{key}")
        fpath = self.model.filepath
        if not fpath:
            self._set_preview_text("No file loaded")
            self._hide_attributes()
            return
        try:
            with h5py.File(fpath, "r") as h5:
                g = h5[grouppath]
                val = g.attrs[key]
                self._set_preview_text(repr(val))
                self._hide_attributes()
        except Exception as exc:
            self._set_preview_text(f"Error reading attribute:\n{exc}")
            self._hide_attributes()

    def preview_group(self, grouppath: str) -> None:
        """Preview a group. If it's a CSV-derived group, show as table."""
        self.preview_label.setText(f"Group: {grouppath}")
        fpath = self.model.filepath
        if not fpath:
            self._set_preview_text("No file loaded")
            self._hide_attributes()
            return

        try:
            with h5py.File(fpath, "r") as h5:
                grp = h5[grouppath]
                if not isinstance(grp, h5py.Group):
                    self._set_preview_text("(Not a group)")
                    self._hide_attributes()
                    return

                # Check if this is a CSV-derived group
                if "source_type" in grp.attrs and grp.attrs["source_type"] == "csv":
                    # Track current CSV group for plotting
                    self._current_csv_group_path = grouppath
                    self._show_csv_table(grp)
                    self._update_plot_action_enabled()
                else:
                    self._current_csv_group_path = None
                    self._saved_plots = []
                    self._refresh_saved_plots_list()
                    self._clear_plot_display()
                    self._set_preview_text("(No content to display)")
                    # Show attributes for the group
                    self._show_attributes(grp)
        except Exception as exc:
            self._set_preview_text(f"Error reading group:\n{exc}")
            self._hide_attributes()

    def _get_th_location(self, ds_key, grp):
        """Get the location of a dataset, checking for optional 'Time History' subgroup.

        Args:
            ds_key: Dataset key name to look for
            grp: HDF5 group to search in

        Returns:
            Tuple of (key_in_group: bool, th_grp: h5py.Group)
        """
        OPTIONAL_GROUP_FOR_COLUMNS = "Time History"
        th_group = OPTIONAL_GROUP_FOR_COLUMNS in grp
        if th_group:
            key_in_group = ds_key in grp[OPTIONAL_GROUP_FOR_COLUMNS]
            th_grp = grp[OPTIONAL_GROUP_FOR_COLUMNS]
        else:
            key_in_group = ds_key in grp
            th_grp = grp

        return key_in_group, th_grp

    def _show_csv_table(self, grp: h5py.Group) -> None:
        """Display CSV-derived group data in a table widget."""
        progress = None
        try:
            # Get column names (for headers)
            if "column_names" in grp.attrs:
                try:
                    col_names = [str(c) for c in list(grp.attrs["column_names"])]
                except Exception:
                    col_names = list(grp.keys())
            else:
                col_names = list(grp.keys())

            # Optional mapping of columns to actual dataset names
            col_ds_names = None
            if "column_dataset_names" in grp.attrs:
                try:
                    col_ds_names = [str(c) for c in list(grp.attrs["column_dataset_names"])]
                    if len(col_ds_names) != len(col_names):
                        col_ds_names = None
                except Exception:
                    col_ds_names = None

            # Estimate total work for progress
            total_cols = len(col_names)
            # Create progress dialog
            progress = self._create_progress_dialog("Loading CSV metadata...")

            # First pass: only get metadata (dataset paths and row counts) - don't load data yet
            dataset_info = {}  # Maps column name to (ds_key, th_grp_path, row_count, dtype)
            max_rows = 0
            for idx, col_name in enumerate(col_names):
                if progress.wasCanceled():
                    progress.close()
                    self._set_preview_text("(CSV display cancelled)")
                    return

                # Update progress
                progress_val = int((idx / total_cols) * 30)
                progress.setValue(progress_val)
                progress.setLabelText(f"Reading metadata {idx + 1}/{total_cols}: {col_name}")
                QApplication.processEvents()

                # Resolve dataset key for this column
                ds_key = None
                if col_ds_names is not None:
                    ds_key = col_ds_names[idx]
                else:
                    # Try sanitized version of the column name
                    cand = sanitize_hdf5_name(str(col_name))
                    if cand in grp:
                        ds_key = cand
                    elif col_name in grp:
                        ds_key = col_name

                key_in_group, th_grp = self._get_th_location(ds_key, grp)
                if ds_key and key_in_group:
                    ds = th_grp[ds_key]
                    if isinstance(ds, h5py.Dataset):
                        # Only get shape and dtype, don't load data
                        if len(ds.shape) > 0:
                            row_count = ds.shape[0]
                        else:
                            row_count = 1
                        # Store group path as string instead of group object
                        th_grp_path = th_grp.name
                        dataset_info[col_name] = (ds_key, th_grp_path, row_count, ds.dtype)
                        max_rows = max(max_rows, row_count)

            if not dataset_info:
                progress.close()
                self._set_preview_text("(No datasets found in CSV group)")
                return

            # Reset all CSV state for new group
            self._csv_dataset_info = dataset_info  # For lazy loading
            self._csv_data_dict = {}  # Will be populated on-demand
            self._csv_column_names = col_names
            self._csv_total_rows = max_rows
            self._csv_sort_specs = []  # Initialize sort specs
            self._csv_visible_columns = col_names.copy()
            self._csv_filtered_indices = None

            # Lazy load initial batch of data
            initial_batch = min(self._table_batch_size, max_rows)
            progress.setLabelText(f"Loading initial {initial_batch} rows...")
            progress.setValue(50)
            QApplication.processEvents()

            # Lazy load columns as needed for initial batch
            # Need to access file separately since grp is from outer context
            fpath = self.model.filepath
            if fpath:
                with h5py.File(fpath, "r") as h5:
                    self._lazy_load_columns(col_names, 0, initial_batch, h5)
            self._table_loaded_rows = initial_batch

            # Defensive: ensure visible columns and indices are valid
            valid_columns = [c for c in self._csv_visible_columns if c in self._csv_column_names]
            if not valid_columns:
                valid_columns = self._csv_column_names.copy()
            self._csv_visible_columns = valid_columns
            # Defensive: ensure filtered indices are valid
            if self._csv_filtered_indices is None or len(self._csv_filtered_indices) == 0:
                self._csv_filtered_indices, _, _ = self._get_filtered_sorted_indices(
                    self._csv_data_dict, self._csv_filters, self._csv_sort_specs
                )
            if self._csv_filtered_indices is None or len(self._csv_filtered_indices) == 0:
                self._csv_filtered_indices = np.arange(self._csv_total_rows)

            # Create and set the CSVTableModel for QTableView
            model = CSVTableModel(
                data_dict=self._csv_data_dict,
                col_names=self._csv_visible_columns,
                row_indices=self._csv_filtered_indices,
                parent=self.preview_table,
            )
            self.preview_table.setModel(model)
            self._csv_table_model = model
            # Ensure column selection via header is enabled for plotting
            self.preview_table.setSelectionBehavior(QAbstractItemView.SelectColumns)
            self.preview_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.preview_table.horizontalHeader().setSectionsClickable(True)
            self.preview_table.horizontalHeader().setHighlightSections(True)
            # Enable right-click context menu on column header
            header = self.preview_table.horizontalHeader()
            header.setContextMenuPolicy(Qt.CustomContextMenu)
            header.customContextMenuRequested.connect(self._on_column_header_context_menu)
            # Make last column resizable
            header.setStretchLastSection(False)
            # Connect selection change to plot button update
            self.preview_table.selectionModel().selectionChanged.connect(
                lambda selected, deselected: self._update_plot_action_enabled()
            )

            # Set uniform row heights for large datasets
            if max_rows > 1000:
                self.preview_table.verticalHeader().setDefaultSectionSize(24)
                self.preview_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)

            # For large datasets, set fixed column widths
            if max_rows > 10000 or len(col_names) > 50:
                progress.setLabelText("Setting column widths...")
                progress.setValue(95)
                QApplication.processEvents()
                model = self.preview_table.model()
                if model:
                    for col_idx in range(model.columnCount()):
                        self.preview_table.setColumnWidth(col_idx, 120)
            else:
                progress.setLabelText("Resizing columns...")
                progress.setValue(95)
                QApplication.processEvents()
                self.preview_table.resizeColumnsToContents()

            # Show table, hide others
            self.preview_table.setVisible(True)
            self.preview_edit.setVisible(False)
            self.preview_image.setVisible(False)
            # Show filter panel for CSV tables
            self.filter_panel.setVisible(True)
            # Show attributes for the CSV group
            self._show_attributes(grp)

            # Connect vertical scrollbar to dynamic row loading
            self.preview_table.verticalScrollBar().valueChanged.connect(self._on_table_scroll)

            if max_rows > initial_batch:
                self.statusBar().showMessage(
                    f"Loaded {initial_batch:,} of {max_rows:,} rows (more will load as you scroll)",
                    8000,
                )

            # Enable/disable plotting action depending on visibility/selection
            self._update_plot_action_enabled()
            progress.close()

            # Load filters, sort, and visible columns from HDF5
            saved_filters = self._load_filters_from_hdf5(grp)
            if saved_filters:
                self._csv_filters = saved_filters
                self.statusBar().showMessage(
                    f"Loaded {len(saved_filters)} saved filter(s) from HDF5 file", 5000
                )
            else:
                self._csv_filters = []

            # Load saved sort from HDF5 group (or clear if none exist)
            saved_sort = self._load_sort_from_hdf5(grp)
            if saved_sort:
                self._csv_sort_specs = saved_sort
                self.btn_clear_sort.setEnabled(True)
                self.statusBar().showMessage(
                    f"Loaded sort by {len(saved_sort)} column(s) from HDF5 file", 5000
                )
            else:
                self._csv_sort_specs = []
                self.btn_clear_sort.setEnabled(False)

            # Load saved column visibility from HDF5 group
            saved_visible_columns = self._load_visible_columns_from_hdf5(grp)
            if saved_visible_columns:
                self._csv_visible_columns = saved_visible_columns
                self.statusBar().showMessage(
                    f"Loaded column visibility ({len(saved_visible_columns)}/{len(col_names)} columns) from HDF5 file",
                    5000,
                )
            else:
                # Default to all columns visible
                self._csv_visible_columns = col_names.copy()

            # Always apply column visibility to ensure correct state
            self._apply_column_visibility()

            # Update model with visible columns for drag-and-drop export
            if self._current_csv_group_path and self.model:
                self.model.set_csv_visible_columns(
                    self._current_csv_group_path, self._csv_visible_columns
                )
                # Also update sort specs in model
                self.model.set_csv_sort_specs(self._current_csv_group_path, self._csv_sort_specs)

            self._load_plot_configs_from_hdf5(grp)

            # Apply filters and sort if present, else show all rows
            if self._csv_filters:
                self._apply_filters()
            else:
                # No filters - all rows are visible, but still need to apply sorting if any
                self.filter_status_label.setText("No filters applied")
                self.btn_clear_filters.setEnabled(False)
                if self._csv_sort_specs:
                    # Apply sorting even without filters
                    self._apply_filters()  # This will apply sorting to all rows
                else:
                    # No filters and no sorting - simple case
                    self._csv_filtered_indices = np.arange(max_rows)
                    # Notify model that no filtering is active
                    if self._current_csv_group_path and self.model:
                        self.model.set_csv_filtered_indices(self._current_csv_group_path, None)

            if self._csv_sort_specs and not self._csv_filters:
                self._apply_sort()

            self.preview_table.verticalScrollBar().setValue(0)

        except Exception as exc:
            if progress:
                progress.close()
            self._set_preview_text(f"Error displaying CSV table:\n{exc}")
            self.preview_table.setVisible(False)
            self.preview_edit.setVisible(True)
            self.preview_image.setVisible(False)
            self._hide_attributes()

    def _ensure_all_data_loaded(self) -> None:
        """Ensure all CSV data is loaded (used before filtering/sorting/plotting)."""
        if not self._csv_dataset_info or not self._current_csv_group_path:
            return

        if not self.model or not self.model.filepath:
            return

        # Check if we need to load more data
        total_rows = getattr(self, "_csv_total_rows", 0)
        if total_rows == 0:
            return

        # Check if all columns are fully loaded
        all_loaded = True
        for col_name in self._csv_column_names:
            if (
                col_name not in self._csv_data_dict
                or len(self._csv_data_dict[col_name]) < total_rows
            ):
                all_loaded = False
                break

        if all_loaded:
            return  # Already loaded

        # Load remaining data with progress dialog
        progress = self._create_progress_dialog(
            "Loading all CSV data for operation...", min_duration=200
        )

        try:
            with h5py.File(self.model.filepath, "r") as h5:
                if self._current_csv_group_path in h5:
                    grp = h5[self._current_csv_group_path]
                    if isinstance(grp, h5py.Group):
                        # Load all columns completely
                        progress.setLabelText("Loading complete dataset...")
                        progress.setValue(10)
                        QApplication.processEvents()

                        self._lazy_load_columns(self._csv_column_names, 0, total_rows, h5)

                        progress.setValue(100)
        except Exception as exc:
            self.statusBar().showMessage(f"Error loading data: {exc}", 5000)
        finally:
            progress.close()

    def _lazy_load_columns(
        self, col_names: list[str], start_row: int, end_row: int, h5: h5py.File
    ) -> None:
        """Lazy load only the columns and rows needed from HDF5.

        Args:
            col_names: List of column names to load
            start_row: Starting row index
            end_row: Ending row index (exclusive)
            h5: Open HDF5 file object
        """
        for col_name in col_names:
            # Check if we need to load or extend this column
            if col_name in self._csv_data_dict:
                existing_data = self._csv_data_dict[col_name]
                # If already fully loaded for this range, skip
                if len(existing_data) >= end_row:
                    continue
                # Otherwise we need to extend - adjust start_row
                actual_start = len(existing_data)
            else:
                actual_start = 0  # Load from beginning

            # Load this column's data for the requested range
            if col_name in self._csv_dataset_info:
                ds_key, th_grp_path, row_count, dtype = self._csv_dataset_info[col_name]
                # Get the group from the file using the stored path
                th_grp = h5[th_grp_path]
                ds = th_grp[ds_key]

                if isinstance(ds, h5py.Dataset):
                    # Determine the actual slice to load
                    load_start = max(actual_start, start_row)
                    load_end = min(end_row, row_count)

                    if load_start >= load_end:
                        continue  # Nothing to load

                    # Load only the slice we need
                    if len(ds.shape) > 0:
                        data = ds[load_start:load_end]
                    else:
                        data = ds[()]

                    if isinstance(data, np.ndarray):
                        # Decode byte strings to UTF-8 strings for display
                        if data.dtype.kind == "S":
                            # Byte strings - decode to UTF-8
                            data = np.array(
                                [
                                    v.decode("utf-8", errors="replace")
                                    if isinstance(v, bytes)
                                    else str(v)
                                    for v in data
                                ],
                                dtype=object,
                            )
                        elif data.dtype.kind == "O":
                            # Object dtype - could be mixed, handle bytes if present
                            data = np.array(
                                [
                                    v.decode("utf-8", errors="replace")
                                    if isinstance(v, bytes)
                                    else v
                                    for v in data
                                ],
                                dtype=object,
                            )

                        # Store or concatenate the data
                        if col_name in self._csv_data_dict:
                            existing_data = self._csv_data_dict[col_name]
                            # If there's a gap between existing and new data, pad with empty strings
                            if load_start > len(existing_data):
                                gap = np.array(
                                    [""] * (load_start - len(existing_data)), dtype=object
                                )
                                self._csv_data_dict[col_name] = np.concatenate(
                                    [existing_data, gap, data]
                                )
                            else:
                                self._csv_data_dict[col_name] = np.concatenate(
                                    [existing_data, data]
                                )
                        else:
                            # First load for this column
                            if load_start > 0:
                                # Need to pad the beginning if not starting from row 0
                                padding = np.array([""] * load_start, dtype=object)
                                self._csv_data_dict[col_name] = np.concatenate([padding, data])
                            else:
                                self._csv_data_dict[col_name] = data
                    else:
                        # Scalar dataset
                        if isinstance(data, bytes):
                            data = data.decode("utf-8", errors="replace")
                        self._csv_data_dict[col_name] = np.array([data], dtype=object)

    def _on_table_scroll(self, value: int) -> None:
        """Handle table scroll events to load more rows as needed.

        Args:
            value: Scroll position value
        """
        if self._table_is_loading:
            return  # Already loading, skip

        if not hasattr(self, "_csv_total_rows") or not hasattr(self, "_csv_data_dict"):
            return  # No CSV data loaded

        # Check if we need to load more rows
        if self._table_loaded_rows >= self._csv_total_rows:
            return  # All rows already loaded

        # Get visible range
        scrollbar = self.preview_table.verticalScrollBar()
        max_value = scrollbar.maximum()
        if max_value == 0:
            return

        # Calculate approximate visible row based on scroll position
        scroll_ratio = value / max_value
        approx_visible_row = int(scroll_ratio * self._csv_total_rows)

        # Add buffer rows above and below visible area
        buffer_rows = 500
        target_row = min(approx_visible_row + buffer_rows, self._csv_total_rows)

        # If the target row is beyond what we've loaded, load up to that point
        if target_row > self._table_loaded_rows:
            self._load_rows_up_to(target_row)

    def _load_rows_up_to(self, target_row: int) -> None:
        """Load all rows from current position up to target_row.

        Args:
            target_row: Target row index to load up to
        """
        if self._table_is_loading:
            return

        if self._table_loaded_rows >= target_row:
            return

        self._table_is_loading = True

        try:
            # Load all rows from current position to target in one go
            start_row = self._table_loaded_rows
            end_row = min(target_row, self._csv_total_rows)

            # Disable updates during batch load for better performance
            self.preview_table.setUpdatesEnabled(False)

            # Lazy load the data for this range if needed
            if self._current_csv_group_path and self.model and self.model.filepath:
                try:
                    with h5py.File(self.model.filepath, "r") as h5:
                        if self._current_csv_group_path in h5:
                            grp = h5[self._current_csv_group_path]
                            if isinstance(grp, h5py.Group):
                                self._lazy_load_columns(
                                    self._csv_column_names, start_row, end_row, h5
                                )
                except Exception as exc:
                    self.statusBar().showMessage(f"Error loading data: {exc}", 5000)

                # model refresh
                if self._csv_table_model:
                    self._csv_table_model.layoutChanged.emit()

                # Update loaded count
                self._table_loaded_rows = end_row

                # Update model row count and refresh view
                if self._csv_table_model:
                    # If no filter/sort is active, show all loaded rows
                    if self._csv_filtered_indices is None:
                        self._csv_table_model.set_row_indices(
                            None, total_rows=self._table_loaded_rows
                        )
                    else:
                        # If filter/sort is active, only show those indices that are within loaded rows
                        filtered = np.array(self._csv_filtered_indices)
                        filtered = filtered[filtered < self._table_loaded_rows]
                        self._csv_table_model.set_row_indices(filtered)

            # Re-enable updates
            self.preview_table.setUpdatesEnabled(True)

            # Update status bar
            if self._table_loaded_rows < self._csv_total_rows:
                self.statusBar().showMessage(
                    f"Loaded {self._table_loaded_rows:,} of {self._csv_total_rows:,} rows", 2000
                )
            else:
                self.statusBar().showMessage(f"All {self._csv_total_rows:,} rows loaded", 3000)
        except Exception as exc:
            self.statusBar().showMessage(f"Error loading rows: {exc}", 5000)
        finally:
            self._table_is_loading = False

    def _get_selected_column_indices(self) -> list[int]:
        """Get the indices of selected columns in the CSV table.

        Returns:
            Sorted list of selected column indices (excluding hidden columns)
        """
        try:
            sel_model = self.preview_table.selectionModel()
            if not sel_model:
                return []
            # Prefer selectedColumns if available
            cols = []
            try:
                cols = [idx.column() for idx in sel_model.selectedColumns()]
            except Exception:  # noqa: BLE001
                # Fallback: derive from selectedIndexes
                cols = list({idx.column() for idx in sel_model.selectedIndexes()})
            # Unique and sorted for stable behavior, excluding hidden columns
            return sorted({c for c in cols if c >= 0 and not self.preview_table.isColumnHidden(c)})
        except Exception:  # noqa: BLE001
            return []

    def _update_plot_action_enabled(self) -> None:
        """Update the enabled state of plot actions based on selection."""
        # Enable plotting when a CSV group is active and >= 1 column is selected
        is_csv = self._current_csv_group_path is not None and self.preview_table.isVisible()
        sel_cols = self._get_selected_column_indices() if is_csv else []
        self.act_plot_selected.setEnabled(is_csv and len(sel_cols) >= 1)
        self.act_contourf_selected.setEnabled(is_csv and len(sel_cols) == 3)

        # Also update plot management buttons
        self._update_plot_buttons_state()

    def _read_csv_columns(self, group_path: str, column_names: list[str]) -> dict[str, np.ndarray]:
        """
        Read one or more column arrays from a CSV-derived group by original column names.

        Args:
            group_path (str): HDF5 path to the CSV-derived group.
            column_names (list[str]): List of original column names to read.

        Returns:
            dict[str, np.ndarray]: Dictionary mapping original column name to numpy array (1-D). Strings remain strings.

        Behavior:
            - Attempts to map original column names to HDF5 dataset keys, using optional 'column_dataset_names' attribute if present.
            - Handles byte string decoding to UTF-8 for string columns.
            - Handles object dtype columns, decoding bytes if present.
            - If no filter or sort is active, updates filtered indices and table model for string columns.
            - Returns only columns found in the group; missing columns are skipped.
            - Handles scalar datasets and bytes gracefully.
        """
        # Always reset mapping and result for each call
        result: dict[str, np.ndarray] = {}
        mapping: dict[str, str] = {}
        fpath = self.model.filepath
        if not fpath:
            return result
        try:
            with h5py.File(fpath, "r") as h5:
                grp = h5[group_path]
                mapping: dict[str, str] = {}
                if "column_names" in grp.attrs:
                    try:
                        orig = [str(c) for c in list(grp.attrs["column_names"])]
                    except Exception:
                        orig = []
                    ds_names: list[str] | None = None
                    if "column_dataset_names" in grp.attrs:
                        try:
                            ds_names = [str(c) for c in list(grp.attrs["column_dataset_names"])]
                            if len(ds_names) != len(orig):
                                ds_names = None
                        except Exception:
                            ds_names = None
                    for i, name in enumerate(orig):
                        key = None
                        if ds_names is not None:
                            key = ds_names[i]
                        else:
                            cand = sanitize_hdf5_name(name)
                            key = cand if cand in grp else (name if name in grp else None)
                        if key is not None:
                            mapping[name] = key
                # Read requested columns
                for name in column_names:
                    ds_key = mapping.get(name)
                    if ds_key is None:
                        # Fallback to direct/sanitized key lookup
                        cand = sanitize_hdf5_name(name)
                        if cand in grp:
                            ds_key = cand
                        elif name in grp:
                            ds_key = name
                    key_in_group, th_grp = self._get_th_location(ds_key, grp)
                    if ds_key is None or not key_in_group:
                        continue
                    ds = th_grp[ds_key]
                    if not isinstance(ds, h5py.Dataset):
                        continue
                    data = ds[()]
                    if isinstance(data, np.ndarray):
                        arr = data
                        # Decode byte strings to UTF-8 strings
                        if arr.dtype.kind == "S":
                            # Byte strings - decode to UTF-8
                            arr = np.array(
                                [
                                    v.decode("utf-8", errors="replace")
                                    if isinstance(v, bytes)
                                    else str(v)
                                    for v in arr
                                ],
                                dtype=object,
                            )
                            # Update filtered indices and model if no filter/sort is active
                            if (not self._csv_filters) and (not self._csv_sort_specs):
                                self._csv_filtered_indices = np.arange(self._table_loaded_rows)
                                if self._csv_table_model:
                                    self._csv_table_model.set_row_indices(
                                        self._csv_filtered_indices
                                    )
                                    self.preview_table.viewport().update()
                        elif arr.dtype.kind == "O":
                            # Object dtype - could be mixed, handle bytes if present
                            arr = np.array(
                                [
                                    v.decode("utf-8", errors="replace")
                                    if isinstance(v, bytes)
                                    else v
                                    for v in arr
                                ],
                                dtype=object,
                            )
                    else:
                        # Handle scalar bytes
                        if isinstance(data, bytes):
                            data = data.decode("utf-8", errors="replace")
                        arr = np.array([data])
                    result[name] = arr
        except Exception:  # noqa: BLE001
            return result
        return result

    def get_col(self, name: str) -> np.ndarray | None:
        """Get a column array by name from the currently loaded CSV data.

        Args:
            name: Column name

        Returns:
            Numpy array of the column data (1-D). None if not found.
        """
        if self._csv_data_dict is None or name not in self._csv_data_dict:
            return None

        # Get the actual data length after loading
        actual_data_len = (
            max(len(self._csv_data_dict[col]) for col in self._csv_data_dict)
            if self._csv_data_dict
            else 0
        )

        # Filter indices to only valid range (in case of partial loading)
        valid_filtered_indices = self._csv_filtered_indices[
            self._csv_filtered_indices < actual_data_len
        ]

        full_data = self._csv_data_dict[name]
        if isinstance(full_data, np.ndarray):
            return full_data[valid_filtered_indices]
        else:
            return np.array([full_data[i] for i in valid_filtered_indices if i < len(full_data)])

    def plot_selected_columns_contourf(self) -> None:
        """Plot selected columns as contourf plot."""
        self.plot_selected_columns(contourf=True)

    def plot_contourf_from_data(
        self,
        col_data: dict[str, np.ndarray],
        x_name: str,
        y_names: list[str],
        ax,
        grid_size: int = 100,
        method: str = "linear",
        cmap: str = "Blues",
        cmap_label: str = "",
        levels: int = 20,
    ) -> None:
        """
        Plot a filled contour plot (contourf) using three columns of data.

        Args:
            col_data (dict[str, np.ndarray]): Dictionary mapping column names to numpy arrays.
            x_name (str): Name of the column to use for the X axis.
            y_names (list[str]): List of two column names; first for Y axis, second for Z values.
            ax (matplotlib.axes.Axes): The matplotlib Axes object to plot on.
            grid_size (int, optional): Size of the grid for interpolation. Default is 100.
            method (str, optional): Interpolation method for griddata (e.g., 'linear', 'nearest', 'cubic'). Default is 'linear'.
            cmap (str, optional): Colormap to use for contourf. Default is 'Blues'.
            cmap_label (str, optional): Label for the colorbar. Default is ''.
            levels (int, optional): Number of contour levels. Default is 20.

        Notes:
            - The function flattens and converts all input arrays to float.
            - Uses scipy.interpolate.griddata to interpolate Z values on a grid.
            - Plots the filled contour using ax.contourf and adds a colorbar.
        """
        x = col_data[x_name]
        y = col_data[y_names[0]]
        z = col_data[y_names[1]]
        # Defensive: flatten and convert to float
        x = np.asarray(x).ravel().astype(float)
        y = np.asarray(y).ravel().astype(float)
        z = np.asarray(z).ravel().astype(float)
        # Try to create a grid for contourf
        from scipy.interpolate import griddata

        # Create grid
        xi = np.linspace(np.nanmin(x), np.nanmax(x), grid_size)
        yi = np.linspace(np.nanmin(y), np.nanmax(y), grid_size)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method=method)
        cf = ax.contourf(xi, yi, zi, levels=levels, cmap=cmap)
        cbar = self.plot_figure.colorbar(cf, ax=ax)
        self.cbar = cbar  # save it so we can adjust it later
        if cmap_label:
            cbar.set_label(cmap_label)
        else:
            # Label colorbar with z-series name (second entry in y_names)
            z_label = y_names[1] if len(y_names) > 1 else ""
            if z_label:
                cbar.set_label(z_label)

    def plot_selected_columns(self, contourf: bool = False) -> None:
        """
        Generate and display a plot from the currently selected columns in the CSV table.

        Args:
            contourf (bool, optional): If True, create a contourf plot using three selected columns.
                If False, create a line plot using one or more selected columns. Default is False.

        Behavior:
            - Ensures all CSV data is loaded before plotting.
            - Determines which columns are selected and builds a plot configuration dict.
            - For contourf, expects exactly three columns selected (X, Y, Z).
            - For line plots, uses one or more columns (X and Y).
            - Calls _apply_saved_plot_config to perform the actual plotting.
        """
        if self._current_csv_group_path is None or not self.preview_table.isVisible():
            return
        self._ensure_all_data_loaded()
        sel_cols = self._get_selected_column_indices()
        if len(sel_cols) < 1:
            return
        if contourf and len(sel_cols) != 3:
            return
        model = self.preview_table.model()
        headers = (
            [str(model.headerData(i, Qt.Horizontal)) for i in range(model.columnCount())]
            if model
            else []
        )
        # Determine x and y indices
        if len(sel_cols) == 1:
            x_idx = None
            y_idxs = sel_cols
        else:
            current_index = self.preview_table.selectionModel().currentIndex()
            current_col = current_index.column() if current_index.isValid() else None
            x_idx = current_col if current_col in sel_cols else min(sel_cols)
            y_idxs = [c for c in sel_cols if c != x_idx]
        # Build plot config
        plot_type = "contourf" if contourf else "line"
        series_visibility = {}
        plot_config = {
            "name": "Quick Plot",
            "csv_group_path": self._current_csv_group_path,
            "column_names": headers,
            "x_col_idx": x_idx,
            "y_col_idxs": y_idxs,
            "filtered_indices": indices_to_ranges(self._csv_filtered_indices)
            if self._csv_filtered_indices is not None
            else None,
            "start_row": int(self._csv_filtered_indices[0])
            if self._csv_filtered_indices is not None and len(self._csv_filtered_indices) > 0
            else 0,
            "end_row": int(self._csv_filtered_indices[-1])
            if self._csv_filtered_indices is not None and len(self._csv_filtered_indices) > 0
            else 0,
            "csv_filters": self._csv_filters.copy() if self._csv_filters else [],
            "csv_sort": self._csv_sort_specs.copy()
            if hasattr(self, "_csv_sort_specs") and self._csv_sort_specs
            else [],
            "timestamp": time.time(),
            "plot_options": {
                "type": plot_type,
                "title": "",
                "xlabel": "",
                "ylabel": "",
                "grid": True,
                "legend": True,
                "series": {},
                "series_visibility": series_visibility,
            },
        }
        # Call unified plot logic
        self._apply_saved_plot_config(plot_config)

    def _apply_saved_plot_config(self, plot_config: dict) -> None:
        """
        Wrapper to apply a plot configuration using the unified logic in _apply_saved_plot.

        Args:
            plot_config (dict): Plot configuration dictionary containing all necessary plot parameters.

        Behavior:
            - Delegates to _apply_saved_plot for actual plotting logic.
        """
        # Always use the provided plot_config for ad-hoc plotting
        self._apply_saved_plot(plot_config=plot_config)

    def _clear_sort(self) -> None:
        """Clear all sorting and display data in original order."""
        self._csv_sort_specs = []
        self._save_sort_to_hdf5()
        self._apply_sort()

    def _save_sort_to_hdf5(self) -> None:
        """Save current sort specifications to the HDF5 file as a JSON attribute."""
        success_msg = (
            f"Saved sort by {len(self._csv_sort_specs)} column(s) to HDF5 file"
            if self._csv_sort_specs
            else None
        )
        clear_msg = "Cleared sort from HDF5 file"
        self._save_csv_attr_to_hdf5("csv_sort", self._csv_sort_specs, success_msg, clear_msg)

        # Also update the model with sort specs for drag-and-drop
        if self._current_csv_group_path and self.model:
            self.model.set_csv_sort_specs(self._current_csv_group_path, self._csv_sort_specs)

    def _load_sort_from_hdf5(self, grp: h5py.Group) -> list:
        """Load sort specifications from the HDF5 group attributes."""
        """
        Load sort specifications from the HDF5 group attributes.

        Args:
            grp: HDF5 group to load sort specs from

        Returns:
            List of sort specifications as tuples (column_name, ascending: bool), or empty list if not found.
        """

        def validate_sort(specs):
            if isinstance(specs, list):
                return [tuple(spec) for spec in specs if isinstance(spec, list) and len(spec) == 2]
            return []

        return self._load_csv_attr_from_hdf5(grp, "csv_sort", validate_sort) or []

    def _configure_columns_dialog(self) -> None:
        """Open dialog to select which columns to display."""
        if not self._csv_column_names:
            QMessageBox.information(self, "No CSV Data", "Load a CSV group first.")
            return

        dialog = ColumnVisibilityDialog(
            self._csv_column_names,
            self._csv_visible_columns
            if self._csv_visible_columns
            else self._csv_column_names.copy(),
            self,
        )

        if dialog.exec() == QDialog.Accepted:
            self._csv_visible_columns = dialog.get_visible_columns()
            self._save_visible_columns_to_hdf5()
            self._apply_column_visibility()

            # Update model so drag-and-drop exports only visible columns
            if self._current_csv_group_path and self.model:
                self.model.set_csv_visible_columns(
                    self._current_csv_group_path, self._csv_visible_columns
                )

    def _apply_column_visibility(self) -> None:
        """Apply column visibility to the table."""
        if not self._csv_column_names:
            return

        # First, ensure all columns are visible (reset state)
        # This is important when switching between CSV groups
        model = self.preview_table.model()
        if model:
            for col_idx in range(model.columnCount()):
                self.preview_table.setColumnHidden(col_idx, False)

        # Hide/show columns based on visibility list
        for col_idx, col_name in enumerate(self._csv_column_names):
            should_hide = col_name not in self._csv_visible_columns
            self.preview_table.setColumnHidden(col_idx, should_hide)

    def _on_column_header_context_menu(self, pos) -> None:
        """Handle right-click context menu on column header.

        Args:
            pos: QPoint position where the context menu was requested
        """
        # Get the column index at the clicked position
        header = self.preview_table.horizontalHeader()
        col_idx = header.logicalIndexAt(pos)

        if col_idx < 0 or col_idx >= len(self._csv_column_names):
            return

        col_name = self._csv_column_names[col_idx]

        # Create context menu
        menu = QMenu(self)
        act_unique = menu.addAction(f"Show Unique Values in '{col_name}'")

        # Connect the triggered signal to handle selection
        def handle_action(action):
            if action == act_unique:
                self._show_unique_values_dialog(col_name)

        menu.triggered.connect(handle_action)

        # Show menu (blocking, closes when clicking outside)
        global_pos = header.mapToGlobal(pos)
        menu.exec_(global_pos)

    def _show_unique_values_dialog(self, col_name: str) -> None:
        """Show dialog with unique values for a specific column.

        Args:
            col_name: Name of the column to show unique values for
        """
        # Ensure all data is loaded for accurate unique value calculation
        self._ensure_all_data_loaded()

        if not self._csv_data_dict or col_name not in self._csv_data_dict:
            QMessageBox.information(self, "No Data", f"Column '{col_name}' has no data loaded.")
            return

        # Get the column data
        col_data = self._csv_data_dict[col_name]

        # If filters are active, use only filtered rows
        if hasattr(self, "_csv_filtered_indices") and self._csv_filtered_indices is not None:
            # Ensure filtered indices are within bounds
            actual_data_len = len(col_data)
            valid_indices = self._csv_filtered_indices[self._csv_filtered_indices < actual_data_len]
            # Filter the data to only include visible rows
            filtered_data = col_data[valid_indices]
        else:
            filtered_data = col_data

        # Get unique values and sort them
        try:
            unique_values = np.unique(filtered_data)
            # Convert to list for display
            unique_list = [str(val) for val in unique_values]
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to get unique values: {e}")
            return

        # Show dialog
        dialog = UniqueValuesDialog(col_name, unique_list, self)
        dialog.exec()

    def _save_visible_columns_to_hdf5(self) -> None:
        """Save visible columns list to the HDF5 file as a JSON attribute."""
        # Only save if not all columns are visible
        value = (
            self._csv_visible_columns
            if (
                self._csv_visible_columns
                and len(self._csv_visible_columns) < len(self._csv_column_names)
            )
            else None
        )
        success_msg = (
            f"Saved column visibility ({len(self._csv_visible_columns)}/{len(self._csv_column_names)} columns) to HDF5 file"
            if value
            else None
        )
        clear_msg = "All columns visible - removed saved visibility preference"
        self._save_csv_attr_to_hdf5("csv_visible_columns", value, success_msg, clear_msg)

    def _load_visible_columns_from_hdf5(self, grp: h5py.Group):
        """Load visible columns list from the HDF5 group attributes.

        Args:
            grp: HDF5 group to load visibility settings from

        Returns:
            List of visible column names, or None if not saved
        """
        return self._load_csv_attr_from_hdf5(
            grp, "csv_visible_columns", lambda v: v if isinstance(v, list) else None
        )

    def _save_filters_to_hdf5(self) -> None:
        """Save current filters to the HDF5 file as a JSON attribute."""
        success_msg = (
            f"Saved {len(self._csv_filters)} filter(s) to HDF5 file" if self._csv_filters else None
        )
        clear_msg = "Cleared filters from HDF5 file"
        self._save_csv_attr_to_hdf5("csv_filters", self._csv_filters, success_msg, clear_msg)

    def _load_filters_from_hdf5(self, grp: h5py.Group) -> list:
        """Load filters from the HDF5 group attributes.

        Args:
            grp: HDF5 group to load filters from

        Returns:
            List of filters in format [column_name, operator, value]
        """

        def validate_filters(filters):
            if isinstance(filters, list):
                return [f for f in filters if isinstance(f, list) and len(f) == 3]
            return []

        return self._load_csv_attr_from_hdf5(grp, "csv_filters", validate_filters) or []

    def _apply_sort(self) -> None:
        """Apply current sort specifications to the CSV table."""
        if not self._csv_data_dict or not self._csv_column_names:
            return

        # Update sort button state
        if self._csv_sort_specs:
            self.btn_clear_sort.setEnabled(True)
        else:
            self.btn_clear_sort.setEnabled(False)

        # After changing sort, reapply filtering and sorting, then update grid
        filtered_indices, start_row, end_row = self._get_filtered_sorted_indices(
            self._csv_data_dict, self._csv_filters, self._csv_sort_specs
        )
        self._csv_filtered_indices = filtered_indices
        if hasattr(self, "_csv_table_model") and self._csv_table_model:
            self._csv_table_model.set_row_indices(filtered_indices)
        # Update filter status label and clear sort button
        if self._csv_sort_specs:
            self.btn_clear_sort.setEnabled(True)
        else:
            self.btn_clear_sort.setEnabled(False)

    def _apply_filters(self) -> None:
        """Apply current filters to the CSV table."""
        if not self._csv_data_dict and not self._csv_dataset_info:
            return

        # Ensure all data is loaded before filtering
        self._ensure_all_data_loaded()

        # Update filter status label
        if self._csv_filters:
            filter_text = f"{len(self._csv_filters)} filter(s) applied"
            self.filter_status_label.setText(filter_text)
            self.btn_clear_filters.setEnabled(True)
        else:
            self.filter_status_label.setText("No filters applied")
            self.btn_clear_filters.setEnabled(False)

        # Use shared routine for filtering and sorting
        filtered_indices, start_row, end_row = self._get_filtered_sorted_indices(
            self._csv_data_dict, self._csv_filters, self._csv_sort_specs
        )

        # Store filtered indices for plotting
        self._csv_filtered_indices = filtered_indices

        # Notify the model about filtered indices for CSV export
        max_rows = (
            max(len(self._csv_data_dict[col]) for col in self._csv_data_dict)
            if self._csv_data_dict
            else 0
        )
        if self._current_csv_group_path and self.model:
            if len(filtered_indices) == max_rows:
                # No filtering active, clear stored indices
                self.model.set_csv_filtered_indices(self._current_csv_group_path, None)
            else:
                # Set filtered indices
                self.model.set_csv_filtered_indices(self._current_csv_group_path, filtered_indices)

        # For QTableView, update the model's row_indices and row_count using helper
        if self._csv_table_model:
            if len(filtered_indices) == max_rows:
                self._csv_table_model.set_row_indices(None, total_rows=max_rows)
            else:
                self._csv_table_model.set_row_indices(filtered_indices)

        # Update status message
        if self._csv_filters:
            total_rows = max_rows
            shown_rows = len(filtered_indices)
            self.statusBar().showMessage(
                f"Showing {shown_rows:,} of {total_rows:,} rows (filtered)", 5000
            )

    def _evaluate_filter(
        self, col_data: np.ndarray | list, operator: str, value_str: str
    ) -> np.ndarray:
        """
        Evaluate a filter condition on column data.

        Args:
            col_data: Array or list of column values to filter.
            operator: Comparison operator as a string (e.g., '==', '!=', '<', '>', etc.).
            value_str: The value to compare against, as a string.

        Handles numeric, datetime, and string comparisons.

        Returns:
            np.ndarray: Boolean mask of the same length as col_data indicating which rows match the filter.
        """

        # Mapping of operators to lambda functions for cleaner code
        ops = {
            "==": lambda a, b: a == b,
            "=": lambda a, b: a == b,  # Support both = and ==
            "!=": lambda a, b: a != b,
            "<": lambda a, b: a < b,
            "<=": lambda a, b: a <= b,
            ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b,
        }

        try:
            # Handle comparison operators using NumPy vectorized operations
            if operator in ["==", "=", "!=", ">", ">=", "<", "<="]:
                arr = np.array(col_data)
                # 1. Try numeric comparison
                try:
                    arr_num = arr.astype(float)
                    val_num = float(value_str)
                    return ops[operator](arr_num, val_num)
                except Exception:
                    pass
                # 2. Try datetime comparison
                try:
                    arr_dt = pd.to_datetime(arr, errors="coerce")
                    val_dt = pd.to_datetime(value_str, errors="coerce")
                    valid = np.logical_not(pd.isna(arr_dt)) & np.logical_not(pd.isna(val_dt))
                    mask = np.zeros(len(arr_dt), dtype=bool)
                    mask[valid] = ops[operator](arr_dt[valid], val_dt)
                    return mask
                except Exception:
                    pass
                # 3. Fall back to string comparison
                arr_str = arr.astype(str)
                return ops[operator](arr_str, value_str)

            # String-based operations
            arr_str = np.array(
                [
                    str(v) if not isinstance(v, bytes) else v.decode("utf-8", errors="replace")
                    for v in col_data
                ]
            )
            if operator == "contains":
                return np.char.find(arr_str, value_str) >= 0
            elif operator == "startswith":
                return np.char.startswith(arr_str, value_str)
            elif operator == "endswith":
                return np.char.endswith(arr_str, value_str)

            # Fallback: all True (no filtering)
            return np.ones(len(arr_str), dtype=bool)

        except Exception:  # noqa: BLE001
            # On error, don't filter any rows
            return np.ones(len(col_data), dtype=bool)

    # ========== Plot Configuration Management ==========

    def _save_plot_config_dialog(self) -> None:
        """Open dialog to save current plot configuration."""
        if not self._current_csv_group_path or not self.preview_table.isVisible():
            QMessageBox.information(
                self, "No CSV Data", "Load a CSV group and create a plot first."
            )
            return

        # Get currently selected columns
        sel_cols = self._get_selected_column_indices()
        if len(sel_cols) < 1:
            QMessageBox.information(
                self,
                "No Plot Selection",
                "Select at least one column before saving a plot configuration.",
            )
            return

        # Determine X and Y columns (same logic as plot_selected_columns)
        if len(sel_cols) == 1:
            x_idx = None  # Single column mode: use point count
            y_idxs = sel_cols
        else:
            current_index = self.preview_table.selectionModel().currentIndex()
            current_col = current_index.column() if current_index.isValid() else None
            x_idx = current_col if current_col in sel_cols else min(sel_cols)
            y_idxs = [c for c in sel_cols if c != x_idx]

        # Prompt for plot name
        plot_name, ok = QInputDialog.getText(
            self,
            "Save Plot Configuration",
            "Enter a name for this plot configuration:",
            QLineEdit.Normal,
            f"Plot {len(self._saved_plots) + 1}",
        )

        if not ok or not plot_name:
            return

        # Store the complete filtered indices array to properly handle non-contiguous filtering
        if self._csv_filtered_indices is not None and len(self._csv_filtered_indices) > 0:
            # Store as compact range format for space efficiency
            filtered_indices = indices_to_ranges(self._csv_filtered_indices)
            start_row = int(self._csv_filtered_indices[0])
            end_row = int(self._csv_filtered_indices[-1])
        else:
            # No filtering - use full range in compressed format
            max_rows = (
                max(len(self._csv_data_dict[col]) for col in self._csv_data_dict)
                if self._csv_data_dict
                else 0
            )
            if max_rows > 0:
                # Store full range in compressed format (e.g., ['0-9999'] instead of all indices)
                filtered_indices = [f"0-{max_rows - 1}"]
                start_row = 0
                end_row = max_rows - 1
            else:
                filtered_indices = []
                start_row = 0
                end_row = 0

        # Create plot configuration dictionary

        model = self.preview_table.model()
        if model:
            column_names = [
                str(model.headerData(i, Qt.Horizontal)) for i in range(model.columnCount())
            ]
        else:
            column_names = []

        # Capture current visibility state from plot if available
        series_visibility = self._capture_plot_visibility_state()

        plot_type = "contourf" if (len(y_idxs) == 2 and x_idx is not None) else "line"
        plot_config = {
            "name": plot_name,
            "csv_group_path": self._current_csv_group_path,
            "column_names": column_names,
            "x_col_idx": x_idx,
            "y_col_idxs": y_idxs,
            "filtered_indices": filtered_indices,  # Store actual filtered row indices
            "start_row": start_row,  # Keep for backward compatibility
            "end_row": end_row,  # Keep for backward compatibility
            "csv_filters": self._csv_filters.copy()
            if self._csv_filters
            else [],  # Store filter specs
            "csv_sort": self._csv_sort_specs.copy()
            if hasattr(self, "_csv_sort_specs") and self._csv_sort_specs
            else [],  # Store sort specs
            "timestamp": time.time(),
            "plot_options": {
                "type": plot_type,
                "title": "",
                "xlabel": "",
                "ylabel": "",
                "grid": True,
                "legend": True,
                "series": {},  # Will be populated with per-series styles in the Edit Options dialog
                "series_visibility": series_visibility,  # Store current visibility state
            },
        }

        # Add to local list
        self._saved_plots.append(plot_config)

        # Save to HDF5
        self._save_plot_configs_to_hdf5()

        # Update list widget
        self._refresh_saved_plots_list()

        # Select the newly added plot in the list
        self.saved_plots_list.setCurrentRow(len(self._saved_plots) - 1)

        self.statusBar().showMessage(f"Saved plot configuration: {plot_name}", 3000)

    def _save_plot_configs_to_hdf5(self) -> None:
        """Save all plot configurations to the HDF5 file as a JSON attribute."""
        if not self._current_csv_group_path or not self.model or not self.model.filepath:
            return

        try:
            with h5py.File(self.model.filepath, "r+") as h5:
                if self._current_csv_group_path in h5:
                    grp = h5[self._current_csv_group_path]
                    if isinstance(grp, h5py.Group):
                        if self._saved_plots:
                            # Convert plot configs to JSON string
                            plots_json = json.dumps(self._saved_plots)
                            grp.attrs["saved_plots"] = plots_json
                        else:
                            # Remove attribute if no plots
                            if "saved_plots" in grp.attrs:
                                del grp.attrs["saved_plots"]

            # Update file size display after modification
            self._update_file_size_display()
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Warning: Could not save plot configs: {exc}", 5000)

    def _load_plot_configs_from_hdf5(self, grp: h5py.Group):
        """Load plot configurations from the HDF5 group attributes.

        Args:
            grp: HDF5 group to load plot configs from
        """
        try:
            if "saved_plots" in grp.attrs:
                plots_json = grp.attrs["saved_plots"]
                if isinstance(plots_json, bytes):
                    plots_json = plots_json.decode("utf-8")
                plots = json.loads(plots_json)
                # Validate format
                if isinstance(plots, list):
                    self._saved_plots = plots
                else:
                    self._saved_plots = []
            else:
                self._saved_plots = []
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Could not load plot configs from HDF5: {exc}")
            self._saved_plots = []

        # Refresh the list widget
        self._refresh_saved_plots_list()

        # Auto-select the first plot if available, or clear plot display if no plots
        if self._saved_plots:
            self.saved_plots_list.setCurrentRow(0)
        else:
            self._clear_plot_display()

    def _refresh_saved_plots_list(self) -> None:
        """Update the saved plots list widget with current configurations."""
        self.saved_plots_list.clear()

        # Get a standard chart/graph icon for plots
        style = QApplication.instance().style() if QApplication.instance() else None
        plot_icon = style.standardIcon(QStyle.SP_FileDialogContentsView) if style else None

        for plot_config in self._saved_plots:
            name = plot_config.get("name", "Unnamed Plot")
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsEditable)

            # Add plot icon to each item
            if plot_icon:
                item.setIcon(plot_icon)

            self.saved_plots_list.addItem(item)

        # Update button states
        self._update_plot_buttons_state()

    def _clear_plot_display(self) -> None:
        """Clear the plot display area."""
        self.plot_figure.clear()
        self.plot_canvas.draw()

    def _update_plot_buttons_state(self) -> None:
        """Enable/disable plot management buttons based on current state."""
        # Enable Save Plot button if CSV is loaded and columns are selected
        csv_loaded = self._current_csv_group_path is not None and self.preview_table.isVisible()
        sel_cols = self._get_selected_column_indices() if csv_loaded else []
        self.btn_save_plot.setEnabled(csv_loaded and len(sel_cols) >= 1)

        # Enable Delete and Edit Options buttons if a plot is selected
        has_selection = self.saved_plots_list.currentRow() >= 0
        self.btn_delete_plot.setEnabled(has_selection)
        self.btn_edit_plot_options.setEnabled(has_selection)

    def _on_saved_plot_selection_changed(self) -> None:
        """Handle selection change in saved plots list."""
        self._update_plot_buttons_state()

        # Automatically apply the selected plot
        current_item = self.saved_plots_list.currentItem()
        if current_item is not None:
            self._apply_saved_plot(current_item)

    def _on_saved_plot_clicked(self, item: QListWidgetItem) -> None:
        """Handle clicking on a saved plot item (even if already selected).

        Args:
            item: The list item that was clicked
        """
        # Apply the plot even if it's already selected
        if item is not None:
            self._apply_saved_plot(item)

    def _on_plot_item_renamed(self, item: QListWidgetItem):
        """Handle when a plot item is renamed by the user.

        Args:
            item: The list item that was edited
        """
        new_name = item.text().strip()
        if not new_name:
            # Don't allow empty names
            row = self.saved_plots_list.row(item)
            if 0 <= row < len(self._saved_plots):
                old_name = self._saved_plots[row].get("name", "Unnamed Plot")
                item.setText(old_name)
            return

        # Get the row index to find the corresponding plot config
        row = self.saved_plots_list.row(item)
        if 0 <= row < len(self._saved_plots):
            old_name = self._saved_plots[row].get("name", "Unnamed Plot")

            # Check if name is different
            if new_name != old_name:
                # Update the plot config
                self._saved_plots[row]["name"] = new_name

                # Save to HDF5 file using the same method as other plot operations
                if self._current_csv_group_path and self.model and self.model.filepath:
                    try:
                        with h5py.File(self.model.filepath, "r+") as h5:
                            if self._current_csv_group_path in h5:
                                grp = h5[self._current_csv_group_path]
                                if isinstance(grp, h5py.Group):
                                    plots_json = json.dumps(self._saved_plots)
                                    grp.attrs["saved_plots"] = plots_json
                        self.statusBar().showMessage(f"Renamed plot to '{new_name}'", 3000)
                    except Exception as e:
                        self.statusBar().showMessage(f"Error saving renamed plot: {e}", 5000)
                        # Revert the name on error
                        item.setText(old_name)
                        self._saved_plots[row]["name"] = old_name

    def _apply_filtered_indices_to_data(
        self,
        col_data_dict: dict[str, np.ndarray],
        filtered_indices: list[int] | None,
        start_row: int = 0,
        end_row: int = -1,
    ) -> dict[str, np.ndarray]:
        """Apply filtered indices to column data with bounds checking.

        Args:
            col_data_dict: Dictionary of column name -> numpy array
            filtered_indices: List of row indices to include, or None
            start_row: Starting row for backward compatibility (used if filtered_indices is None)
            end_row: Ending row for backward compatibility (used if filtered_indices is None)

        Returns:
            Dictionary with filtered data
        """
        result = {}
        for col_name, col_array in col_data_dict.items():
            if isinstance(col_array, np.ndarray) and len(col_array) > 0:
                if filtered_indices is not None:
                    # Use the stored filtered indices (handles non-contiguous filtering)
                    filtered_indices_array = np.array(filtered_indices, dtype=int)
                    # Filter out any indices that are out of bounds for the current data
                    valid_indices = filtered_indices_array[filtered_indices_array < len(col_array)]
                    if len(valid_indices) > 0:
                        result[col_name] = col_array[valid_indices]
                    else:
                        result[col_name] = np.array([])
                elif end_row >= 0 and end_row < len(col_array):
                    # Backward compatibility: use row range
                    result[col_name] = col_array[start_row : end_row + 1]
                else:
                    result[col_name] = col_array[start_row:]
            else:
                result[col_name] = col_array
        return result

    def _apply_saved_plot(
        self, item: QListWidgetItem | None = None, plot_config: dict = None
    ) -> None:
        """
        Apply a saved plot configuration, or an ad-hoc plot config if provided.

        Args:
            item: QListWidgetItem that was clicked/selected (optional)
            plot_config: Optional plot configuration dict. If provided, use this config directly.
        """
        if plot_config is not None:
            config = plot_config
            # Use headers from config
            headers = config.get("column_names", [])
        else:
            if item is None:
                item = self.saved_plots_list.currentItem()
            if item is None:
                return
            row = self.saved_plots_list.row(item)
            if row < 0 or row >= len(self._saved_plots):
                return
            config = self._saved_plots[row]
            # Use headers from current model
            model = self.preview_table.model()
            if model:
                headers = [
                    str(model.headerData(i, Qt.Horizontal)) for i in range(model.columnCount())
                ]
            else:
                headers = []
        x_idx = config.get("x_col_idx")
        y_idxs = config.get("y_col_idxs", [])
        self.cbar = None  # initialize
        if isinstance(y_idxs, int):
            y_idxs = [y_idxs]
        filtered_indices_raw = config.get("filtered_indices")
        if filtered_indices_raw is not None and len(filtered_indices_raw) > 0:
            # Check if it's in the new compact format (contains strings or is a mixed list)
            if any(isinstance(x, str) for x in filtered_indices_raw):
                filtered_indices = ranges_to_indices(filtered_indices_raw)
            else:
                filtered_indices = np.array(filtered_indices_raw, dtype=np.int64)
        else:
            filtered_indices = filtered_indices_raw
        start_row = config.get("start_row", 0)
        end_row = config.get("end_row", -1)
        if not y_idxs:
            QMessageBox.warning(
                self, "Invalid Configuration", "Plot configuration is missing column information."
            )
            return
        # Check if we have the CSV data loaded
        if not self._csv_data_dict or not self._current_csv_group_path:
            QMessageBox.information(self, "No Data", "CSV data is not loaded.")
            return
        # Validate column indices
        if (x_idx is not None and x_idx >= len(headers)) or any(
            y_idx >= len(headers) for y_idx in y_idxs
        ):
            QMessageBox.warning(
                self, "Invalid Columns", "Plot configuration references invalid column indices."
            )
            return
        try:
            x_name = headers[x_idx] if x_idx is not None else "Point"
            y_names = [headers[i] for i in y_idxs]
        except Exception:
            QMessageBox.warning(
                self, "Plot Error", "Failed to resolve column headers for plotting."
            )
            return
        # Read column data directly from HDF5 for plotting (don't use table's lazy-loading cache)
        columns_to_read = y_names if x_idx is None else [x_name] + y_names
        col_data = self._read_csv_columns(self._current_csv_group_path, columns_to_read)
        if not col_data:
            QMessageBox.warning(self, "Plot Error", "Failed to read column data from HDF5.")
            return
        # Handle non-array data
        for name in list(col_data.keys()):
            if not isinstance(col_data[name], np.ndarray):
                col_data[name] = np.array([col_data[name]])
        # Apply filtered indices with bounds checking
        col_data = self._apply_filtered_indices_to_data(
            col_data, filtered_indices, start_row, end_row
        )
        if not any(name in col_data for name in y_names):
            QMessageBox.warning(self, "Plot Error", "Failed to get column data for plotting.")
            return
        try:
            plot_options = config.get("plot_options", {})
            x_arr, x_num, x_is_string, xaxis_datetime, min_len = self._process_x_axis_data(
                x_idx, col_data, y_names, x_name, plot_options
            )
            if min_len <= 0:
                QMessageBox.warning(self, "Plot Error", "No data to plot.")
                return
            # Clear previous plot
            self.plot_figure.clear()
            # Get plot options to check for dark background
            use_dark = plot_options.get("dark_background", False)
            # Create subplot and apply style
            ax = self.plot_figure.add_subplot(111)
            self._apply_plot_style(self.plot_figure, ax, use_dark)
            # Disable offset notation on axes
            ax.ticklabel_format(useOffset=False)
            series_visibility = plot_options.get("series_visibility", {})
            contourf = plot_options.get("type", "line") == "contourf"
            if contourf:
                if (
                    x_name not in col_data
                    or len(y_names) != 2
                    or any(y not in col_data for y in y_names)
                ):
                    QMessageBox.warning(
                        self,
                        "Plot Error",
                        "Contourf plot requires one X and two Y columns (Z as second Y).",
                    )
                    return
                try:
                    cmap = plot_options.get("cmap", "Blues")
                    cmap_label = plot_options.get("cmap_label", "")
                    levels = plot_options.get("levels", 20)
                    self.plot_contourf_from_data(
                        col_data, x_name, y_names, ax,
                        cmap=cmap, cmap_label=cmap_label, levels=levels
                    )
                except Exception as exc:
                    QMessageBox.critical(self, "Plot Error", f"Failed to plot contourf: {exc}")
                    return
            else:
                # Get series styling options
                series_styles = plot_options.get("series", {})
                series_visibility = plot_options.get("series_visibility", {})
                any_plotted = False
                for y_name in y_names:
                    if y_name not in col_data:
                        continue
                    y_arr = col_data[y_name].ravel()[:min_len]
                    y_num = (
                        pd.to_numeric(pd.Series(y_arr), errors="coerce").astype(float).to_numpy()
                    )
                    valid = np.isfinite(x_num) & np.isfinite(y_num)
                    if valid.any():
                        series_opts = series_styles.get(y_name, {})
                        any_plotted = self._plot_series_with_options(
                            ax, x_num, y_num, valid, y_name, series_opts, any_plotted
                        )
                if not any_plotted:
                    QMessageBox.information(self, "Plot", "No valid numeric data found to plot.")
                    return
            # Adjust title for row range if needed
            title_suffix = ""
            if start_row > 0 or end_row < len(self._csv_data_dict.get(x_name, [])) - 1:
                title_suffix = f" (rows {start_row}-{end_row})"
            # Create modified plot_config with title suffix
            modified_config = config.copy()
            if not plot_options.get("title", "").strip():
                modified_config["name"] = config.get("name", "Plot") + title_suffix
            # Format x-axis (datetime or categorical strings)
            self._format_xaxis(
                ax, self.plot_figure, xaxis_datetime, x_is_string, x_arr, min_len, plot_options
            )
            # Apply axis limits
            self._apply_axis_limits(ax, plot_options)
            # Apply labels, fonts, grid, legend, log scale, and reference lines
            self._apply_plot_labels_and_formatting(
                ax, self.plot_figure, x_name, y_names, modified_config, plot_options, use_dark
            )
            if not contourf:
                # Apply saved visibility state AFTER formatting (so legend is already created)
                self._apply_plot_visibility_state(ax, y_names, series_visibility)
            # Refresh canvas
            self.updateCanvas()
            # Switch to Plot tab
            self.bottom_tabs.setCurrentIndex(1)
            self.statusBar().showMessage(f"Applied plot: {config.get('name', 'Unnamed')}", 3000)
        except Exception as exc:
            QMessageBox.critical(self, "Plot Error", f"Failed to plot data:\n{exc}")

    def _configure_filters_dialog(self) -> None:
        """Open dialog to configure column filters."""
        if not self._csv_column_names:
            QMessageBox.information(self, "No CSV Data", "Load a CSV group first.")
            return

        dialog = ColumnFilterDialog(self._csv_column_names, self)
        dialog.set_filters(self._csv_filters)

        if dialog.exec() == QDialog.Accepted:
            self._csv_filters = dialog.get_filters()
            self._save_filters_to_hdf5()
            self._apply_filters()

    def _show_statistics_dialog(self) -> None:
        """Open dialog to show statistics for CSV columns."""
        if not self._csv_column_names or (not self._csv_data_dict and not self._csv_dataset_info):
            QMessageBox.information(self, "No CSV Data", "Load a CSV group first.")
            return

        # Ensure all data is loaded before calculating statistics
        self._ensure_all_data_loaded()

        # Use filtered indices if available
        filtered_indices = (
            self._csv_filtered_indices if hasattr(self, "_csv_filtered_indices") else None
        )

        dialog = ColumnStatisticsDialog(
            self._csv_column_names, self._csv_data_dict, filtered_indices, self
        )
        dialog.exec()

    def _clear_filters(self) -> None:
        """Clear all active filters and show full dataset."""
        self._csv_filters = []
        self._save_filters_to_hdf5()
        self._apply_filters()

    def _configure_sort_dialog(self) -> None:
        """Open dialog to configure column sorting."""
        if not self._csv_column_names:
            QMessageBox.information(self, "No CSV Data", "Load a CSV group first.")
            return

        dialog = ColumnSortDialog(self._csv_column_names, self)
        dialog.set_sort_specs(self._csv_sort_specs)

        if dialog.exec() == QDialog.Accepted:
            self._csv_sort_specs = dialog.get_sort_specs()
            self._save_sort_to_hdf5()
            self._apply_sort()

    def _configure_filters_dialog(self) -> None:
        """Open dialog to configure column filters."""
        if not self._csv_column_names:
            QMessageBox.information(self, "No CSV Data", "Load a CSV group first.")
            return

        dialog = ColumnFilterDialog(self._csv_column_names, self)
        dialog.set_filters(self._csv_filters)

        if dialog.exec() == QDialog.Accepted:
            self._csv_filters = dialog.get_filters()
            self._save_filters_to_hdf5()
            self._apply_filters()

    def _show_statistics_dialog(self) -> None:
        """Open dialog to show statistics for CSV columns."""
        if not self._csv_column_names or (not self._csv_data_dict and not self._csv_dataset_info):
            QMessageBox.information(self, "No CSV Data", "Load a CSV group first.")
            return

        # Ensure all data is loaded before calculating statistics
        self._ensure_all_data_loaded()

        # Use filtered indices if available
        filtered_indices = (
            self._csv_filtered_indices if hasattr(self, "_csv_filtered_indices") else None
        )

        dialog = ColumnStatisticsDialog(
            self._csv_column_names, self._csv_data_dict, filtered_indices, self
        )
        dialog.exec()

    def _export_plot_to_file(self, plot_config: dict, filepath: str) -> tuple[bool, str]:
        """Export a plot configuration to a file using unified plot logic."""
        try:
            # Save current figure size
            prev_size = self.plot_figure.get_size_inches()
            # Apply plot config and update plot
            self._apply_saved_plot_config(plot_config)
            plot_options = plot_config.get("plot_options", {})
            figwidth = plot_options.get("figwidth", 8.0)
            figheight = plot_options.get("figheight", 6.0)
            dpi = plot_options.get("dpi", 100)
            # Set export size and save
            self.plot_figure.set_size_inches(figwidth, figheight)
            self.plot_figure.savefig(filepath, dpi=dpi, bbox_inches="tight")
            # Restore previous figure size and update canvas
            self.plot_figure.set_size_inches(prev_size[0], prev_size[1])
            self.updateCanvas()
            return True, ""
        except Exception as e:
            error_msg = f"Error exporting plot: {e}"
            print(error_msg)
            traceback.print_exc()
            return False, error_msg

    def _export_all_plots(self) -> None:
        """Export all saved plots to a selected directory."""
        if not self._saved_plots:
            QMessageBox.information(self, "No Plots", "There are no saved plots to export.")
            return

        # Ask user to select output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Export Plots",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly,
        )

        if not output_dir:
            return  # User cancelled

        # Create progress dialog
        progress = QProgressDialog("Exporting plots...", "Cancel", 0, len(self._saved_plots), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        # Export each plot
        exported_count = 0
        failed_plots = []

        for i, plot_config in enumerate(self._saved_plots):
            if progress.wasCanceled():
                break

            plot_name = plot_config.get("name", f"plot_{i + 1}")
            progress.setLabelText(f"Exporting: {plot_name}")
            progress.setValue(i)
            QApplication.processEvents()

            # Sanitize filename (remove invalid characters)
            safe_name = "".join(
                c if c.isalnum() or c in (" ", "-", "_") else "_" for c in plot_name
            )
            safe_name = safe_name.strip()
            if not safe_name:
                safe_name = f"plot_{i + 1}"

            # Get export format from plot options, default to PNG
            plot_options = plot_config.get("plot_options", {})
            export_format = plot_options.get("export_format", "png").lower()
            if export_format not in ["png", "pdf", "svg", "jpg", "jpeg"]:
                export_format = "png"

            filepath = os.path.join(output_dir, f"{safe_name}.{export_format}")

            # If file exists, add number suffix
            counter = 1
            while os.path.exists(filepath):
                filepath = os.path.join(output_dir, f"{safe_name}_{counter}.{export_format}")
                counter += 1

            # Export the plot
            success, error_msg = self._export_plot_to_file(plot_config, filepath)

            if success:
                exported_count += 1
            else:
                failed_plots.append((plot_name, error_msg))

        progress.setValue(len(self._saved_plots))
        progress.close()

        # Show results
        if failed_plots:
            failure_details = "\n".join([f"• {name}: {error}" for name, error in failed_plots])
            QMessageBox.warning(
                self,
                "Export Complete with Errors",
                f"Exported {exported_count} of {len(self._saved_plots)} plots to:\n{output_dir}\n\n"
                f"Failed plots:\n{failure_details}",
            )
        else:
            QMessageBox.information(
                self,
                "Export Complete",
                f"Successfully exported {exported_count} plot(s) to:\n{output_dir}",
            )

        self.statusBar().showMessage(f"Exported {exported_count} plot(s) to {output_dir}", 5000)

    def _delete_plot_config(self) -> None:
        """Delete the selected plot configuration."""
        current_row = self.saved_plots_list.currentRow()
        if current_row < 0 or current_row >= len(self._saved_plots):
            return

        plot_config = self._saved_plots[current_row]
        plot_name = plot_config.get("name", "Unnamed Plot")

        # Confirm deletion
        resp = QMessageBox.question(
            self,
            "Delete Plot Configuration",
            f"Are you sure you want to delete '{plot_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if resp != QMessageBox.Yes:
            return

        # Remove from list
        del self._saved_plots[current_row]

        # Save to HDF5
        self._save_plot_configs_to_hdf5()

        # Refresh list widget
        self._refresh_saved_plots_list()

        # Select and display previous plot if present
        new_count = len(self._saved_plots)
        if new_count > 0:
            # Select previous plot, or first if deleted first
            new_row = min(current_row, new_count - 1)
            self.saved_plots_list.setCurrentRow(new_row)
            self._apply_saved_plot(self.saved_plots_list.item(new_row))
        else:
            self._clear_plot_display()

        self.statusBar().showMessage(f"Deleted plot configuration: {plot_name}", 3000)

    def _edit_plot_options_dialog(self) -> None:
        """Open dialog to edit plot options for the selected plot configuration."""
        current_row = self.saved_plots_list.currentRow()
        if current_row < 0 or current_row >= len(self._saved_plots):
            return

        plot_config = self._saved_plots[current_row]

        model = self.preview_table.model()
        if model:
            headers = [str(model.headerData(i, Qt.Horizontal)) for i in range(model.columnCount())]
        else:
            headers = []

        # Show the options dialog (pass all headers so indices work correctly)
        dialog = PlotOptionsDialog(plot_config, headers, self)
        if dialog.exec() == QDialog.Accepted:
            # Update the configuration with the new options
            updated_config = dialog.get_plot_config()

            # Check if filters or sort changed - if so, recalculate filtered_indices
            old_filters = plot_config.get("csv_filters", [])
            old_sort = plot_config.get("csv_sort", [])
            new_filters = updated_config.get("csv_filters", [])
            new_sort = updated_config.get("csv_sort", [])

            if old_filters != new_filters or old_sort != new_sort:
                # Recalculate filtered_indices based on new filters and sort
                self._recalculate_plot_filtered_indices(updated_config)

            self._saved_plots[current_row] = updated_config

            # Save to HDF5
            self._save_plot_configs_to_hdf5()

            # Refresh the list (in case the name changed)
            self._refresh_saved_plots_list()

            # Re-select the same row
            self.saved_plots_list.setCurrentRow(current_row)

            # Reapply the plot to show the changes
            self._apply_saved_plot(None)

            self.statusBar().showMessage(
                f"Updated plot options: {updated_config.get('name', 'Unnamed')}", 3000
            )

    def _recalculate_plot_filtered_indices(self, plot_config: dict) -> None:
        """Recalculate filtered_indices for a plot based on its filter and sort settings.

        Args:
            plot_config: Plot configuration dictionary to update
        """
        if not self._csv_data_dict or not self._csv_column_names:
            return
        # Get filters and sort from plot config
        csv_filters = plot_config.get("csv_filters", [])
        csv_sort = plot_config.get("csv_sort", [])
        filtered_indices, start_row, end_row = self._get_filtered_sorted_indices(
            self._csv_data_dict, csv_filters, csv_sort
        )
        # Update plot config with new filtered indices
        if len(filtered_indices) > 0:
            plot_config["filtered_indices"] = indices_to_ranges(filtered_indices)
            plot_config["start_row"] = start_row
            plot_config["end_row"] = end_row
        else:
            plot_config["filtered_indices"] = []
            plot_config["start_row"] = 0
            plot_config["end_row"] = 0

    def _copy_plot_json_to_clipboard(self):
        """Copy the selected plot's JSON configuration to clipboard."""
        current_row = self.saved_plots_list.currentRow()
        if current_row < 0 or current_row >= len(self._saved_plots):
            return

        plot_config = self._saved_plots[current_row]

        try:
            # Convert plot config to JSON with nice formatting
            json_str = json.dumps(plot_config, indent=2, default=str)

            # Copy to clipboard
            clipboard = QApplication.clipboard()
            clipboard.setText(json_str)

            plot_name = plot_config.get("name", "Unnamed")
            self.statusBar().showMessage(f"Copied JSON for '{plot_name}' to clipboard", 3000)
        except Exception as e:
            QMessageBox.warning(
                self, "Copy Failed", f"Failed to copy plot JSON to clipboard.\n\nError: {e}"
            )

    def _on_attrs_context_menu(self, point):
        """Show context menu for attributes table."""
        item = self.attrs_table.itemAt(point)
        if item is None:
            return

        row = item.row()
        col = item.column()

        menu = QMenu(self)

        # Only show copy if a value cell is clicked (column 1)
        if col == 1:
            act_copy = menu.addAction("Copy Value")
        else:
            act_copy = None

        # Always show paste option for value cells
        if col == 1:
            act_paste = menu.addAction("Paste Value")
        else:
            act_paste = None

        if act_copy is None and act_paste is None:
            return  # No actions available

        global_pos = self.attrs_table.viewport().mapToGlobal(point)
        chosen = menu.exec(global_pos)

        if chosen == act_copy:
            # Copy full attribute value from HDF5 file (not truncated display text)
            name_item = self.attrs_table.item(row, 0)
            if not name_item:
                return

            attr_name = name_item.text()

            # Get the current object path from selection
            sel = self.tree.selectionModel().selectedIndexes()
            if not sel:
                return

            index = sel[0].sibling(sel[0].row(), 0)
            item = self.model.itemFromIndex(index)
            if item is None:
                return

            obj_path = item.data(self.model.ROLE_PATH)
            if not obj_path:
                return

            # Read the full attribute value from the HDF5 file
            fpath = self.model.filepath
            if not fpath:
                return

            try:
                with h5py.File(fpath, "r") as h5:
                    obj = h5[obj_path]
                    if attr_name in obj.attrs:
                        attr_value = obj.attrs[attr_name]
                        # Convert to string for clipboard
                        if isinstance(attr_value, (np.ndarray, list)):
                            # For arrays/lists, use repr to get full representation
                            value_str = repr(attr_value)
                        else:
                            value_str = str(attr_value)

                        clipboard = QApplication.clipboard()
                        clipboard.setText(value_str)
                        self.statusBar().showMessage(
                            "Full attribute value copied to clipboard", 2000
                        )
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to copy attribute: {exc}", 3000)
        elif chosen == act_paste:
            # Paste value from system clipboard
            clipboard = QApplication.clipboard()
            clipboard_text = clipboard.text().strip()
            if not clipboard_text:
                QMessageBox.warning(self, "Paste", "Clipboard is empty.")
                return

            # Get attribute name
            name_item = self.attrs_table.item(row, 0)
            if not name_item:
                return

            attr_name = name_item.text()

            # Get the current object path from selection
            sel = self.tree.selectionModel().selectedIndexes()
            if not sel:
                QMessageBox.warning(self, "Paste", "No item selected in tree.")
                return

            index = sel[0].sibling(sel[0].row(), 0)
            item = self.model.itemFromIndex(index)
            if item is None:
                return

            obj_path = item.data(self.model.ROLE_PATH)
            if not obj_path:
                return

            # Update the attribute in the HDF5 file
            fpath = self.model.filepath
            if not fpath:
                QMessageBox.warning(self, "Paste", "No HDF5 file loaded.")
                return

            try:
                with h5py.File(fpath, "r+") as h5:
                    obj = h5[obj_path]

                    # Simply store as string - no parsing or conversion
                    obj.attrs[attr_name] = clipboard_text

                # Update the table display
                value_item = self.attrs_table.item(row, 1)
                if value_item:
                    value_item.setText(clipboard_text)

                self.statusBar().showMessage(f"Updated attribute '{attr_name}' in HDF5 file", 3000)

                # Special handling for saved_plots attribute - recalculate filtered indices
                if attr_name == "saved_plots" and obj_path == self._current_csv_group_path:
                    try:
                        # Parse the JSON to get the updated plots
                        import json

                        plots = json.loads(clipboard_text)
                        if isinstance(plots, list):
                            # Update each plot's filtered_indices based on its filters/sort
                            for plot_config in plots:
                                if plot_config.get("csv_filters") or plot_config.get("csv_sort"):
                                    self._recalculate_plot_filtered_indices(plot_config)

                            # Save the updated plots back to HDF5
                            with h5py.File(fpath, "r+") as h5:
                                if obj_path in h5:
                                    h5[obj_path].attrs[attr_name] = json.dumps(plots)

                            # Reload the plots in the UI
                            self._saved_plots = plots
                            self._refresh_saved_plots_list()

                            self.statusBar().showMessage(
                                f"Updated {len(plots)} plot(s) and recalculated filtered indices",
                                3000,
                            )
                    except Exception as e:
                        print(f"Warning: Could not recalculate plot filtered indices: {e}")

                # Refresh the display to reflect any changes
                kind = item.data(self.model.ROLE_KIND)
                if kind == "dataset":
                    self.preview_dataset(obj_path)
                elif kind == "group":
                    self.preview_group(obj_path)
                elif kind == "attr":
                    # If we're on an attr node, refresh its parent
                    parent_path = item.data(self.model.ROLE_PATH)
                    if parent_path:
                        # Determine if parent is dataset or group
                        try:
                            with h5py.File(fpath, "r") as h5:
                                parent_obj = h5[parent_path]
                                if isinstance(parent_obj, h5py.Dataset):
                                    self.preview_dataset(parent_path)
                                elif isinstance(parent_obj, h5py.Group):
                                    self.preview_group(parent_path)
                        except Exception:
                            pass

            except Exception as exc:
                QMessageBox.critical(self, "Paste Failed", f"Failed to update attribute:\n{exc}")

    def _show_dag_visualization_pyqtgraph(self, dataset_path: str | None = None) -> None:
        """
        Visualize the HDF5 file structure as a DAG using pyqtgraph (interactive), in a separate dialog window.

        Args:
            dataset_path (str | None): Optional path to a specific dataset or group within the HDF5 file. If provided, the DAG will be constructed starting from this path; otherwise, the DAG will represent the entire file structure.
        """

        from qtpy.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QHBoxLayout,
            QPushButton,
            QFileDialog,
            QToolTip,
            QMessageBox,
            QComboBox,
            QLabel,
            QScrollArea,
        )
        import pyqtgraph as pg
        from pyqtgraph.exporters import ImageExporter
        import networkx as nx
        import numpy as np
        import os

        # Custom GraphItem with tooltip support
        class TooltipGraphItem(pg.GraphItem):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._positions = None

            def set_positions(self, positions):
                self._positions = positions

            def hoverEvent(self, event):
                if event.isExit():
                    QToolTip.hideText()
                    return
                if self._positions is None:
                    return
                pos = event.pos()
                x, y = pos.x(), pos.y()
                dists = np.linalg.norm(self._positions - np.array([x, y]), axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < 0.05:
                    tip = node_tooltip(min_idx)
                    QToolTip.showText(event.screenPos().toPoint(), tip)
                else:
                    QToolTip.hideText()

        fpath = self.model.filepath
        if not fpath:
            QMessageBox.warning(self, "No file", "No HDF5 file is loaded.")
            return

        try:
            fpath = self.model.filepath
            if not fpath:
                QMessageBox.warning(self, "No file", "No HDF5 file is loaded.")
                return
            with h5py.File(fpath, "r") as h5:
                G = nx.DiGraph()

                def add_group(g, parent=None):
                    group_id = f"group:{g.name}"
                    is_csv = False
                    if g.name == "/":
                        label = "root"
                        kind = "root"
                    else:
                        label = g.name.split("/")[-1]
                        kind = "group"
                        if "source_type" in g.attrs and g.attrs["source_type"] == "csv":
                            is_csv = True
                            kind = "csv"
                    G.add_node(group_id, label=label, kind=kind)
                    if parent:
                        G.add_edge(parent, group_id)
                    for key in g:
                        item = g[key]
                        if isinstance(item, h5py.Group):
                            add_group(item, group_id)
                        else:
                            ds_id = f"dataset:{item.name}"
                            ds_label = item.name.split("/")[-1]
                            ds_kind = "csv_dataset" if is_csv else "dataset"
                            G.add_node(ds_id, label=ds_label, kind=ds_kind)
                            G.add_edge(group_id, ds_id)

                add_group(h5[dataset_path] if dataset_path else h5)

            # different layout options from networkx:
            layout_options = {
                "Pygraphviz (neato)": lambda G: nx.nx_agraph.pygraphviz_layout(G, prog="neato"),
                "Pygraphviz (dot)": lambda G: nx.nx_agraph.pygraphviz_layout(G, prog="dot"),
                "Spring": lambda G: nx.spring_layout(G, k=1.5, iterations=100),
                "Circular": nx.circular_layout,
                "Shell": nx.shell_layout,
                "Kamada-Kawai": nx.kamada_kawai_layout,
                "Spectral": nx.spectral_layout,
                "Random": nx.random_layout,
                "ForceAtlas2": nx.forceatlas2_layout,
                "Planar": nx.planar_layout,
                "BFS": lambda G: nx.bfs_layout(G, start="group:/"),
                "ARF": lambda G: nx.arf_layout(G),
                "Spiral": lambda G: nx.spiral_layout(G),
            }

            # Create a QDialog to serve as the DAG window
            dag_dialog = QDialog(self)
            dag_dialog.setWindowTitle("DAG Visualization")
            main_layout = QVBoxLayout(dag_dialog)

            # Layout selection dropdown
            layout_select_layout = QHBoxLayout()
            layout_label = QLabel("Layout:")
            layout_combo = QComboBox()
            layout_combo.addItems(list(layout_options.keys()))
            layout_select_layout.addWidget(layout_label)
            layout_select_layout.addWidget(layout_combo)
            main_layout.addLayout(layout_select_layout)

            plot_widget = pg.GraphicsLayoutWidget()
            graph_item = None
            plot_widget.setMinimumSize(100, 100)
            scroll_area = QScrollArea()
            scroll_area.setWidget(plot_widget)
            scroll_area.setWidgetResizable(True)

            # Custom GraphItem with tooltip support
            class TooltipGraphItem(pg.GraphItem):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    self._positions = None

                def set_positions(self, positions):
                    self._positions = positions

                def hoverEvent(self, event):
                    if event.isExit():
                        QToolTip.hideText()
                        return
                    if self._positions is None:
                        return
                    pos = event.pos()
                    x, y = pos.x(), pos.y()
                    dists = np.linalg.norm(self._positions - np.array([x, y]), axis=1)
                    min_idx = np.argmin(dists)
                    if dists[min_idx] < 0.05:
                        tip = node_tooltip(min_idx)
                        QToolTip.showText(event.screenPos().toPoint(), tip)
                    else:
                        QToolTip.hideText()

            # Node info tooltip helper
            def node_tooltip(idx):
                if idx is None or idx < 0 or idx >= len(node_ids):
                    return ""
                node_id = node_ids[idx]
                label = node_labels[idx]
                kind = node_kinds[idx]
                # If this is a dataset node, show rich info
                if kind in ("dataset", "csv_dataset"):
                    # node_id is like 'dataset:/path/to/dataset'
                    if node_id.startswith("dataset:"):
                        dataset_path = node_id[len("dataset:") :]
                        info = self._get_dataset_info(dataset_path)
                        if info:
                            # Show key info in tooltip
                            lines = [f"<b>{label}</b>", f"<i>{dataset_path}</i>"]
                            for k in [
                                "Shape",
                                "Data Type",
                                "Size",
                                "Memory Size",
                                "Storage Size",
                                "Compression",
                                "Min Value",
                                "Max Value",
                                "Mean Value",
                                "Std Dev",
                                "Attributes",
                                "Attribute Names",
                            ]:
                                if k in info:
                                    lines.append(f"<b>{k}:</b> {info[k]}")
                            return "<br>".join(lines)
                # Otherwise, show basic info
                return f"<b>{label}</b><br><i>{node_id}</i><br>Type: {kind}"

            # Prepare node and edge data for pyqtgraph
            node_ids = list(G.nodes)
            node_labels = [G.nodes[n].get("label", n) for n in node_ids]
            node_kinds = [G.nodes[n].get("kind", "group") for n in node_ids]
            edges = [(node_ids.index(e[0]), node_ids.index(e[1])) for e in G.edges]

            # Map node kind to symbol
            kind_to_symbol = {
                "root": "o",  # circle
                "group": "s",  # square
                "csv": "t",  # triangle
                "csv_dataset": "d",  # diamond
                "dataset": "h",  # hexagon
            }
            node_symbols = [kind_to_symbol.get(k, "o") for k in node_kinds]

            def get_color(kind):
                if kind == "root":
                    return (210, 207, 184, 255)
                elif kind == "csv":
                    return (237, 222, 240, 255)
                elif kind == "group":
                    return (224, 247, 250, 255)
                elif kind == "csv_dataset":
                    return (255, 228, 228, 255)
                else:
                    return (254, 255, 245, 255)

            node_colors = np.array([get_color(k) for k in node_kinds], dtype=np.ubyte)

            text_items = []
            positions = None
            view = pg.ViewBox()
            plot_widget.addItem(view)
            view.setAspectLocked()
            main_layout.addWidget(scroll_area)

            def update_layout():
                nonlocal positions, graph_item
                view.clear()
                # Get selected layout
                layout_name = layout_combo.currentText()
                pos_dict = layout_options[layout_name](G)
                positions = np.array([pos_dict[n] for n in node_ids])
                if "pygraphviz" in layout_name.lower():
                    # Normalize positions for pygraphviz layout
                    # [this is so the mouseover logic will work for the tooltips]
                    min_xy = positions.min(axis=0)
                    max_xy = positions.max(axis=0)
                    span_xy = max_xy - min_xy
                    span_xy[span_xy == 0] = 1.0  # Avoid division by zero
                    positions = (positions - min_xy) / span_xy
                    positions = positions * 2.0 - 1.0  # Scale to [-1, 1] for both axes
                # Add node labels as text items (below nodes)
                for i, (x, y) in enumerate(positions):
                    label = node_labels[i]
                    text_item = pg.TextItem(label, anchor=(0.5, -0.2), color=(255, 255, 255))
                    text_item.setPos(x, y)
                    view.addItem(text_item)
                    text_items.append(text_item)
                graph_item = TooltipGraphItem()
                graph_item.setData(
                    pos=positions,
                    adj=np.array(edges),
                    size=18,
                    symbol=node_symbols,
                    pxMode=True,
                    text=node_labels,
                    pen={"color": (255, 255, 255, 255), "width": 2},  # White edge lines
                    brush=node_colors,
                )
                graph_item.set_positions(positions)
                view.addItem(graph_item)
                # Use pyqtgraph's built-in 'View All' feature to recenter the plot
                view.enableAutoRange()
                view.autoRange()

            # Initial plot
            update_layout()

            # Update plot when layout changes
            layout_combo.currentIndexChanged.connect(update_layout)

            # Add Save and Close buttons
            btn_layout = QHBoxLayout()
            save_btn = QPushButton("Save As...")
            close_btn = QPushButton("Close")
            btn_layout.addWidget(save_btn)
            btn_layout.addWidget(close_btn)
            main_layout.addLayout(btn_layout)

            def save_dag_image():
                file_path, _ = QFileDialog.getSaveFileName(
                    dag_dialog,
                    "Save DAG Image",
                    os.path.splitext(self.model.filepath)[0] + "_dag.png",
                    "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg)",
                )
                if file_path:
                    exporter = ImageExporter(plot_widget.scene())
                    exporter.export(file_path)
                    QMessageBox.information(
                        dag_dialog, "Saved", f"DAG image saved to:\n{file_path}"
                    )

            save_btn.clicked.connect(save_dag_image)
            close_btn.clicked.connect(dag_dialog.close)
            dag_dialog.resize(900, 700)
            dag_dialog.exec()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to generate DAG visualization: {exc}")

    def _show_dag_visualization(self) -> None:
        """Visualize the HDF5 file structure as a DAG using python-graphviz."""
        import tempfile, os
        import h5py
        import graphviz
        from qtpy.QtGui import QPixmap
        from qtpy.QtWidgets import (
            QLabel,
            QDialog,
            QVBoxLayout,
            QPushButton,
            QScrollArea,
            QHBoxLayout,
            QMessageBox,
        )

        try:
            fpath = self.model.filepath
            if not fpath:
                QMessageBox.warning(self, "No file", "No HDF5 file is loaded.")
                return
            dot = graphviz.Digraph(comment="HDF5 DAG")
            fontcolor = "#222"

            def sanitize_id(name: str) -> str:
                return name.replace("/", "_").replace(":", "_")

            with h5py.File(fpath, "r") as h5:

                def add_group(g, parent_id=None):
                    group_id = sanitize_id(f"group:{g.name}")
                    is_csv = False
                    if g.name == "/":
                        label = "root"
                        shape = "folder"
                        fillcolor = "#d2cfb8"
                    else:
                        label = g.name.split("/")[-1]
                        if "source_type" in g.attrs and g.attrs["source_type"] == "csv":
                            is_csv = True
                            shape = "box3d"
                            fillcolor = "#eddef0"
                        else:
                            shape = "folder"
                            fillcolor = "#e0f7fa"
                    dot.node(
                        group_id,
                        label,
                        shape=shape,
                        style="filled",
                        fillcolor=fillcolor,
                        fontcolor=fontcolor,
                    )
                    if parent_id:
                        dot.edge(parent_id, group_id)
                    for key in g:
                        item = g[key]
                        if isinstance(item, h5py.Group):
                            add_group(item, group_id)
                        else:
                            ds_id = sanitize_id(f"dataset:{item.name}")
                            ds_label = item.name.split("/")[-1]
                            if is_csv:
                                dot.node(
                                    ds_id,
                                    ds_label,
                                    shape="ellipse",
                                    style="filled",
                                    fillcolor="#ffe4e4",
                                )
                            else:
                                dot.node(
                                    ds_id,
                                    ds_label,
                                    shape="box",
                                    style="filled",
                                    fillcolor="#fefff5",
                                )
                            dot.edge(group_id, ds_id)

                add_group(h5)

            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = f"{tmpdir}/hdf5_dag"
                dot.format = "png"
                dot.render(filename=out_path, cleanup=True)
                png_path = out_path + ".png"
                pixmap = QPixmap(png_path)
                dialog = QDialog(self)
                dialog.setWindowTitle("HDF5 DAG Visualization")
                layout = QVBoxLayout(dialog)
                scroll_area = QScrollArea(dialog)
                label = QLabel()
                if not pixmap or pixmap.isNull():
                    label.setText("Failed to load DAG image.")
                else:
                    label.setPixmap(pixmap)
                    label.setAlignment(Qt.AlignCenter)
                scroll_area.setWidget(label)
                scroll_area.setWidgetResizable(True)
                layout.addWidget(scroll_area)

                btn_layout = QHBoxLayout()
                save_btn = QPushButton("Save As...")
                close_btn = QPushButton("Close")
                btn_layout.addWidget(save_btn)
                btn_layout.addWidget(close_btn)
                layout.addLayout(btn_layout)

                def save_dag_image():
                    file_path, _ = QFileDialog.getSaveFileName(
                        dialog,
                        "Save DAG Image",
                        os.path.splitext(self.model.filepath)[0] + "_dag.png",
                        "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;SVG Image (*.svg);;PDF File (*.pdf)",
                    )
                    if file_path:
                        dot.format = os.path.splitext(file_path)[1].lower().strip(".")
                        try:
                            dot.render(filename=os.path.splitext(file_path)[0], cleanup=True)
                            QMessageBox.information(
                                dialog, "Saved", f"DAG image saved to:\n{file_path}"
                            )
                        except Exception as exc:
                            QMessageBox.critical(
                                dialog, "Save Failed", f"Failed to save DAG image:\n{exc}"
                            )

                save_btn.clicked.connect(save_dag_image)
                close_btn.clicked.connect(dialog.accept)
                dialog.resize(900, 700)
                dialog.exec()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to generate DAG visualization: {exc}")

    def _duplicate_plot_config(self):
        """Duplicate the selected plot configuration."""
        current_row = self.saved_plots_list.currentRow()
        if current_row < 0 or current_row >= len(self._saved_plots):
            return

        plot_config = self._saved_plots[current_row]
        plot_name = plot_config.get("name", "Unnamed Plot")

        # Create a deep copy of the plot config
        import copy

        duplicated_config = copy.deepcopy(plot_config)

        # Update the name to indicate it's a copy
        duplicated_config["name"] = f"{plot_name} (copy)"
        duplicated_config["timestamp"] = time.time()

        # Add to the end of the list
        self._saved_plots.append(duplicated_config)

        # Save to HDF5
        self._save_plot_configs_to_hdf5()

        # Refresh list widget
        self._refresh_saved_plots_list()

        # Select the new duplicated plot
        self.saved_plots_list.setCurrentRow(len(self._saved_plots) - 1)

        self.statusBar().showMessage(f"Duplicated plot: {plot_name}", 3000)

    def _on_saved_plots_context_menu(self, point):
        """Show context menu for saved plots list."""
        item = self.saved_plots_list.itemAt(point)

        menu = QMenu(self)

        # Get standard icon theme
        style = self.style()

        # Actions that require a selected item
        if item is not None:
            act_duplicate = menu.addAction("Duplicate Plot")
            act_copy_json = menu.addAction("Copy JSON to Clipboard")
            menu.addSeparator()
            act_delete = menu.addAction("Delete Plot")
            act_delete.setIcon(style.standardIcon(QStyle.SP_TrashIcon))
            menu.addSeparator()

        # Export all plots action (always available if plots exist)
        act_export_all = None
        if len(self._saved_plots) > 0:
            act_export_all = menu.addAction("Export All Plots...")

        # If no actions available, don't show menu
        if menu.isEmpty():
            return

        global_pos = self.saved_plots_list.viewport().mapToGlobal(point)
        chosen = menu.exec(global_pos)

        if item is not None:
            if chosen == act_duplicate:
                self._duplicate_plot_config()
            elif chosen == act_copy_json:
                self._copy_plot_json_to_clipboard()
            elif chosen == act_delete:
                self._delete_plot_config()

        if chosen == act_export_all:
            self._export_all_plots()
