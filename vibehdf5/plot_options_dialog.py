import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib as mpl

from qtpy.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QComboBox, QTabWidget, QWidget, QScrollArea, QFrame, QDoubleSpinBox, QSpinBox, QPushButton, QDialogButtonBox, QColorDialog
from qtpy.QtGui import QColor, QDoubleValidator

from .column_sort_dialog import ColumnSortDialog
from .column_filter_dialog import ColumnFilterDialog


class PlotOptionsDialog(QDialog):
    """Dialog for configuring plot options (title, labels, line styles, etc.)."""

    # Available line styles and colors
    LINE_STYLES = ["-", "--", "-.", ":", "None"]
    LINE_STYLE_NAMES = ["Solid", "Dashed", "Dash-dot", "Dotted", "None"]
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]   # matplotlib's default color cycle
    MARKERS = ["", "o", "s", "^", "v", "D", "*", "+", "x", "."]
    MARKER_NAMES = [
        "None",
        "Circle",
        "Square",
        "Triangle Up",
        "Triangle Down",
        "Diamond",
        "Star",
        "Plus",
        "X",
        "Point",
    ]
    PLOT_TYPES = [
        ("line", "Line Plot"),
        ("contourf", "Filled Contour Plot"),
    ]

    def __init__(self, plot_config, column_names, parent=None):
        """Initialize the plot options dialog.

        Args:
            plot_config: Dictionary containing plot configuration
            column_names: List of available column names
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Plot Options")
        self.resize(600, 500)

        self.plot_config = plot_config.copy()  # Work on a copy
        self.column_names = column_names

        layout = QVBoxLayout(self)

        # Create tab widget for different option categories
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: General options
        general_tab = QWidget()
        general_layout = QVBoxLayout(general_tab)

        # Plot type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Plot Type:"))
        self.type_combo = QComboBox()
        for type_val, type_name in self.PLOT_TYPES:
            self.type_combo.addItem(type_name, type_val)
        # Set current value from config
        current_type = self.plot_config.get("plot_options", {}).get("type", "line")
        idx = self.type_combo.findData(current_type)
        if idx >= 0:
            self.type_combo.setCurrentIndex(idx)
        type_layout.addWidget(self.type_combo)
        type_layout.addStretch()
        general_layout.addLayout(type_layout)

        # Plot name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Plot Name:"))
        self.name_edit = QLineEdit(self.plot_config.get("name", "Plot"))
        name_layout.addWidget(self.name_edit)
        general_layout.addLayout(name_layout)

        # Title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Plot Title:"))
        self.title_edit = QLineEdit(self.plot_config.get("plot_options", {}).get("title", ""))
        self.title_edit.setPlaceholderText("Auto-generated from CSV group name")
        title_layout.addWidget(self.title_edit)
        general_layout.addLayout(title_layout)

        # X-axis label
        xlabel_layout = QHBoxLayout()
        xlabel_layout.addWidget(QLabel("X-axis Label:"))
        self.xlabel_edit = QLineEdit(self.plot_config.get("plot_options", {}).get("xlabel", ""))
        self.xlabel_edit.setPlaceholderText("Auto-generated from column name")
        xlabel_layout.addWidget(self.xlabel_edit)
        general_layout.addLayout(xlabel_layout)

        # Y-axis label
        ylabel_layout = QHBoxLayout()
        ylabel_layout.addWidget(QLabel("Y-axis Label:"))
        self.ylabel_edit = QLineEdit(self.plot_config.get("plot_options", {}).get("ylabel", ""))
        self.ylabel_edit.setPlaceholderText("Auto-generated from column names")
        ylabel_layout.addWidget(self.ylabel_edit)
        general_layout.addLayout(ylabel_layout)

        # Grid options
        grid_group = QWidget()
        grid_layout = QHBoxLayout(grid_group)
        grid_layout.setContentsMargins(0, 10, 0, 10)

        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(self.plot_config.get("plot_options", {}).get("grid", True))
        grid_layout.addWidget(self.grid_checkbox)

        self.legend_checkbox = QCheckBox("Show Legend")
        self.legend_checkbox.setChecked(
            self.plot_config.get("plot_options", {}).get("legend", True)
        )
        grid_layout.addWidget(self.legend_checkbox)

        grid_layout.addWidget(QLabel("Legend Position:"))
        self.legend_loc_combo = QComboBox()
        # Matplotlib legend location options
        legend_locations = [
            ("best", "Best"),
            ("upper right", "Upper Right"),
            ("upper left", "Upper Left"),
            ("lower left", "Lower Left"),
            ("lower right", "Lower Right"),
            ("right", "Right"),
            ("center left", "Center Left"),
            ("center right", "Center Right"),
            ("lower center", "Lower Center"),
            ("upper center", "Upper Center"),
            ("center", "Center"),
        ]
        for loc_value, loc_name in legend_locations:
            self.legend_loc_combo.addItem(loc_name, loc_value)
        # Set current value
        current_loc = self.plot_config.get("plot_options", {}).get("legend_loc", "best")
        index = self.legend_loc_combo.findData(current_loc)
        if index >= 0:
            self.legend_loc_combo.setCurrentIndex(index)
        grid_layout.addWidget(self.legend_loc_combo)

        self.dark_background_checkbox = QCheckBox("Dark Background")
        self.dark_background_checkbox.setChecked(
            self.plot_config.get("plot_options", {}).get("dark_background", False)
        )
        grid_layout.addWidget(self.dark_background_checkbox)

        grid_layout.addStretch()
        general_layout.addWidget(grid_group)

        # Axis limits section
        limits_label = QLabel("<b>Axis Limits:</b>")
        general_layout.addWidget(limits_label)

        # X-axis limits
        xlim_layout = QHBoxLayout()
        xlim_layout.addWidget(QLabel("X-axis:"))
        xlim_layout.addWidget(QLabel("Min:"))
        self.xlim_min_edit = QLineEdit()
        self.xlim_min_edit.setPlaceholderText("auto")
        self.xlim_min_edit.setMaximumWidth(100)
        xlim_min_val = self.plot_config.get("plot_options", {}).get("xlim_min", "")
        if xlim_min_val not in (None, ""):
            self.xlim_min_edit.setText(str(xlim_min_val))
        xlim_layout.addWidget(self.xlim_min_edit)

        xlim_layout.addWidget(QLabel("Max:"))
        self.xlim_max_edit = QLineEdit()
        self.xlim_max_edit.setPlaceholderText("auto")
        self.xlim_max_edit.setMaximumWidth(100)
        xlim_max_val = self.plot_config.get("plot_options", {}).get("xlim_max", "")
        if xlim_max_val not in (None, ""):
            self.xlim_max_edit.setText(str(xlim_max_val))
        xlim_layout.addWidget(self.xlim_max_edit)

        xlim_layout.addStretch()
        general_layout.addLayout(xlim_layout)

        # Y-axis limits
        ylim_layout = QHBoxLayout()
        ylim_layout.addWidget(QLabel("Y-axis:"))
        ylim_layout.addWidget(QLabel("Min:"))
        self.ylim_min_edit = QLineEdit()
        self.ylim_min_edit.setPlaceholderText("auto")
        self.ylim_min_edit.setMaximumWidth(100)
        ylim_min_val = self.plot_config.get("plot_options", {}).get("ylim_min", "")
        if ylim_min_val not in (None, ""):
            self.ylim_min_edit.setText(str(ylim_min_val))
        ylim_layout.addWidget(self.ylim_min_edit)

        ylim_layout.addWidget(QLabel("Max:"))
        self.ylim_max_edit = QLineEdit()
        self.ylim_max_edit.setPlaceholderText("auto")
        self.ylim_max_edit.setMaximumWidth(100)
        ylim_max_val = self.plot_config.get("plot_options", {}).get("ylim_max", "")
        if ylim_max_val not in (None, ""):
            self.ylim_max_edit.setText(str(ylim_max_val))
        ylim_layout.addWidget(self.ylim_max_edit)

        ylim_layout.addStretch()
        general_layout.addLayout(ylim_layout)

        # Log scale options
        log_scale_label = QLabel("<b>Logarithmic Scale:</b>")
        general_layout.addWidget(log_scale_label)

        log_scale_layout = QHBoxLayout()
        log_scale_layout.setContentsMargins(0, 5, 0, 10)

        self.xlog_checkbox = QCheckBox("X-axis Log Scale")
        self.xlog_checkbox.setChecked(
            self.plot_config.get("plot_options", {}).get("xlog", False)
        )
        log_scale_layout.addWidget(self.xlog_checkbox)

        self.ylog_checkbox = QCheckBox("Y-axis Log Scale")
        self.ylog_checkbox.setChecked(
            self.plot_config.get("plot_options", {}).get("ylog", False)
        )
        log_scale_layout.addWidget(self.ylog_checkbox)

        log_scale_layout.addStretch()
        general_layout.addWidget(QWidget())  # Spacer
        general_layout.addLayout(log_scale_layout)

        # Date/Time X-axis options
        datetime_label = QLabel("<b>Date/Time X-axis:</b>")
        general_layout.addWidget(datetime_label)

        datetime_layout = QVBoxLayout()
        datetime_layout.setContentsMargins(0, 5, 0, 10)

        self.xaxis_datetime_checkbox = QCheckBox("X-axis is Date/Time")
        self.xaxis_datetime_checkbox.setChecked(
            self.plot_config.get("plot_options", {}).get("xaxis_datetime", False)
        )
        datetime_layout.addWidget(self.xaxis_datetime_checkbox)

        # Date format input
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Date Format:"))
        self.datetime_format_edit = QLineEdit()
        self.datetime_format_edit.setPlaceholderText("%Y-%m-%d %H:%M:%S")
        datetime_format = self.plot_config.get("plot_options", {}).get("datetime_format", "")
        if datetime_format:
            self.datetime_format_edit.setText(datetime_format)
        self.datetime_format_edit.setToolTip(
            "Python datetime format string (e.g., %Y-%m-%d, %Y-%m-%d %H:%M:%S, %m/%d/%Y)\n"
            "Common codes: %Y=year, %m=month, %d=day, %H=hour, %M=minute, %S=second"
        )
        format_layout.addWidget(self.datetime_format_edit)
        format_layout.addStretch()
        datetime_layout.addLayout(format_layout)

        # Date display format
        display_format_layout = QHBoxLayout()
        display_format_layout.addWidget(QLabel("Display Format:"))
        self.datetime_display_format_edit = QLineEdit()
        self.datetime_display_format_edit.setPlaceholderText("%Y-%m-%d")
        datetime_display_format = self.plot_config.get("plot_options", {}).get("datetime_display_format", "")
        if datetime_display_format:
            self.datetime_display_format_edit.setText(datetime_display_format)
        self.datetime_display_format_edit.setToolTip(
            "Format for axis labels (e.g., %Y-%m-%d, %b %d, %m/%d)\n"
            "Leave empty to use matplotlib's automatic formatting"
        )
        display_format_layout.addWidget(self.datetime_display_format_edit)
        display_format_layout.addStretch()
        datetime_layout.addLayout(display_format_layout)

        general_layout.addLayout(datetime_layout)

        general_layout.addStretch()
        self.tabs.addTab(general_tab, "General")

        # Tab 2: Figure Size & Export
        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)

        # Figure size options
        figsize_label = QLabel("<b>Figure Size:</b>")
        export_layout.addWidget(figsize_label)

        figsize_layout = QHBoxLayout()
        figsize_layout.setContentsMargins(0, 5, 0, 10)

        figsize_layout.addWidget(QLabel("Width:"))
        self.figwidth_spin = QDoubleSpinBox()
        self.figwidth_spin.setRange(1.0, 50.0)
        self.figwidth_spin.setSingleStep(0.5)
        self.figwidth_spin.setValue(self.plot_config.get("plot_options", {}).get("figwidth", 8.0))
        self.figwidth_spin.setSuffix(" in")
        self.figwidth_spin.setToolTip("Figure width in inches")
        self.figwidth_spin.setMinimumWidth(100)
        figsize_layout.addWidget(self.figwidth_spin)

        figsize_layout.addWidget(QLabel("Height:"))
        self.figheight_spin = QDoubleSpinBox()
        self.figheight_spin.setRange(1.0, 50.0)
        self.figheight_spin.setSingleStep(0.5)
        self.figheight_spin.setValue(self.plot_config.get("plot_options", {}).get("figheight", 6.0))
        self.figheight_spin.setSuffix(" in")
        self.figheight_spin.setToolTip("Figure height in inches")
        self.figheight_spin.setMinimumWidth(100)
        figsize_layout.addWidget(self.figheight_spin)

        figsize_layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(50, 600)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setValue(self.plot_config.get("plot_options", {}).get("dpi", 100))
        self.dpi_spin.setToolTip("Dots per inch for export")
        self.dpi_spin.setMinimumWidth(100)
        figsize_layout.addWidget(self.dpi_spin)

        figsize_layout.addStretch()
        export_layout.addLayout(figsize_layout)

        # Export format options
        export_format_label = QLabel("<b>Export Format:</b>")
        export_layout.addWidget(export_format_label)

        export_format_layout = QHBoxLayout()
        export_format_layout.setContentsMargins(0, 5, 0, 10)
        export_format_layout.addWidget(QLabel("File Format:"))
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["png", "pdf", "svg", "jpg", "eps"])
        current_format = self.plot_config.get("plot_options", {}).get("export_format", "png")
        format_idx = self.export_format_combo.findText(current_format)
        if format_idx >= 0:
            self.export_format_combo.setCurrentIndex(format_idx)
        self.export_format_combo.setToolTip("File format for drag-and-drop export")
        self.export_format_combo.setMinimumWidth(100)
        export_format_layout.addWidget(self.export_format_combo)
        export_format_layout.addStretch()
        export_layout.addLayout(export_format_layout)

        export_layout.addStretch()
        self.tabs.addTab(export_tab, "Figure & Export")

        # Tab 3: Fonts
        fonts_tab = QWidget()
        fonts_layout = QVBoxLayout(fonts_tab)

        # Font sizes
        font_size_label = QLabel("<b>Font Sizes:</b>")
        fonts_layout.addWidget(font_size_label)

        font_size_layout = QHBoxLayout()
        font_size_layout.setContentsMargins(0, 5, 0, 10)

        font_size_layout.addWidget(QLabel("Title:"))
        self.title_fontsize_spin = QSpinBox()
        self.title_fontsize_spin.setRange(6, 72)
        self.title_fontsize_spin.setValue(self.plot_config.get("plot_options", {}).get("title_fontsize", 12))
        self.title_fontsize_spin.setSuffix(" pt")
        self.title_fontsize_spin.setToolTip("Font size for plot title")
        self.title_fontsize_spin.setMinimumWidth(80)
        font_size_layout.addWidget(self.title_fontsize_spin)

        font_size_layout.addWidget(QLabel("Axis Labels:"))
        self.axis_label_fontsize_spin = QSpinBox()
        self.axis_label_fontsize_spin.setRange(6, 72)
        self.axis_label_fontsize_spin.setValue(self.plot_config.get("plot_options", {}).get("axis_label_fontsize", 10))
        self.axis_label_fontsize_spin.setSuffix(" pt")
        self.axis_label_fontsize_spin.setToolTip("Font size for axis labels")
        self.axis_label_fontsize_spin.setMinimumWidth(80)
        font_size_layout.addWidget(self.axis_label_fontsize_spin)

        font_size_layout.addWidget(QLabel("Tick Labels:"))
        self.tick_fontsize_spin = QSpinBox()
        self.tick_fontsize_spin.setRange(6, 72)
        self.tick_fontsize_spin.setValue(self.plot_config.get("plot_options", {}).get("tick_fontsize", 9))
        self.tick_fontsize_spin.setSuffix(" pt")
        self.tick_fontsize_spin.setToolTip("Font size for axis tick labels")
        self.tick_fontsize_spin.setMinimumWidth(80)
        font_size_layout.addWidget(self.tick_fontsize_spin)

        font_size_layout.addWidget(QLabel("Legend:"))
        self.legend_fontsize_spin = QSpinBox()
        self.legend_fontsize_spin.setRange(6, 72)
        self.legend_fontsize_spin.setValue(self.plot_config.get("plot_options", {}).get("legend_fontsize", 9))
        self.legend_fontsize_spin.setSuffix(" pt")
        self.legend_fontsize_spin.setToolTip("Font size for legend text")
        self.legend_fontsize_spin.setMinimumWidth(80)
        font_size_layout.addWidget(self.legend_fontsize_spin)

        font_size_layout.addStretch()
        fonts_layout.addLayout(font_size_layout)

        # Font family
        font_family_label = QLabel("<b>Font Family:</b>")
        fonts_layout.addWidget(font_family_label)

        font_family_layout = QHBoxLayout()
        font_family_layout.setContentsMargins(0, 5, 0, 10)
        font_family_layout.addWidget(QLabel("Family:"))
        self.font_family_combo = QComboBox()
        self.font_family_combo.addItems(["serif", "sans-serif", "monospace", "cursive", "fantasy"])
        current_family = self.plot_config.get("plot_options", {}).get("font_family", "serif")
        family_idx = self.font_family_combo.findText(current_family)
        if family_idx >= 0:
            self.font_family_combo.setCurrentIndex(family_idx)
        self.font_family_combo.setToolTip("Font family for all plot text")
        self.font_family_combo.setMinimumWidth(150)
        font_family_layout.addWidget(self.font_family_combo)
        font_family_layout.addStretch()
        fonts_layout.addLayout(font_family_layout)

        fonts_layout.addStretch()
        self.tabs.addTab(fonts_tab, "Fonts")

        # Store tab references for dynamic switching
        self._contour_tab = None
        self._series_tab = None
        self._reflines_tab = None

        def add_contour_tab():
            contour_tab = QWidget()
            contour_layout = QVBoxLayout(contour_tab)
            contour_layout.addWidget(QLabel("Colormap (cmap):"))
            self.cmap_combo = QComboBox()
            # see https://matplotlib.org/stable/users/explain/colors/colormaps.html
            cmaps = [
                "viridis", "plasma", "inferno", "magma", "cividis",
                "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
                "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu", "GnBu", "PuBu", "YlGnBu", "PuBuGn", "YlGn",
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper',
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
                'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic',
                'berlin', 'managua', 'vanimo',
                'twilight', 'twilight_shifted', 'hsv',
                'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                'tab20c',
                'flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
                'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
                'turbo', 'nipy_spectral', 'gist_ncar'
            ]
            self.cmap_combo.addItems(cmaps)
            current_cmap = self.plot_config.get("plot_options", {}).get("cmap", "Blues")
            idx = self.cmap_combo.findText(current_cmap)
            if idx >= 0:
                self.cmap_combo.setCurrentIndex(idx)
            contour_layout.addWidget(self.cmap_combo)

            # Add colorbar preview
            self.cmap_colorbar_fig = Figure(figsize=(3, 0.4), dpi=100)
            self.cmap_colorbar_canvas = FigureCanvas(self.cmap_colorbar_fig)
            contour_layout.addWidget(self.cmap_colorbar_canvas)

            # Add cmap label field
            cmap_label_layout = QHBoxLayout()
            cmap_label_layout.addWidget(QLabel("Colorbar Label:"))
            self.cmap_label_edit = QLineEdit()
            self.cmap_label_edit.setPlaceholderText("Label for colorbar (e.g. Z series name)")
            current_label = self.plot_config.get("plot_options", {}).get("cmap_label", "")
            self.cmap_label_edit.setText(current_label)
            cmap_label_layout.addWidget(self.cmap_label_edit)
            contour_layout.addLayout(cmap_label_layout)


            def update_colorbar():
                cmap_name = self.cmap_combo.currentText()
                self.cmap_colorbar_fig.clear()
                ax = self.cmap_colorbar_fig.add_subplot(111)
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                cb = mpl.colorbar.ColorbarBase(ax, cmap=mpl.colormaps[cmap_name], norm=norm, orientation='horizontal')
                ax.set_xticks([])
                ax.set_yticks([])
                # Use label from field
                label = self.cmap_label_edit.text()
                cb.set_label(label, fontsize=9)
                ax.set_title(cmap_name, fontsize=8)
                self.cmap_colorbar_canvas.draw()

            self.cmap_combo.currentIndexChanged.connect(update_colorbar)
            self.cmap_label_edit.textChanged.connect(update_colorbar)
            update_colorbar()  # Initial draw

            contour_layout.addStretch()
            self._contour_tab = contour_tab
            self.tabs.addTab(contour_tab, "Contour")

        # Always define refline_widgets to avoid AttributeError
        self.refline_widgets = []

        def add_series_and_reflines_tabs():
            # Series Styles Tab
            series_tab = QWidget()
            series_layout = QVBoxLayout(series_tab)
            series_label = QLabel("Configure line style for each data series:")
            series_label.setStyleSheet("font-weight: bold;")
            series_layout.addWidget(series_label)

            # Scroll area for series
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.StyledPanel)
            series_container = QWidget()
            self.series_layout = QVBoxLayout(series_container)
            self.series_layout.setContentsMargins(5, 5, 5, 5)

            # Get Y column indices and names
            y_idxs = self.plot_config.get("y_col_idxs", [])
            # Ensure y_idxs is a list (handle legacy configs where it might be an integer)
            if isinstance(y_idxs, int):
                y_idxs = [y_idxs]
            series_options = self.plot_config.get("plot_options", {}).get("series", {})
            self.series_widgets = []
            for idx, y_idx in enumerate(y_idxs):
                if y_idx < len(column_names):
                    y_name = column_names[y_idx]
                    series_widget = self._create_series_widget(
                        y_name, idx, series_options.get(y_name, {})
                    )
                    self.series_layout.addWidget(series_widget)
                    self.series_widgets.append((y_name, series_widget))
            self.series_layout.addStretch()
            scroll.setWidget(series_container)
            series_layout.addWidget(scroll)
            self._series_tab = series_tab
            self.tabs.addTab(series_tab, "Series Styles")

            # Reference Lines Tab
            reflines_tab = QWidget()
            reflines_layout = QVBoxLayout(reflines_tab)
            reflines_label = QLabel("Add horizontal and vertical reference lines:")
            reflines_label.setStyleSheet("font-weight: bold;")
            reflines_layout.addWidget(reflines_label)
            # Scroll area for reference lines
            reflines_scroll = QScrollArea()
            reflines_scroll.setWidgetResizable(True)
            reflines_scroll.setFrameShape(QFrame.StyledPanel)
            reflines_container = QWidget()
            self.reflines_layout = QVBoxLayout(reflines_container)
            self.reflines_layout.setContentsMargins(5, 5, 5, 5)
            # Load existing reference lines
            self.refline_widgets = []
            existing_reflines = self.plot_config.get("plot_options", {}).get("reference_lines", [])
            for refline in existing_reflines:
                self._add_refline_widget(
                    refline.get("type", "horizontal"),
                    refline.get("value", ""),
                    refline.get("color", "#FF0000"),
                    refline.get("linestyle", "--"),
                    refline.get("linewidth", 1.0),
                    refline.get("label", "")
                )
            self.reflines_layout.addStretch()
            reflines_scroll.setWidget(reflines_container)
            reflines_layout.addWidget(reflines_scroll)
            # Buttons to add reference lines
            reflines_buttons = QHBoxLayout()
            add_hline_btn = QPushButton("+ Add Horizontal Line")
            add_hline_btn.setToolTip("Add a horizontal reference line at a specific Y value")
            add_hline_btn.clicked.connect(lambda: self._add_refline_widget("horizontal"))
            reflines_buttons.addWidget(add_hline_btn)
            add_vline_btn = QPushButton("+ Add Vertical Line")
            add_vline_btn.setToolTip("Add a vertical reference line at a specific X value")
            add_vline_btn.clicked.connect(lambda: self._add_refline_widget("vertical"))
            reflines_buttons.addWidget(add_vline_btn)
            reflines_buttons.addStretch()
            reflines_layout.addLayout(reflines_buttons)
            self._reflines_tab = reflines_tab
            self.tabs.addTab(reflines_tab, "Reference Lines")

        # Initial tab setup
        plot_type = self.plot_config.get("plot_options", {}).get("type", "line")
        if plot_type == "contourf":
            add_contour_tab()
        else:
            add_series_and_reflines_tabs()

        # Handle dynamic tab switching on plot type change
        def on_type_changed(_):
            plot_type = self.type_combo.currentData()
            # Remove all dynamic tabs
            for tab in [self._contour_tab, self._series_tab, self._reflines_tab]:
                if tab is not None:
                    idx = self.tabs.indexOf(tab)
                    if idx != -1:
                        self.tabs.removeTab(idx)
            self._contour_tab = None
            self._series_tab = None
            self._reflines_tab = None
            # Add appropriate tabs
            if plot_type == "contourf":
                add_contour_tab()
            else:
                add_series_and_reflines_tabs()

        self.type_combo.currentIndexChanged.connect(on_type_changed)

        # Tab 6: Filters & Sort
        filters_sort_tab = QWidget()
        filters_sort_layout = QVBoxLayout(filters_sort_tab)

        # Filters section
        filters_label = QLabel("<b>Data Filters:</b>")
        filters_sort_layout.addWidget(filters_label)

        filters_info = QLabel("Configure which rows from the CSV are included in this plot.")
        filters_info.setWordWrap(True)
        filters_sort_layout.addWidget(filters_info)

        # Store filters and sort in dialog - use list() to create proper copies
        self.csv_filters = list(self.plot_config.get("csv_filters", []))
        self.csv_sort = list(self.plot_config.get("csv_sort", []))

        # Filter status display
        self.filter_status_label = QLabel(self._get_filter_status_text())
        self.filter_status_label.setWordWrap(True)
        filters_sort_layout.addWidget(self.filter_status_label)

        # Filter buttons
        filter_buttons_layout = QHBoxLayout()
        edit_filters_btn = QPushButton("Configure Filters...")
        edit_filters_btn.setToolTip("Add or modify filter conditions")
        edit_filters_btn.clicked.connect(self._edit_filters)
        filter_buttons_layout.addWidget(edit_filters_btn)

        clear_filters_btn = QPushButton("Clear Filters")
        clear_filters_btn.setToolTip("Remove all filter conditions")
        clear_filters_btn.clicked.connect(self._clear_filters)
        filter_buttons_layout.addWidget(clear_filters_btn)

        filter_buttons_layout.addStretch()
        filters_sort_layout.addLayout(filter_buttons_layout)

        # Sort section
        filters_sort_layout.addSpacing(20)
        sort_label = QLabel("<b>Data Sorting:</b>")
        filters_sort_layout.addWidget(sort_label)

        sort_info = QLabel("Configure how the data is sorted before plotting.")
        sort_info.setWordWrap(True)
        filters_sort_layout.addWidget(sort_info)

        # Sort status display
        self.sort_status_label = QLabel(self._get_sort_status_text())
        self.sort_status_label.setWordWrap(True)
        filters_sort_layout.addWidget(self.sort_status_label)

        # Sort buttons
        sort_buttons_layout = QHBoxLayout()
        edit_sort_btn = QPushButton("Configure Sort...")
        edit_sort_btn.setToolTip("Configure multi-column sorting")
        edit_sort_btn.clicked.connect(self._edit_sort)
        sort_buttons_layout.addWidget(edit_sort_btn)

        clear_sort_btn = QPushButton("Clear Sort")
        clear_sort_btn.setToolTip("Remove all sorting")
        clear_sort_btn.clicked.connect(self._clear_sort)
        sort_buttons_layout.addWidget(clear_sort_btn)

        sort_buttons_layout.addStretch()
        filters_sort_layout.addLayout(sort_buttons_layout)

        filters_sort_layout.addStretch()
        self.tabs.addTab(filters_sort_tab, "Data")

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_series_widget(self, series_name, series_idx, series_options):
        """Create a widget for configuring one series.

        Args:
            series_name: Name of the data series (column name)
            series_idx: Index of the series in the list
            series_options: Dictionary of existing options for this series

        Returns:
            QFrame widget configured for the series
        """
        widget = QFrame()
        widget.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)

        # Series name label
        name_label = QLabel(f"<b>{series_name}</b>")
        layout.addWidget(name_label)

        # Plot type selection
        plot_type_layout = QHBoxLayout()
        plot_type_layout.addWidget(QLabel("Plot Type:"))
        plot_type_combo = QComboBox()
        plot_type_combo.addItem("Line", "line")
        plot_type_combo.addItem("Bar", "bar")
        current_plot_type = series_options.get("plot_type", "line")
        plot_type_idx = 0 if current_plot_type == "line" else 1
        plot_type_combo.setCurrentIndex(plot_type_idx)
        plot_type_combo.setToolTip("Choose between line plot or bar chart")
        plot_type_combo.setMinimumWidth(100)
        plot_type_layout.addWidget(plot_type_combo)
        plot_type_layout.addStretch()
        layout.addLayout(plot_type_layout)

        # Color, line style, and marker in one row
        style_layout = QHBoxLayout()

        # Color picker button
        style_layout.addWidget(QLabel("Color:"))
        color_button = QPushButton()
        color_button.setToolTip("Click to choose a custom color for this data series")
        color_button.setMaximumWidth(80)
        color_button.setMinimumHeight(25)

        # Get current color (hex string or default from cycle)
        current_color = series_options.get("color", self.COLORS[series_idx % len(self.COLORS)])
        # Parse color to QColor
        qcolor = QColor(current_color)
        if not qcolor.isValid():
            # Fallback to first default color if invalid
            qcolor = QColor(self.COLORS[0])

        # Set button style with current color
        color_button.setStyleSheet(f"background-color: {qcolor.name()}; border: 1px solid #999;")
        color_button._color = qcolor

        # Connect to color picker dialog
        def pick_color():
            color = QColorDialog.getColor(color_button._color, self, "Select Color")
            if color.isValid():
                color_button._color = color
                color_button.setStyleSheet(
                    f"background-color: {color.name()}; border: 1px solid #999;"
                )

        color_button.clicked.connect(pick_color)
        style_layout.addWidget(color_button)

        # Line style
        style_layout.addWidget(QLabel("Line Style:"))
        linestyle_combo = QComboBox()
        for i, name in enumerate(self.LINE_STYLE_NAMES):
            linestyle_combo.addItem(name, self.LINE_STYLES[i])
        current_linestyle = series_options.get("linestyle", "-")
        idx = (
            self.LINE_STYLES.index(current_linestyle)
            if current_linestyle in self.LINE_STYLES
            else 0
        )
        linestyle_combo.setCurrentIndex(idx)
        linestyle_combo.setMinimumWidth(100)
        style_layout.addWidget(linestyle_combo)

        # Marker
        style_layout.addWidget(QLabel("Marker:"))
        marker_combo = QComboBox()
        for i, name in enumerate(self.MARKER_NAMES):
            marker_combo.addItem(name, self.MARKERS[i])
        current_marker = series_options.get("marker", "")
        marker_idx = self.MARKERS.index(current_marker) if current_marker in self.MARKERS else 0
        marker_combo.setCurrentIndex(marker_idx)
        marker_combo.setMinimumWidth(100)
        style_layout.addWidget(marker_combo)

        style_layout.addStretch()
        layout.addLayout(style_layout)

        # Line width and Marker size in one row
        size_layout = QHBoxLayout()

        size_layout.addWidget(QLabel("Line Width:"))
        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.5, 5.0)
        width_spin.setSingleStep(0.5)
        width_spin.setValue(series_options.get("linewidth", 1.5))
        width_spin.setMinimumWidth(80)
        size_layout.addWidget(width_spin)

        size_layout.addWidget(QLabel("Marker Size:"))
        markersize_spin = QDoubleSpinBox()
        markersize_spin.setRange(1.0, 20.0)
        markersize_spin.setSingleStep(1.0)
        markersize_spin.setValue(series_options.get("markersize", 6.0))
        markersize_spin.setMinimumWidth(80)
        size_layout.addWidget(markersize_spin)

        size_layout.addStretch()
        layout.addLayout(size_layout)

        # Legend label in its own row
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Legend Label:"))
        label_edit = QLineEdit()
        label_edit.setText(series_options.get("label", series_name))
        label_edit.setPlaceholderText(f"Default: {series_name}")
        label_layout.addWidget(label_edit)
        layout.addLayout(label_layout)

        # Smoothing options
        smooth_layout = QVBoxLayout()
        smooth_checkbox = QCheckBox("Apply Moving Average Smoothing")
        smooth_checkbox.setChecked(series_options.get("smooth", False))
        smooth_layout.addWidget(smooth_checkbox)

        smooth_params_layout = QHBoxLayout()
        smooth_params_layout.addWidget(QLabel("Window Size:"))
        smooth_window_spin = QSpinBox()
        smooth_window_spin.setRange(2, 1000)
        smooth_window_spin.setValue(series_options.get("smooth_window", 5))
        smooth_window_spin.setToolTip("Number of points to average (must be odd for centered averaging)")
        smooth_window_spin.setMinimumWidth(80)
        smooth_params_layout.addWidget(smooth_window_spin)

        smooth_params_layout.addWidget(QLabel("Show:"))
        smooth_mode_combo = QComboBox()
        smooth_mode_combo.addItem("Smoothed Only", "smoothed")
        smooth_mode_combo.addItem("Original + Smoothed", "both")
        smooth_mode_combo.addItem("Original Only (Smoothing Off)", "original")
        current_mode = series_options.get("smooth_mode", "smoothed")
        mode_idx = 0 if current_mode == "smoothed" else (1 if current_mode == "both" else 2)
        smooth_mode_combo.setCurrentIndex(mode_idx)
        smooth_mode_combo.setMinimumWidth(150)
        smooth_params_layout.addWidget(smooth_mode_combo)

        smooth_params_layout.addStretch()
        smooth_layout.addLayout(smooth_params_layout)
        layout.addLayout(smooth_layout)

        # Trend line options
        trend_layout = QVBoxLayout()
        trend_checkbox = QCheckBox("Add Trend Line")
        trend_checkbox.setChecked(series_options.get("trendline", False))
        trend_layout.addWidget(trend_checkbox)

        trend_params_layout = QHBoxLayout()
        trend_params_layout.addWidget(QLabel("Type:"))
        trend_type_combo = QComboBox()
        trend_type_combo.addItem("Linear", "linear")
        trend_type_combo.addItem("Polynomial (degree 2)", "poly2")
        trend_type_combo.addItem("Polynomial (degree 3)", "poly3")
        trend_type_combo.addItem("Polynomial (degree 4)", "poly4")
        current_trend_type = series_options.get("trendline_type", "linear")
        trend_type_idx = 0 if current_trend_type == "linear" else (1 if current_trend_type == "poly2" else (2 if current_trend_type == "poly3" else 3))
        trend_type_combo.setCurrentIndex(trend_type_idx)
        trend_type_combo.setMinimumWidth(150)
        trend_params_layout.addWidget(trend_type_combo)

        trend_params_layout.addWidget(QLabel("Show:"))
        trend_mode_combo = QComboBox()
        trend_mode_combo.addItem("Trend Only", "trend")
        trend_mode_combo.addItem("Original + Trend", "both")
        current_trend_mode = series_options.get("trendline_mode", "both")
        trend_mode_idx = 0 if current_trend_mode == "trend" else 1
        trend_mode_combo.setCurrentIndex(trend_mode_idx)
        trend_mode_combo.setMinimumWidth(150)
        trend_params_layout.addWidget(trend_mode_combo)

        trend_params_layout.addStretch()
        trend_layout.addLayout(trend_params_layout)
        layout.addLayout(trend_layout)

        # Store references
        widget._plot_type_combo = plot_type_combo
        widget._color_button = color_button
        widget._linestyle_combo = linestyle_combo
        widget._marker_combo = marker_combo
        widget._width_spin = width_spin
        widget._markersize_spin = markersize_spin
        widget._label_edit = label_edit
        widget._smooth_checkbox = smooth_checkbox
        widget._smooth_window_spin = smooth_window_spin
        widget._smooth_mode_combo = smooth_mode_combo
        widget._trend_checkbox = trend_checkbox
        widget._trend_type_combo = trend_type_combo
        widget._trend_mode_combo = trend_mode_combo

        return widget

    def _add_refline_widget(self, line_type, value=None, color=None, linestyle=None, linewidth=None, label=None):
        """Add a reference line configuration widget.

        Args:
            line_type: "horizontal" or "vertical"
            value: Position value (y for horizontal, x for vertical)
            color: Line color (hex string or None for default)
            linestyle: Line style string (default: "solid")
            linewidth: Line width float (default: 1.5)
            label: Optional label for the line
        """
        widget = QFrame()
        widget.setFrameStyle(QFrame.Panel | QFrame.Raised)
        layout = QVBoxLayout(widget)

        # Header row: Type and Remove button
        header_layout = QHBoxLayout()
        type_label = QLabel(f"{'Horizontal' if line_type == 'horizontal' else 'Vertical'} Line")
        type_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(type_label)
        header_layout.addStretch()

        remove_btn = QPushButton("Remove")
        remove_btn.setToolTip("Remove this reference line from the plot")
        remove_btn.clicked.connect(lambda: self._remove_refline_widget(widget))
        header_layout.addWidget(remove_btn)
        layout.addLayout(header_layout)

        # Value and Color in one row
        value_color_layout = QHBoxLayout()

        value_color_layout.addWidget(QLabel("Value:"))
        value_edit = QLineEdit()
        value_edit.setPlaceholderText("0.0")
        if value is not None:
            value_edit.setText(str(value))
        # Add numeric validator
        value_edit.setValidator(QDoubleValidator())
        value_edit.setMinimumWidth(100)
        value_color_layout.addWidget(value_edit)

        value_color_layout.addWidget(QLabel("Color:"))
        color_button = QPushButton()
        color_button.setToolTip("Click to choose the color for this reference line")
        color_button.setMinimumWidth(60)
        color_button.setMaximumWidth(60)
        # Set initial color
        if color:
            initial_color = QColor(color)
        else:
            initial_color = QColor("#000000")  # Black default
        color_button._color = initial_color
        color_button.setStyleSheet(f"background-color: {initial_color.name()};")

        def choose_color():
            color = QColorDialog.getColor(color_button._color, self, "Choose Reference Line Color")
            if color.isValid():
                color_button._color = color
                color_button.setStyleSheet(f"background-color: {color.name()};")

        color_button.clicked.connect(choose_color)
        value_color_layout.addWidget(color_button)

        value_color_layout.addStretch()
        layout.addLayout(value_color_layout)

        # Style, Width, and Label in one row
        style_layout = QHBoxLayout()

        style_layout.addWidget(QLabel("Style:"))
        linestyle_combo = QComboBox()
        linestyle_combo.addItem("Solid", "solid")
        linestyle_combo.addItem("Dashed", "dashed")
        linestyle_combo.addItem("Dash-dot", "dashdot")
        linestyle_combo.addItem("Dotted", "dotted")
        # Set current style
        styles = ["solid", "dashed", "dashdot", "dotted"]
        current_style = linestyle if linestyle in styles else "solid"
        linestyle_combo.setCurrentIndex(styles.index(current_style))
        linestyle_combo.setMinimumWidth(100)
        style_layout.addWidget(linestyle_combo)

        style_layout.addWidget(QLabel("Width:"))
        width_spin = QDoubleSpinBox()
        width_spin.setRange(0.5, 5.0)
        width_spin.setSingleStep(0.5)
        width_spin.setValue(linewidth if linewidth is not None else 1.5)
        width_spin.setMinimumWidth(80)
        style_layout.addWidget(width_spin)

        style_layout.addWidget(QLabel("Label:"))
        label_edit = QLineEdit()
        label_edit.setPlaceholderText("Optional")
        if label:
            label_edit.setText(label)
        label_edit.setMinimumWidth(120)
        style_layout.addWidget(label_edit)

        style_layout.addStretch()
        layout.addLayout(style_layout)

        # Store references and metadata
        widget._line_type = line_type
        widget._value_edit = value_edit
        widget._color_button = color_button
        widget._linestyle_combo = linestyle_combo
        widget._width_spin = width_spin
        widget._label_edit = label_edit

        # Add to layout and tracking list
        self.refline_widgets.append(widget)
        # Insert before the stretch
        self.reflines_layout.insertWidget(self.reflines_layout.count() - 1, widget)

        return widget

    def _remove_refline_widget(self, widget):
        """Remove a reference line widget.

        Args:
            widget: The reference line widget to remove
        """
        if widget in self.refline_widgets:
            self.refline_widgets.remove(widget)
            self.reflines_layout.removeWidget(widget)
            widget.deleteLater()

    def _get_filter_status_text(self):
        """Get status text for filters."""
        if not self.csv_filters:
            return "No filters applied"
        return f"{len(self.csv_filters)} filter(s) applied: " + ", ".join(
            f"{col} {op} {val}" for col, op, val in self.csv_filters[:3]
        ) + ("..." if len(self.csv_filters) > 3 else "")

    def _get_sort_status_text(self):
        """Get status text for sort."""
        if not self.csv_sort:
            return "No sorting applied"
        return f"Sorted by {len(self.csv_sort)} column(s): " + ", ".join(
            f"{col} ({order})" for col, order in self.csv_sort[:3]
        ) + ("..." if len(self.csv_sort) > 3 else "")

    def _edit_filters(self):
        """Open the filter configuration dialog."""
        dialog = ColumnFilterDialog(self.column_names, self)
        dialog.set_filters(self.csv_filters)

        if dialog.exec() == QDialog.Accepted:
            self.csv_filters = dialog.get_filters()
            self.filter_status_label.setText(self._get_filter_status_text())

    def _clear_filters(self):
        """Clear all filters."""
        self.csv_filters = []
        self.filter_status_label.setText(self._get_filter_status_text())

    def _edit_sort(self):
        """Open the sort configuration dialog."""
        dialog = ColumnSortDialog(self.column_names, self)
        dialog.set_sort_specs(self.csv_sort)

        if dialog.exec() == QDialog.Accepted:
            self.csv_sort = dialog.get_sort_specs()
            self.sort_status_label.setText(self._get_sort_status_text())

    def _clear_sort(self):
        """Clear all sorting."""
        self.csv_sort = []
        self.sort_status_label.setText(self._get_sort_status_text())

    def get_plot_config(self):
        """Return updated plot configuration."""
        # Update name
        self.plot_config["name"] = self.name_edit.text()

        # Create or update plot_options
        if "plot_options" not in self.plot_config:
            self.plot_config["plot_options"] = {}

        plot_opts = self.plot_config["plot_options"]
        # Save plot type
        plot_opts["type"] = self.type_combo.currentData()
        plot_opts["title"] = self.title_edit.text()
        plot_opts["xlabel"] = self.xlabel_edit.text()
        plot_opts["ylabel"] = self.ylabel_edit.text()
        plot_opts["grid"] = self.grid_checkbox.isChecked()
        plot_opts["legend"] = self.legend_checkbox.isChecked()
        plot_opts["legend_loc"] = self.legend_loc_combo.currentData()
        plot_opts["dark_background"] = self.dark_background_checkbox.isChecked()

        # Save axis limits (convert to float or None)
        def parse_limit(text):
            text = text.strip()
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                return None

        plot_opts["xlim_min"] = parse_limit(self.xlim_min_edit.text())
        plot_opts["xlim_max"] = parse_limit(self.xlim_max_edit.text())
        plot_opts["ylim_min"] = parse_limit(self.ylim_min_edit.text())
        plot_opts["ylim_max"] = parse_limit(self.ylim_max_edit.text())

        # Save log scale options
        plot_opts["xlog"] = self.xlog_checkbox.isChecked()
        plot_opts["ylog"] = self.ylog_checkbox.isChecked()

        # Save datetime x-axis options
        plot_opts["xaxis_datetime"] = self.xaxis_datetime_checkbox.isChecked()
        plot_opts["datetime_format"] = self.datetime_format_edit.text().strip()
        plot_opts["datetime_display_format"] = self.datetime_display_format_edit.text().strip()

        # Save figure size and export options
        plot_opts["figwidth"] = self.figwidth_spin.value()
        plot_opts["figheight"] = self.figheight_spin.value()
        plot_opts["dpi"] = self.dpi_spin.value()
        plot_opts["export_format"] = self.export_format_combo.currentText()

        # Save font size options
        plot_opts["title_fontsize"] = self.title_fontsize_spin.value()
        plot_opts["axis_label_fontsize"] = self.axis_label_fontsize_spin.value()
        plot_opts["tick_fontsize"] = self.tick_fontsize_spin.value()
        plot_opts["legend_fontsize"] = self.legend_fontsize_spin.value()

        # Save font family option
        plot_opts["font_family"] = self.font_family_combo.currentText()

        # Save reference lines
        ref_lines = []
        for widget in self.refline_widgets:
            value_text = widget._value_edit.text().strip()
            if value_text:  # Only save if value is provided
                try:
                    value = float(value_text)
                    label_text = widget._label_edit.text().strip()
                    ref_lines.append({
                        "type": widget._line_type,
                        "value": value,
                        "color": widget._color_button._color.name(),
                        "linestyle": widget._linestyle_combo.currentData(),
                        "linewidth": widget._width_spin.value(),
                        "label": label_text if label_text else None
                    })
                except ValueError:
                    # Skip invalid values
                    pass
        plot_opts["reference_lines"] = ref_lines

        # Save colormap and label if contourf is selected
        if plot_opts["type"] == "contourf" and hasattr(self, "cmap_combo"):
            plot_opts["cmap"] = self.cmap_combo.currentText()
            plot_opts["cmap_label"] = self.cmap_label_edit.text()

        # Update series options
        series_opts = {}
        for series_name, widget in getattr(self, "series_widgets", []):
            series_opts[series_name] = {
                "plot_type": widget._plot_type_combo.currentData(),
                "label": widget._label_edit.text(),
                "color": widget._color_button._color.name(),  # Get hex color from QColor
                "linestyle": widget._linestyle_combo.currentData(),
                "marker": widget._marker_combo.currentData(),
                "linewidth": widget._width_spin.value(),
                "markersize": widget._markersize_spin.value(),
                "smooth": widget._smooth_checkbox.isChecked(),
                "smooth_window": widget._smooth_window_spin.value(),
                "smooth_mode": widget._smooth_mode_combo.currentData(),
                "trendline": widget._trend_checkbox.isChecked(),
                "trendline_type": widget._trend_type_combo.currentData(),
                "trendline_mode": widget._trend_mode_combo.currentData(),
            }
        plot_opts["series"] = series_opts

        # Save filters and sort
        self.plot_config["csv_filters"] = self.csv_filters
        self.plot_config["csv_sort"] = self.csv_sort

        return self.plot_config

