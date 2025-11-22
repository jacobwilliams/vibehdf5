Usage Guide
===========

This guide covers the main features and workflows of vibehdf5.

Launching the Application
--------------------------

Command Line
~~~~~~~~~~~~

After installation, launch from the command line:

.. code-block:: bash

   vibehdf5

Or open a specific file:

.. code-block:: bash

   vibehdf5 /path/to/your/file.h5

From Python
~~~~~~~~~~~

.. code-block:: python

   from vibehdf5 import main
   main()

Development Mode
~~~~~~~~~~~~~~~~

Run directly from source:

.. code-block:: bash

   python -m vibehdf5 [file.h5]

Working with Files
------------------

Creating New Files
~~~~~~~~~~~~~~~~~~

1. Click **New HDF5 File…** in the toolbar or press ``Ctrl+N``
2. Choose a location and filename (``.h5`` extension added automatically)
3. Confirm overwrite if file exists
4. Start adding content to the new file

Opening Files
~~~~~~~~~~~~~

1. Click **Open HDF5…** or press ``Ctrl+O``
2. Select an HDF5 file (``.h5`` or ``.hdf5``)
3. The tree will populate with the file structure
4. Recently opened files appear in **File > Open Recent** menu
5. Clear recent files list via **File > Open Recent > Clear Recent Files**

Adding Content
--------------

Add Individual Files
~~~~~~~~~~~~~~~~~~~~

1. Click **Add Files…** or press ``Ctrl+Shift+F``
2. Select one or more files
3. Files are added to the currently selected group

Add Folders
~~~~~~~~~~~

1. Click **Add Folder…** or press ``Ctrl+Shift+D``
2. Select a directory
3. The entire folder structure is recursively imported

Drag & Drop
~~~~~~~~~~~

* Drag files or folders from your file manager
* Drop onto any group in the tree
* Content is automatically added to the appropriate location

Deleting Content
----------------

1. Right-click on a dataset, group, or attribute
2. Select the delete option from the context menu
3. Confirm the deletion

.. warning::
   Deletions are permanent and modify the HDF5 file immediately.

Exporting Content
-----------------

* Drag any dataset or group from the tree to your file manager
* Datasets are extracted as individual files
* Groups are extracted as folders with full hierarchy

Viewing Data
------------

Dataset Preview
~~~~~~~~~~~~~~~

* Click any dataset to see a preview in the right panel
* PNG images are automatically rendered
* Text data displays with syntax highlighting
* Binary data shows as hex dump

Syntax Highlighting
~~~~~~~~~~~~~~~~~~~

Automatic syntax highlighting is supported for:

* Python
* JavaScript
* C/C++
* Fortran
* JSON, YAML, XML
* HTML, CSS
* Markdown
* And more...

Working with CSV Data
---------------------

Importing CSV Files
~~~~~~~~~~~~~~~~~~~

1. Use **Add Files…** or drag-and-drop to import a CSV file
2. CSV files are automatically converted to HDF5 groups with:

   * One dataset per column preserving data types
   * Column names stored as group attributes
   * Source file metadata for reference

Viewing CSV Tables
~~~~~~~~~~~~~~~~~~

1. Click on a CSV group in the tree
2. Data displays as an interactive table with column headers
3. Select multiple columns (Ctrl/Cmd+Click) for analysis

Filtering CSV Data
~~~~~~~~~~~~~~~~~~

1. Click **Configure Filters…** above the table
2. Add filter conditions:

   * Select column name
   * Choose operator (==, !=, >, >=, <, <=, contains, startswith, endswith)
   * Enter value to compare against

3. Add multiple filters (combined with AND logic)
4. Filters are automatically saved to the HDF5 file
5. Click **Clear Filters** to remove all filters

Filter Features:

* Filters persist when closing and reopening files
* Each CSV group has independent filters
* Numeric comparisons automatically convert values
* String operations for text data
* Real-time table updates when filters change

Sorting CSV Data
~~~~~~~~~~~~~~~~

1. Click **Sort…** above the table
2. Add sort columns in order of priority
3. Configure ascending/descending order
4. Use up/down arrows to reorder sort priority
5. Click **Clear Sort** to restore original order

Column Statistics
~~~~~~~~~~~~~~~~~

1. Click **Statistics…** above the table
2. View statistical summaries for each column:

   * Count, Min, Max
   * Mean, Median, Std Dev (numeric only)
   * Sum (numeric only)
   * Unique Values count

Column Visibility
~~~~~~~~~~~~~~~~~

1. Click **Columns…** above the table
2. Choose "Show All Columns" or "Show Selected Columns"
3. Check/uncheck columns to show or hide them
4. Visibility settings are saved to the HDF5 file
5. Hidden columns are excluded from CSV exports
6. Each CSV group maintains independent settings

Unique Values
~~~~~~~~~~~~~

1. Right-click on any column header in the CSV table
2. Select **Show Unique Values in '[column name]'**
3. View all unique values in a sortable dialog
4. Shows count of unique values for data inspection
5. Respects active filters (shows unique values from filtered data)

Plotting Data
~~~~~~~~~~~~~

1. Select 2 or more columns in the table (Ctrl/Cmd+Click)
2. Click **Save Plot** to create a new plot configuration
3. Enter a name for the plot
4. The plot appears in the **Saved Plots** list
5. Click any saved plot to display it

Plot Customization
~~~~~~~~~~~~~~~~~~

1. Select a saved plot and click **Edit Options**
2. **General Tab**:

   * Set title, axis labels, grid, legend
   * Enable dark background
   * Set axis limits (X/Y min/max)
   * Configure figure size and export DPI
   * Choose export format (PNG, PDF, SVG, EPS)

3. **Fonts & Styling Tab**:

   * Set font sizes for title, axes, ticks, legend
   * Enable logarithmic scale for X/Y axes
   * Add horizontal/vertical reference lines

4. **Series Styles Tab**:

   * Configure colors, line styles, markers
   * Adjust line width and marker size
   * Apply smoothing with moving average

5. Click **OK** to apply changes

Plot Management
~~~~~~~~~~~~~~~

* **Auto-Apply**: Click any plot to instantly display it
* **Rename**: Double-click a plot name to rename inline
* **Duplicate**: Right-click and select **Duplicate** to copy
* **Export Single**: Drag plot to file manager to export
* **Export All**: Right-click and select **Export All Plots**
* **Copy JSON**: Right-click and select **Copy Plot JSON**
* **Delete**: Click **Delete** or right-click to remove

Plot Features:

* Interactive matplotlib toolbar (zoom, pan, save)
* Multi-series support
* Per-series styling
* Plot configurations persist in HDF5 file

Exporting Filtered Data
~~~~~~~~~~~~~~~~~~~~~~~~

1. Drag CSV group from tree to your file manager
2. Exported CSV file contains only filtered rows
3. Original column names and order preserved

Keyboard Shortcuts
------------------

* ``Ctrl+N``: Create new HDF5 file
* ``Ctrl+O``: Open HDF5 file
* ``Ctrl+Shift+F``: Add files
* ``Ctrl+Shift+D``: Add folder
* ``Ctrl++``: Increase GUI font size
* ``Ctrl+-``: Decrease GUI font size
* ``Ctrl+0``: Reset GUI font size
* ``Ctrl+Q``: Quit

Tips & Best Practices
----------------------

Performance
~~~~~~~~~~~

* Initial load may take seconds for very large files
* CSV tables with many columns may take time to populate
* Filters are applied in-memory for fast updates

File Organization
~~~~~~~~~~~~~~~~~

* Use descriptive group names for organization
* Store metadata as attributes when appropriate
* Use file extensions in dataset names for preview features

CSV Data Management
~~~~~~~~~~~~~~~~~~~

* Filters and plots are stored as JSON in HDF5 attributes
* Each CSV group maintains independent settings
* Use filters before exporting for specific data subsets
* Create multiple plot views with different styling
