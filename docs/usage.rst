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

   python -m vibehdf5.hdf5_viewer [file.h5]

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
2. **General Tab**: Set title, axis labels, grid, legend
3. **Series Styles Tab**: Configure colors, line styles, markers
4. Click **OK** to apply changes

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

Troubleshooting
---------------

Application Won't Start
~~~~~~~~~~~~~~~~~~~~~~~

* Ensure PySide6 is installed: ``pip install PySide6``
* On Apple Silicon Macs, use native ARM64 build

Drag-and-Drop Not Working
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Ensure you're dropping onto the tree view
* Verify file is opened in read-write mode
* Check file permissions

Image Preview Not Working
~~~~~~~~~~~~~~~~~~~~~~~~~~

* Check dataset name ends with ``.png``
* Verify dataset contains valid PNG binary data
* Re-import images with proper encoding
