vibehdf5 Documentation
======================

**vibehdf5** is a powerful, lightweight GUI application for browsing, managing, and visualizing HDF5 file structures. Built with PySide6, it provides an intuitive tree-based interface for exploring groups, datasets, and attributes, with advanced features for content management and data preview.

.. image:: ../media/screenshot.png
   :alt: vibehdf5 screenshot
   :align: center

Features
--------

* **Browse & Explore**: Hierarchical tree view, dataset information, attribute display
* **Data Preview**: Text, images, binary data with syntax highlighting
* **CSV Data Support**: Import, filter, sort, and plot CSV data
* **Content Management**: Add/delete files, folders, and HDF5 items
* **Export & Extract**: Drag-and-drop export to filesystem
* **Interactive Plotting**: Embedded matplotlib plots with customization

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api/modules

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install vibehdf5

Launch
~~~~~~

.. code-block:: bash

   vibehdf5
   # or with a file
   vibehdf5 /path/to/file.h5

From Python
~~~~~~~~~~~

.. code-block:: python

   from vibehdf5 import main
   main()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
