Installation
============

Using pip
---------

The easiest way to install vibehdf5 is using pip:

.. code-block:: bash

   pip install vibehdf5

From Source
-----------

To install from source:

.. code-block:: bash

   git clone https://github.com/jacobwilliams/vibehdf5.git
   cd vibehdf5
   pip install -e .

Using pixi (for development)
-----------------------------

For development work, you can use pixi:

.. code-block:: bash

   cd vibehdf5/env
   pixi shell

Dependencies
------------

vibehdf5 requires the following packages:

* Python ≥ 3.8
* PySide6 or PyQt6 (via qtpy abstraction)
* h5py - HDF5 interface
* numpy - Array operations
* pandas - CSV import and data filtering
* matplotlib - Plotting (optional, for CSV plotting features)
* qtpy - Qt abstraction layer for PySide6/PyQt6 compatibility

All dependencies are automatically installed when using pip.
