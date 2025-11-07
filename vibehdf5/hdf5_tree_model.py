from __future__ import annotations

import h5py
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItem, QStandardItemModel
from PySide6.QtWidgets import QStyle, QApplication


class HDF5TreeModel(QStandardItemModel):
    """A simple tree model to display HDF5 file structure.

    Columns:
    - Name: group/dataset/attribute name
    - Info: type, shape, dtype (for datasets), attribute value preview
    """

    COL_NAME = 0
    COL_INFO = 1

    ROLE_PATH = Qt.UserRole + 1
    ROLE_KIND = Qt.UserRole + 2  # 'file', 'group', 'dataset', 'attr', 'attrs-folder'
    ROLE_ATTR_KEY = Qt.UserRole + 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHorizontalHeaderLabels(["Name", "Info"])
        self._style = QApplication.instance().style() if QApplication.instance() else None
        self._filepath: str | None = None

    def load_file(self, filepath: str) -> None:
        """Load the HDF5 file and populate the model."""
        self.clear()
        self.setHorizontalHeaderLabels(["Name", "Info"])  # reset headers after clear()
        self._filepath = filepath

        root_name = filepath.split("/")[-1]
        root_item = QStandardItem(root_name)
        info_item = QStandardItem("HDF5 file")

        if self._style:
            root_item.setIcon(self._style.standardIcon(QStyle.SP_DriveHDIcon))

        self.invisibleRootItem().appendRow([root_item, info_item])
        root_item.setData("/", self.ROLE_PATH)
        root_item.setData("file", self.ROLE_KIND)

        with h5py.File(filepath, "r") as h5:
            self._add_group(h5, root_item)

    @property
    def filepath(self) -> str | None:
        return self._filepath

    # Internal helpers
    def _add_group(self, group: h5py.Group, parent_item: QStandardItem) -> None:
        """Recursively add a group and its children to the model."""
        # Attributes (if any)
        if len(group.attrs):
            attrs_item = QStandardItem("Attributes")
            attrs_info = QStandardItem(f"{len(group.attrs)} item(s)")
            if self._style:
                attrs_item.setIcon(self._style.standardIcon(QStyle.SP_DirIcon))
            parent_item.appendRow([attrs_item, attrs_info])
            attrs_item.setData(group.name, self.ROLE_PATH)
            attrs_item.setData("attrs-folder", self.ROLE_KIND)
            for key, val in group.attrs.items():
                name_item = QStandardItem(str(key))
                value_preview = _value_preview(val)
                info_item = QStandardItem(f"attr = {value_preview}")
                if self._style:
                    name_item.setIcon(self._style.standardIcon(QStyle.SP_MessageBoxInformation))
                attrs_item.appendRow([name_item, info_item])
                name_item.setData(group.name, self.ROLE_PATH)
                name_item.setData("attr", self.ROLE_KIND)
                name_item.setData(str(key), self.ROLE_ATTR_KEY)

        # Child groups and datasets
        for name, obj in group.items():
            if isinstance(obj, h5py.Group):
                g_item = QStandardItem(name)
                g_info = QStandardItem("group")
                if self._style:
                    g_item.setIcon(self._style.standardIcon(QStyle.SP_DirIcon))
                parent_item.appendRow([g_item, g_info])
                g_item.setData(obj.name, self.ROLE_PATH)
                g_item.setData("group", self.ROLE_KIND)
                self._add_group(obj, g_item)
            elif isinstance(obj, h5py.Dataset):
                d_item = QStandardItem(name)
                shape = obj.shape
                dtype = obj.dtype
                space = f"{shape}" if shape is not None else "(scalar)"
                d_info = QStandardItem(f"dataset | shape={space} | dtype={dtype}")
                if self._style:
                    d_item.setIcon(self._style.standardIcon(QStyle.SP_FileIcon))
                parent_item.appendRow([d_item, d_info])
                d_item.setData(obj.name, self.ROLE_PATH)
                d_item.setData("dataset", self.ROLE_KIND)
            else:  # pragma: no cover - unknown kinds
                unk_item = QStandardItem(name)
                unk_info = QStandardItem(type(obj).__name__)
                parent_item.appendRow([unk_item, unk_info])


def _value_preview(val, max_len: int = 80) -> str:
    """Create a compact one-line preview for attribute values."""
    try:
        text = repr(val)
    except Exception:
        text = str(val)
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text
