"""
SQLAlchemy models for SQLite storage backend.

This module defines the database schema for storing hierarchical data
structures (similar to HDF5) in SQLite.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Node(Base):
    """Represents a hierarchical node (like HDF5 group or dataset)."""

    __tablename__ = "nodes"

    id = Column(Integer, primary_key=True)
    path = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    node_type = Column(String, nullable=False)  # 'group', 'dataset', 'csv-group', 'file'
    parent_path = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    modified_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    dataset = relationship("Dataset", back_populates="node", uselist=False, cascade="all, delete-orphan")
    attributes = relationship("Attribute", back_populates="node", cascade="all, delete-orphan")
    csv_columns = relationship("CSVColumn", back_populates="node", cascade="all, delete-orphan")
    csv_metadata = relationship("CSVMetadata", back_populates="node", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Node(path={self.path!r}, type={self.node_type!r})>"


class Dataset(Base):
    """Stores dataset data and metadata."""

    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey("nodes.id", ondelete="CASCADE"), unique=True, nullable=False)
    dtype = Column(String)  # numpy dtype string
    shape = Column(Text)  # JSON array, e.g., "[100, 3]"
    data = Column(LargeBinary)  # Actual data (numpy array as bytes or string)
    compression = Column(String)  # 'gzip', 'none', etc.
    compression_opts = Column(Text)  # JSON for compression parameters
    chunks = Column(Text)  # JSON for chunk shape

    # Relationship
    node = relationship("Node", back_populates="dataset")

    def __repr__(self):
        return f"<Dataset(node_id={self.node_id}, dtype={self.dtype!r}, shape={self.shape!r})>"


class Attribute(Base):
    """Stores attributes (metadata) for nodes."""

    __tablename__ = "attributes"
    __table_args__ = (UniqueConstraint("node_id", "key", name="uix_node_key"),)

    id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False, index=True)
    key = Column(String, nullable=False)
    value_type = Column(String)  # 'str', 'int', 'float', 'array', etc.
    value = Column(Text)  # JSON-encoded value

    # Relationship
    node = relationship("Node", back_populates="attributes")

    def __repr__(self):
        return f"<Attribute(node_id={self.node_id}, key={self.key!r})>"


class CSVColumn(Base):
    """Stores individual columns for CSV groups."""

    __tablename__ = "csv_columns"
    __table_args__ = (UniqueConstraint("node_id", "column_name", name="uix_node_column"),)

    id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey("nodes.id", ondelete="CASCADE"), nullable=False, index=True)
    column_name = Column(String, nullable=False)
    column_index = Column(Integer)  # Original order in CSV
    dtype = Column(String)  # numpy dtype string
    data = Column(LargeBinary)  # Column data as numpy array bytes
    visible = Column(Boolean, default=True)

    # Relationship
    node = relationship("Node", back_populates="csv_columns")

    def __repr__(self):
        return f"<CSVColumn(node_id={self.node_id}, name={self.column_name!r}, index={self.column_index})>"


class CSVMetadata(Base):
    """Stores CSV-specific metadata (filters, sorts, etc.)."""

    __tablename__ = "csv_metadata"

    id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey("nodes.id", ondelete="CASCADE"), unique=True, nullable=False)
    filter_spec = Column(Text)  # JSON with filter configuration
    sort_spec = Column(Text)  # JSON with sort configuration
    filtered_indices = Column(Text)  # JSON array of filtered row indices

    # Relationship
    node = relationship("Node", back_populates="csv_metadata")

    def __repr__(self):
        return f"<CSVMetadata(node_id={self.node_id})>"


class Plot(Base):
    """Stores plot configurations and thumbnails."""

    __tablename__ = "plots"

    id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey("nodes.id", ondelete="CASCADE"), unique=True, nullable=False)
    plot_config = Column(Text)  # JSON with plot specification
    thumbnail = Column(LargeBinary)  # PNG thumbnail

    def __repr__(self):
        return f"<Plot(node_id={self.node_id})>"
