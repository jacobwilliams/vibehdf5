# GUI Tooltips Documentation

## Overview
Added comprehensive tooltips to all GUI buttons and toolbar actions to provide helpful context and usage information to users.

## Tooltips Added

### Toolbar Actions
1. **New HDF5 File** - "Create a new empty HDF5 file (Ctrl+N)"
2. **Open HDF5** - "Open an existing HDF5 file for browsing and editing (Ctrl+O)"
3. **Add Files** - "Import one or more files into the HDF5 archive (Ctrl+Shift+F)"
4. **Add Folder** - "Import an entire folder structure recursively into the HDF5 archive (Ctrl+Shift+D)"
5. **New Folder** - "Create a new empty group (folder) in the selected location (Ctrl+Shift+N)"
6. **Expand All** - "Expand all groups in the tree view to show full hierarchy"
7. **Collapse All** - "Collapse all groups in the tree view to show only top level"
8. **Quit** - "Close the application (Ctrl+Q)"
9. **Plot Selected Columns** - "Plot selected table columns (first selection is X, others are Y)"

### Plot Management Buttons
1. **Save Plot** - "Save current table column selection as a named plot configuration"
2. **Edit Options** - "Customize appearance, styling, and export settings for the selected plot"
3. **Delete** - "Delete the selected plot configuration permanently"

### Filter Panel Buttons
1. **Configure Filters...** - "Add or modify filter conditions to show only specific rows (filters are saved with the file)"
2. **Clear Filters** - "Remove all active filters and show all rows"
3. **Statistics...** - "View statistical summaries (min, max, mean, median, etc.) for each column using filtered data"
4. **Sort...** - "Configure multi-column sorting with ascending/descending order (sort settings are saved with the file)"
5. **Clear Sort** - "Remove all sorting and display rows in original order"

### Sort Dialog Buttons
1. **+ Add Sort Column** - "Add a new column to sort by (columns are sorted in order from top to bottom)"
2. **↑ (Move Up)** - "Move this sort column up (higher priority)"
3. **↓ (Move Down)** - "Move this sort column down (lower priority)"
4. **Remove** - "Remove this sort column"

### Filter Dialog Buttons
1. **+ Add Filter** - "Add a new filter condition (all filters are combined with AND logic)"
2. **Remove** - "Remove this filter condition"

### Plot Options Dialog - Reference Lines Tab
1. **+ Add Horizontal Line** - "Add a horizontal reference line at a specific Y value"
2. **+ Add Vertical Line** - "Add a vertical reference line at a specific X value"
3. **Remove** - "Remove this reference line from the plot"
4. **Color Button** - "Click to choose the color for this reference line"

### Plot Options Dialog - Series Styles Tab
1. **Color Button** - "Click to choose a custom color for this data series"

## Benefits

1. **Improved Discoverability** - Users can hover over buttons to understand their function
2. **Keyboard Shortcuts** - Tooltips display keyboard shortcuts where applicable
3. **Feature Awareness** - Tooltips inform users about persistence (e.g., "filters are saved with the file")
4. **Context-Specific Help** - Each tooltip explains what happens when the button is clicked
5. **Better User Experience** - Reduces need to consult documentation for basic operations

## Implementation Details

- All tooltips added using `setToolTip()` method
- Tooltips are concise but descriptive
- Include keyboard shortcuts in parentheses where applicable
- Mention persistence behavior (saving to HDF5) where relevant
- Use action-oriented language ("Add", "Remove", "Configure", etc.)

## Testing

Hover over any button or toolbar action to see the tooltip appear after a brief delay (typically 700ms on most systems).
