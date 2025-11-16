# CSV Sort Feature Implementation

## Overview
Added comprehensive multi-column sorting capability for CSV data in the HDF5 viewer. Users can configure sort order by multiple columns with independent ascending/descending control, and sort configurations are automatically persisted to the HDF5 file.

## Features Implemented

### 1. ColumnSortDialog Class
- **Location**: `vibehdf5/hdf5_viewer.py` (lines ~462-606)
- **Purpose**: Dialog for configuring multi-column sorting
- **Features**:
  - Add multiple sort columns with priority ordering
  - Choose Ascending or Descending for each column independently
  - Reorder sort priority with up/down arrow buttons
  - Remove individual sort specifications
  - Visual feedback with instructional text

### 2. Sort UI Controls
- **Sort Button**: Opens the sort configuration dialog
- **Clear Sort Button**: Removes all sorting and displays data in original order
- **Button States**: Clear Sort button is only enabled when sorting is active
- **Location**: Filter panel above CSV table (alongside filter controls)

### 3. Sort Logic
- **Multi-Column Sorting**: Uses pandas `sort_values()` for robust multi-column sorting
- **Type Handling**: Correctly handles numeric, string, and mixed-type columns
- **Filter Integration**: Sorting is applied to filtered data (respects active filters)
- **NaN Handling**: Missing values are placed at the end of sorted results
- **Performance**: Efficient sorting even with large datasets

### 4. Persistence
- **Save to HDF5**: Sort specifications saved as JSON in `csv_sort` attribute
- **Auto-Load**: Sort configurations automatically restored when opening files
- **Format**: Stored as list of `[column_name, ascending]` tuples
- **Per-Group**: Each CSV group maintains its own independent sort configuration

### 5. Integration with Existing Features
- **Filters**: Sorting respects active filters (sorts only visible rows)
- **Plotting**: Plots use sorted and filtered data
- **Statistics**: Statistics computed on sorted and filtered data
- **Export**: CSV export includes sorted and filtered data

## User Workflow

### Configuring Sort
1. Open a CSV group to display the table
2. Click **Sort...** button above the table
3. Click **+ Add Sort Column**
4. Select column and choose Ascending/Descending
5. Add more columns as needed (priority is top-to-bottom)
6. Use arrow buttons to reorder sort priority
7. Click **OK** to apply

### Sort Priority
- First column = primary sort key
- Second column = secondary sort (breaks ties in first column)
- Third column = tertiary sort (breaks ties in first two columns)
- And so on...

### Example
Sort by "Status" (ascending), then by "Date" (descending):
1. Add "Status" as Ascending (first priority)
2. Add "Date" as Descending (second priority)
3. Result: Groups by status, within each status group, shows newest dates first

### Clearing Sort
- Click **Clear Sort** button to remove all sorting
- Data returns to original file order
- Sort attribute is removed from HDF5 file

## Technical Details

### Data Structure
```python
# Sort specifications stored as list of tuples
_csv_sort_specs = [
    ("Column1", True),   # Ascending
    ("Column2", False),  # Descending
    ("Column3", True)    # Ascending
]
```

### HDF5 Storage
```python
# Saved as JSON string in group attributes
grp.attrs["csv_sort"] = '[["Column1", true], ["Column2", false]]'
```

### Sort Algorithm
```python
# Uses pandas DataFrame.sort_values()
df.sort_values(
    by=sort_columns,           # ["Column1", "Column2"]
    ascending=sort_orders,     # [True, False]
    na_position='last'         # NaN values at end
)
```

## Benefits

1. **Multi-Level Sorting**: Sort by multiple columns with different orders
2. **Persistent State**: Sort configurations saved with the file
3. **User-Friendly**: Intuitive dialog with visual priority ordering
4. **Robust**: Handles mixed data types, missing values, and edge cases
5. **Integrated**: Works seamlessly with filters, plotting, and statistics
6. **Flexible**: Easy to add, remove, and reorder sort columns

## Files Modified

1. **vibehdf5/hdf5_viewer.py**:
   - Added `ColumnSortDialog` class
   - Added sort buttons to filter panel
   - Added `_configure_sort_dialog()` method
   - Added `_clear_sort()` method
   - Added `_save_sort_to_hdf5()` method
   - Added `_load_sort_from_hdf5()` method
   - Added `_apply_sort()` method
   - Modified `_apply_filters()` to include sorting logic
   - Modified `_show_csv_table()` to load saved sort configurations
   - Added `_csv_sort_specs` state variable

2. **README.md**:
   - Updated feature list to include multi-column sorting
   - Added detailed "Sorting CSV Data" section
   - Updated "Independent Settings" to mention sort configurations

## Testing Recommendations

1. **Basic Sort**: Sort by single column ascending and descending
2. **Multi-Column**: Sort by 2-3 columns with mixed orders
3. **Type Handling**: Sort numeric, string, and date columns
4. **Filter Integration**: Apply filters then sort, and vice versa
5. **Persistence**: Save sort, close file, reopen and verify sort is restored
6. **Edge Cases**: Test with missing values, empty strings, mixed types
7. **Priority Reordering**: Use arrow buttons to change sort priority
8. **Performance**: Test with large datasets (10,000+ rows)

## Future Enhancements

Possible future improvements:
- Natural sort for alphanumeric strings (e.g., "file1, file2, file10" vs "file1, file10, file2")
- Case-sensitive/insensitive toggle for string sorting
- Custom sort order specification (e.g., specific value ordering)
- Visual sort indicators in table headers (↑↓ arrows)
- Click column headers to sort (quick sort shortcut)
- Sort preview in dialog showing first few rows
