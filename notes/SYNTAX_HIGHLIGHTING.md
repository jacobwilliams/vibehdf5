# Syntax Highlighting Feature

## Overview

The HDF5 viewer now includes syntax highlighting for text file previews. When viewing datasets that represent text files (identified by their extension in the dataset name), the preview panel will automatically apply syntax highlighting for better readability.

## Supported Languages

The syntax highlighter currently supports the following languages:

### Programming Languages
- **Python** (`.py`, `.pyw`)
- **JavaScript/TypeScript** (`.js`, `.jsx`, `.ts`, `.tsx`)
- **C** (`.c`, `.h`)
- **C++** (`.cpp`, `.cxx`, `.cc`, `.hpp`, `.hxx`, `.hh`)
- **Fortran** (`.f`, `.f90`, `.f95`, `.f03`, `.f08`, `.for`)

### Markup & Data Languages
- **HTML** (`.html`, `.htm`)
- **XML** (`.xml`)
- **CSS** (`.css`, `.scss`, `.sass`)
- **Markdown** (`.md`, `.markdown`)
- **JSON** (`.json`)
- **YAML** (`.yaml`, `.yml`)
- **TOML** (`.toml`)
- **INI/Config** (`.ini`, `.cfg`, `.conf`)

## How It Works

1. When you select a dataset in the tree, the viewer checks if the dataset name has a recognized file extension
2. If a match is found, the appropriate syntax highlighter is applied
3. The preview panel displays the text with color-coded syntax elements:
   - **Keywords**: Blue, bold
   - **Strings**: Green
   - **Comments**: Gray, italic
   - **Numbers**: Orange
   - **Functions/Classes**: Dark magenta, bold
   - **Operators**: Black, bold
   - **Built-ins**: Indigo

## Adding New Languages

The syntax highlighting system is designed to be easily extensible. To add support for a new language:

### 1. Define Language Patterns

Edit `vibehdf5/syntax_highlighter.py` and add a new entry to the `LANGUAGE_PATTERNS` dictionary:

```python
LANGUAGE_PATTERNS = {
    # ... existing languages ...
    "your_language": {
        "keywords": ["keyword1", "keyword2", ...],
        "builtins": ["builtin1", "builtin2", ...],
        "function_pattern": r"regex_for_functions",
        "number_pattern": r"regex_for_numbers",
        "string_patterns": [r"regex1", r"regex2", ...],
        "comment_patterns": [r"regex1", r"regex2", ...],
        "operators": ["+", "-", "*", ...]
    },
}
```

### 2. Map File Extensions

Add the file extensions to the `EXTENSION_TO_LANGUAGE` dictionary:

```python
EXTENSION_TO_LANGUAGE = {
    # ... existing mappings ...
    ".ext": "your_language",
    ".ext2": "your_language",
}
```

### Example: Adding Ruby Support

```python
# In LANGUAGE_PATTERNS:
"ruby": {
    "keywords": [
        "begin", "end", "class", "module", "def", "if", "else", "elsif",
        "unless", "case", "when", "while", "until", "for", "break", "next",
        "redo", "retry", "return", "yield", "super", "self", "true", "false",
        "nil", "and", "or", "not", "alias", "defined?", "BEGIN", "END"
    ],
    "builtins": [
        "puts", "print", "p", "gets", "require", "include", "attr_reader",
        "attr_writer", "attr_accessor", "raise", "fail"
    ],
    "function_pattern": r"\bdef\s+([A-Za-z_][A-Za-z0-9_!?]*)",
    "number_pattern": r"\b[0-9]+\.?[0-9]*\b",
    "string_patterns": [
        r'"[^"\\]*(\\.[^"\\]*)*"',
        r"'[^'\\]*(\\.[^'\\]*)*'",
        r"%[qQwWiIrsx]?\{[^}]*\}",
        r"%[qQwWiIrsx]?\([^)]*\)",
        r"%[qQwWiIrsx]?\[[^\]]*\]",
    ],
    "comment_patterns": [r"#[^\n]*"],
    "operators": ["+", "-", "*", "/", "%", "**", "=", "==", "!=", "<", ">", "<=", ">=", "&&", "||"]
},

# In EXTENSION_TO_LANGUAGE:
".rb": "ruby",
".rake": "ruby",
```

## Pattern Syntax

### Keywords and Builtins
Simple list of strings. The highlighter will automatically add word boundaries (`\b`) around them.

### Regex Patterns
All patterns use Qt's `QRegularExpression` syntax (similar to PCRE):
- `\b` - Word boundary
- `.` - Any character
- `*` - Zero or more
- `+` - One or more
- `?` - Zero or one
- `[...]` - Character class
- `(...)` - Capturing group
- `(?=...)` - Lookahead assertion

### String Patterns
Multiple patterns can be provided to handle different string delimiters (single quotes, double quotes, triple quotes, etc.)

### Comment Patterns
Multiple patterns for single-line and multi-line comments

## Technical Details

### Architecture
- **SyntaxHighlighter**: Base class extending `QSyntaxHighlighter`
- **Language Detection**: Automatic based on file extension in dataset name
- **Pattern Matching**: Regex-based with efficient compiled patterns
- **Performance**: Lazy initialization - highlighter only created when needed

### Color Scheme
The default color scheme is designed for readability on light backgrounds:
- Keywords: `#0000FF` (Blue)
- Strings: `#008000` (Green)
- Comments: `#808080` (Gray)
- Numbers: `#FF6600` (Orange)
- Functions: `#8B008B` (Dark Magenta)
- Operators: `#000000` (Black)
- Builtins: `#4B0082` (Indigo)

### Customization
To customize colors, edit the `_setup_formats()` method in the `SyntaxHighlighter` class.

## Usage Examples

### Viewing Python Code
If your HDF5 file contains a dataset named `script.py`:
```python
with h5py.File('myfile.h5', 'w') as f:
    f.create_dataset('script.py', data=python_code_string)
```

When you select it in the viewer, it will automatically be highlighted as Python code.

### Viewing JSON Data
```python
with h5py.File('myfile.h5', 'w') as f:
    f.create_dataset('config.json', data=json_string)
```

Selecting `config.json` will display it with JSON syntax highlighting.

## Performance Considerations

- Highlighting is applied incrementally as text is displayed
- Large files (>1MB) are automatically truncated in the preview
- Highlighting is disabled for very large datasets to maintain responsiveness
- Old highlighters are properly cleaned up when switching between datasets

## Limitations

- Only text datasets are highlighted (binary data is shown as hex)
- Extension must be in the dataset name (e.g., `mycode.py` not just `mycode`)
- Multi-line regex patterns may not work perfectly in all cases
- Color scheme is currently fixed (no theme support yet)

## Future Enhancements

Potential improvements for future versions:
- Theme support (light/dark/custom)
- Additional language support (Rust, Go, Swift, etc.)
- Configurable color schemes
- Line numbering in preview
- Code folding for large files
- Semantic highlighting (requires language servers)
