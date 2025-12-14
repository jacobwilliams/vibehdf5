"""Syntax highlighting support for the HDF5 viewer text preview panel.

This module provides a flexible system for adding syntax highlighting to various
file types displayed in the preview panel. New languages can be added by extending
the LANGUAGE_PATTERNS dictionary.
"""

from __future__ import annotations

import os
import re

from qtpy.QtCore import QRegularExpression
from qtpy.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat


class SyntaxHighlighter(QSyntaxHighlighter):
    """Base syntax highlighter with extensible pattern-based highlighting."""

    def __init__(self, document, language: str = "plain"):
        """Initialize the syntax highlighter.
        Args:
            document: QTextDocument to apply highlighting to.
            language: Programming language for syntax rules (default: "plain").
        """
        super().__init__(document)
        self.language = language
        self.highlighting_rules = []
        self._setup_formats()
        self._setup_rules()

    def _setup_formats(self):
        """Define text formats for different syntax elements."""
        # Keyword format
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("#5B5BE8"))  # Blue
        self.keyword_format.setFontWeight(QFont.Bold)

        # String format
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("#008000"))  # Green

        # Comment format
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("#808080"))  # Gray
        self.comment_format.setFontItalic(True)

        # Number format
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#FF6600"))  # Orange

        # Function/class format
        self.function_format = QTextCharFormat()
        self.function_format.setForeground(QColor("#8B008B"))  # Dark magenta
        self.function_format.setFontWeight(QFont.Bold)

        # Operator format
        self.operator_format = QTextCharFormat()
        self.operator_format.setForeground(QColor("#97824D"))  # Black
        self.operator_format.setFontWeight(QFont.Bold)

        # Builtin format
        self.builtin_format = QTextCharFormat()
        self.builtin_format.setForeground(QColor("#9D40E0"))  # Indigo

    def _setup_rules(self):
        """Setup highlighting rules based on the selected language."""
        patterns = LANGUAGE_PATTERNS.get(self.language, {})

        if not patterns:
            return  # No highlighting for unknown languages

        # Keywords
        if "keywords" in patterns:
            for keyword in patterns["keywords"]:
                pattern = QRegularExpression(rf"\b{keyword}\b")
                pattern.optimize()  # Optimize pattern for better performance
                self.highlighting_rules.append((pattern, self.keyword_format))

        # Builtins
        if "builtins" in patterns:
            for builtin in patterns["builtins"]:
                pattern = QRegularExpression(rf"\b{builtin}\b")
                pattern.optimize()  # Optimize pattern for better performance
                self.highlighting_rules.append((pattern, self.builtin_format))

        # Functions/Classes
        if "function_pattern" in patterns:
            pattern = QRegularExpression(patterns["function_pattern"])
            pattern.optimize()  # Optimize pattern for better performance
            self.highlighting_rules.append((pattern, self.function_format))

        # Numbers
        if "number_pattern" in patterns:
            pattern = QRegularExpression(patterns["number_pattern"])
            pattern.optimize()  # Optimize pattern for better performance
            self.highlighting_rules.append((pattern, self.number_format))

        # Operators (before strings and comments so they don't override them)
        if "operators" in patterns:
            for operator in patterns["operators"]:
                # Escape all special regex characters to match literally
                escaped = re.escape(operator)
                pattern = QRegularExpression(escaped)
                pattern.optimize()  # Optimize pattern for better performance
                self.highlighting_rules.append((pattern, self.operator_format))

        # Strings (must come before comments to avoid highlighting strings in comments)
        if "string_patterns" in patterns:
            for str_pattern in patterns["string_patterns"]:
                pattern = QRegularExpression(str_pattern)
                pattern.optimize()  # Optimize pattern for better performance
                self.highlighting_rules.append((pattern, self.string_format))

        # Comments (should be last to override other patterns)
        if "comment_patterns" in patterns:
            for comment_pattern in patterns["comment_patterns"]:
                pattern = QRegularExpression(comment_pattern)
                pattern.optimize()  # Optimize pattern for better performance
                self.highlighting_rules.append((pattern, self.comment_format))

    def highlightBlock(self, text: str):
        """Apply syntax highlighting to a block of text.

        Args:
            text: The text block to apply syntax highlighting to.
        """
        for pattern, text_format in self.highlighting_rules:
            match_iterator = pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), text_format)


class FortranNamelistHighlighter(QSyntaxHighlighter):
    """Special highlighter for Fortran namelist files that handles &NAMELIST / blocks."""

    def __init__(self, document):
        """Initialize the Fortran namelist highlighter.

        Args:
            document: QTextDocument to apply highlighting to.
        """
        super().__init__(document)
        self._setup_formats()
        self._setup_patterns()

    def _setup_formats(self):
        """Define text formats for different syntax elements."""
        # Comment format
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("#808080"))  # Gray
        self.comment_format.setFontItalic(True)

        # Namelist block name format (&NAMELIST_NAME)
        self.namelist_format = QTextCharFormat()
        self.namelist_format.setForeground(QColor("#5B5BE8"))  # Blue
        self.namelist_format.setFontWeight(QFont.Bold)

        # Variable name format
        self.variable_format = QTextCharFormat()
        self.variable_format.setForeground(QColor("#C2BB3E"))  # Dark magenta
        self.variable_format.setFontWeight(QFont.Bold)

        # Number format
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#FF6600"))  # Orange

        # String format
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("#008000"))  # Green

        # Logical (boolean) format
        self.logical_format = QTextCharFormat()
        self.logical_format.setForeground(QColor("#9D40E0"))  # Purple
        self.logical_format.setFontWeight(QFont.Bold)

        # Array index format
        self.array_index_format = QTextCharFormat()
        self.array_index_format.setForeground(QColor("#00A0A0"))  # Cyan

        # Operator format
        self.operator_format = QTextCharFormat()
        self.operator_format.setForeground(QColor("#97824D"))
        self.operator_format.setFontWeight(QFont.Bold)

    def _setup_patterns(self):
        """Setup regex patterns for Fortran namelist syntax."""
        # Namelist block markers: &NAMELIST_NAME and /
        self.namelist_start_pattern = QRegularExpression(r"&[A-Za-z_][A-Za-z0-9_]*")
        self.namelist_end_pattern = QRegularExpression(r"/\s*$")

        # Variable assignment pattern: variable_name = (includes % for derived types and () for arrays)
        self.variable_pattern = QRegularExpression(r"\b[A-Za-z_][A-Za-z0-9_%(),]*(?=\s*=)")

        # Array index pattern: numbers/expressions within parentheses (applied after variables)
        self.array_index_pattern = QRegularExpression(r"\([^)]*\)")

        # Logical values: .true., .false., .t., .f.
        self.logical_pattern = QRegularExpression(r"\.[TtFf](?:[Rr][Uu][Ee]|[Aa][Ll][Ss][Ee])?\.", QRegularExpression.CaseInsensitiveOption)

        # Number pattern: integers, floats, scientific notation (with D or E)
        self.number_pattern = QRegularExpression(r"\b[+-]?[0-9]+\.?[0-9]*([eEdD][+-]?[0-9]+)?\b")

        # String patterns: single or double quoted
        self.string_single_pattern = QRegularExpression(r"'[^']*'")
        self.string_double_pattern = QRegularExpression(r'"[^"]*"')

        # Comment patterns: ! or C/c in first column
        self.comment_pattern = QRegularExpression(r"![^\n]*")
        self.comment_column1_pattern = QRegularExpression(r"^[Cc][^\n]*")

        # Operators
        self.operator_pattern = QRegularExpression(r"[=,()]")

        # Optimize all patterns
        self.namelist_start_pattern.optimize()
        self.namelist_end_pattern.optimize()
        self.variable_pattern.optimize()
        self.array_index_pattern.optimize()
        self.logical_pattern.optimize()
        self.number_pattern.optimize()
        self.string_single_pattern.optimize()
        self.string_double_pattern.optimize()
        self.comment_pattern.optimize()
        self.comment_column1_pattern.optimize()
        self.operator_pattern.optimize()

    def highlightBlock(self, text: str):
        """Apply syntax highlighting to a block of text.

        Args:
            text: The text block to apply syntax highlighting to.
        """
        # Comments have highest priority (applied first, others won't override)
        # C or c in first column
        if self.comment_column1_pattern.match(text).hasMatch():
            self.setFormat(0, len(text), self.comment_format)
            return

        # Inline comments with !
        match_iterator = self.comment_pattern.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self.comment_format)

        # Namelist block markers (&NAMELIST_NAME)
        match_iterator = self.namelist_start_pattern.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self.namelist_format)

        # Namelist end marker (/)
        match_iterator = self.namelist_end_pattern.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self.namelist_format)

        # Strings (must come before numbers to prevent conflicts)
        match_iterator = self.string_single_pattern.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self.string_format)

        match_iterator = self.string_double_pattern.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self.string_format)

        # Logical values (.true., .false., etc.)
        match_iterator = self.logical_pattern.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            self.setFormat(match.capturedStart(), match.capturedLength(), self.logical_format)

        # Variable names (only those followed by =)
        match_iterator = self.variable_pattern.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            start = match.capturedStart()
            length = match.capturedLength()
            # Only apply if not already formatted (e.g., not inside a string)
            if self.format(start) == QTextCharFormat():
                self.setFormat(start, length, self.variable_format)

        # Array indices within variables (override variable color for indices)
        match_iterator = self.array_index_pattern.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            start = match.capturedStart()
            length = match.capturedLength()
            # Only highlight if this is within a variable (check if start-1 is variable formatted)
            if start > 0 and self.format(start - 1).foreground().color() == self.variable_format.foreground().color():
                # Highlight the contents inside parentheses, not the parentheses themselves
                # Find actual content (skip opening paren)
                if length > 2:  # Has content between ()
                    self.setFormat(start + 1, length - 2, self.array_index_format)

        # Numbers
        match_iterator = self.number_pattern.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            start = match.capturedStart()
            length = match.capturedLength()
            # Only apply if not already formatted
            if self.format(start) == QTextCharFormat():
                self.setFormat(start, length, self.number_format)

        # Operators
        match_iterator = self.operator_pattern.globalMatch(text)
        while match_iterator.hasNext():
            match = match_iterator.next()
            start = match.capturedStart()
            length = match.capturedLength()
            # Only apply if not already formatted
            if self.format(start) == QTextCharFormat():
                self.setFormat(start, length, self.operator_format)


class NAIFPCKHighlighter(QSyntaxHighlighter):
    """Special highlighter for NAIF PCK files that handles \\begindata and \\begintext blocks."""

    def __init__(self, document):
        """Initialize the NAIF PCK highlighter.

        Args:
            document: QTextDocument to apply highlighting to.
        """
        super().__init__(document)
        self._setup_formats()
        self._setup_patterns()
        self._in_data_block = False

    def _setup_formats(self):
        """Define text formats for different syntax elements."""
        # Comment format (for \begintext blocks)
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("#808080"))  # Gray
        self.comment_format.setFontItalic(True)

        # Keyword format (for \begindata, \begintext markers)
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("#5B5BE8"))  # Blue
        self.keyword_format.setFontWeight(QFont.Bold)

        # Variable name format (BODY399_RADII, etc.)
        self.variable_format = QTextCharFormat()
        self.variable_format.setForeground(QColor("#8B008B"))  # Dark magenta
        self.variable_format.setFontWeight(QFont.Bold)

        # Number format
        self.number_format = QTextCharFormat()
        self.number_format.setForeground(QColor("#FF6600"))  # Orange

        # String format
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("#008000"))  # Green

        # Operator format
        self.operator_format = QTextCharFormat()
        self.operator_format.setForeground(QColor("#97824D"))
        self.operator_format.setFontWeight(QFont.Bold)

    def _setup_patterns(self):
        """Setup regex patterns for NAIF PCK syntax."""
        # Block markers - must appear on a line by themselves (with optional whitespace)
        self.begindata_pattern = QRegularExpression(r"^\s*\\begindata\s*$")
        self.begintext_pattern = QRegularExpression(r"^\s*\\begintext\s*$")

        # Patterns for data blocks
        self.variable_pattern = QRegularExpression(r"\bBODY[0-9]+_[A-Z_0-9]+")
        # Variables starting with @ (like @2000-JAN-1/12:00 or @1972-JAN-1)
        # Match @ followed by any non-whitespace characters
        self.at_variable_pattern = QRegularExpression(r"@\S+")
        # Number pattern: match standalone numbers only (not after @ or within variable names)
        # Use word boundary and ensure it's not preceded by @ or variable characters
        self.number_pattern = QRegularExpression(r"\b[0-9]+\.?[0-9]*([eEdD][+-]?[0-9]+)?\b")
        self.string_pattern = QRegularExpression(r"'[^']*'")
        self.operator_pattern = QRegularExpression(r"[=+\-()]")

        # Optimize patterns
        self.begindata_pattern.optimize()
        self.begintext_pattern.optimize()
        self.variable_pattern.optimize()
        self.at_variable_pattern.optimize()
        self.number_pattern.optimize()
        self.string_pattern.optimize()
        self.operator_pattern.optimize()

    def highlightBlock(self, text: str):
        """Apply syntax highlighting to a block of text.

        Args:
            text: The text block to apply syntax highlighting to.
        """
        # Check for block markers
        if self.begindata_pattern.match(text).hasMatch():
            # Highlight the \begindata keyword
            match = self.begindata_pattern.match(text)
            self.setFormat(match.capturedStart(), match.capturedLength(), self.keyword_format)
            # Mark that we're now in a data block
            self.setCurrentBlockState(1)
            return

        if self.begintext_pattern.match(text).hasMatch():
            # Highlight the \begintext keyword
            match = self.begintext_pattern.match(text)
            self.setFormat(match.capturedStart(), match.capturedLength(), self.keyword_format)
            # Mark that we're now in a text (comment) block
            self.setCurrentBlockState(0)
            return

        # Determine current state
        previous_state = self.previousBlockState()
        if previous_state == -1:
            # Initial state: treat as text/comment block until we see \begindata
            current_state = 0
        else:
            current_state = previous_state

        self.setCurrentBlockState(current_state)

        # Apply highlighting based on current block state
        if current_state == 0:
            # In text/comment block - highlight entire line as comment
            self.setFormat(0, len(text), self.comment_format)
        else:
            # In data block - apply syntax highlighting
            # @-prefixed variables (like @2000-JAN-1/12:00) - MUST come first to prevent numbers from matching
            match_iterator = self.at_variable_pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), self.string_format)

            # Variables (like BODY399_RADII)
            match_iterator = self.variable_pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), self.variable_format)

            # Strings (apply before numbers to prevent conflicts)
            match_iterator = self.string_pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), self.string_format)

            # Numbers - only highlight if not already formatted
            match_iterator = self.number_pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                start = match.capturedStart()
                length = match.capturedLength()
                # Only apply number format if this position hasn't been formatted yet
                if self.format(start) == QTextCharFormat():
                    self.setFormat(start, length, self.number_format)

            # Operators
            match_iterator = self.operator_pattern.globalMatch(text)
            while match_iterator.hasNext():
                match = match_iterator.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), self.operator_format)


# Language-specific patterns
# To add a new language, add an entry to this dictionary
LANGUAGE_PATTERNS = {
    "python": {
        "keywords": [
            "False",
            "None",
            "True",
            "and",
            "as",
            "assert",
            "async",
            "await",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "try",
            "while",
            "with",
            "yield",
        ],
        "builtins": [
            "abs",
            "all",
            "any",
            "ascii",
            "bin",
            "bool",
            "bytearray",
            "bytes",
            "callable",
            "chr",
            "classmethod",
            "compile",
            "complex",
            "delattr",
            "dict",
            "dir",
            "divmod",
            "enumerate",
            "eval",
            "exec",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "globals",
            "hasattr",
            "hash",
            "help",
            "hex",
            "id",
            "input",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "locals",
            "map",
            "max",
            "memoryview",
            "min",
            "next",
            "object",
            "oct",
            "open",
            "ord",
            "pow",
            "print",
            "property",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "staticmethod",
            "str",
            "sum",
            "super",
            "tuple",
            "type",
            "vars",
            "zip",
        ],
        "function_pattern": r"\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\()",
        "number_pattern": r"\b[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?\b",
        "string_patterns": [
            r'""".*?"""',  # Triple double quotes
            r"'''.*?'''",  # Triple single quotes
            r'"[^"\\]*(\\.[^"\\]*)*"',  # Double quotes
            r"'[^'\\]*(\\.[^'\\]*)*'",  # Single quotes
        ],
        "comment_patterns": [r"#[^\n]*"],
        "operators": ["+", "-", "*", "/", "//", "%", "**", "=", "==", "!=", "<", ">", "<=", ">="],
    },
    "javascript": {
        "keywords": [
            "async",
            "await",
            "break",
            "case",
            "catch",
            "class",
            "const",
            "continue",
            "debugger",
            "default",
            "delete",
            "do",
            "else",
            "export",
            "extends",
            "finally",
            "for",
            "function",
            "if",
            "import",
            "in",
            "instanceof",
            "let",
            "new",
            "return",
            "super",
            "switch",
            "this",
            "throw",
            "try",
            "typeof",
            "var",
            "void",
            "while",
            "with",
            "yield",
        ],
        "builtins": [
            "Array",
            "Boolean",
            "Date",
            "Error",
            "Function",
            "JSON",
            "Math",
            "Number",
            "Object",
            "RegExp",
            "String",
            "console",
            "undefined",
            "null",
            "true",
            "false",
        ],
        "function_pattern": r"\b[A-Za-z_$][A-Za-z0-9_$]*(?=\s*\()",
        "number_pattern": r"\b[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?\b",
        "string_patterns": [
            r"`[^`]*`",  # Template literals
            r'"[^"\\]*(\\.[^"\\]*)*"',  # Double quotes
            r"'[^'\\]*(\\.[^'\\]*)*'",  # Single quotes
        ],
        "comment_patterns": [r"//[^\n]*", r"/\*.*?\*/"],
        "operators": [
            "+",
            "-",
            "*",
            "/",
            "%",
            "=",
            "==",
            "===",
            "!=",
            "!==",
            "<",
            ">",
            "<=",
            ">=",
            "&&",
            "||",
        ],
    },
    "json": {
        "keywords": ["true", "false", "null"],
        "number_pattern": r"-?[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?",
        "string_patterns": [r'"[^"\\]*(\\.[^"\\]*)*"'],
        "comment_patterns": [],  # JSON doesn't have comments
        "operators": [],
    },
    "xml": {
        "keywords": [],
        "function_pattern": r"</?[A-Za-z_][A-Za-z0-9_-]*",  # Tags
        "string_patterns": [r'"[^"]*"', r"'[^']*'"],
        "comment_patterns": [r"<!--.*?-->"],
        "operators": [],
    },
    "html": {
        "keywords": [],
        "function_pattern": r"</?[A-Za-z_][A-Za-z0-9_-]*",  # Tags
        "string_patterns": [r'"[^"]*"', r"'[^']*'"],
        "comment_patterns": [r"<!--.*?-->"],
        "operators": [],
    },
    "css": {
        "keywords": ["important", "inherit", "initial", "unset", "auto", "none"],
        "function_pattern": r"\b[A-Za-z-]+(?=\s*:)",  # Properties
        "number_pattern": r"[0-9]+\.?[0-9]*(px|em|rem|%|vh|vw)?",
        "string_patterns": [r'"[^"]*"', r"'[^']*'"],
        "comment_patterns": [r"/\*.*?\*/"],
        "operators": [],
    },
    "markdown": {
        "keywords": [],
        "function_pattern": r"^#{1,6}\s+.*$",  # Headers
        "string_patterns": [
            r"`[^`]+`",  # Inline code
            r"```.*?```",  # Code blocks
        ],
        "comment_patterns": [r"<!--.*?-->"],
        "operators": [],
    },
    "c": {
        "keywords": [
            "auto",
            "break",
            "case",
            "char",
            "const",
            "continue",
            "default",
            "do",
            "double",
            "else",
            "enum",
            "extern",
            "float",
            "for",
            "goto",
            "if",
            "int",
            "long",
            "register",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "struct",
            "switch",
            "typedef",
            "union",
            "unsigned",
            "void",
            "volatile",
            "while",
        ],
        "builtins": ["printf", "scanf", "malloc", "free", "sizeof"],
        "function_pattern": r"\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\()",
        "number_pattern": r"\b[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?\b",
        "string_patterns": [r'"[^"\\]*(\\.[^"\\]*)*"', r"'[^'\\]*(\\.[^'\\]*)*'"],
        "comment_patterns": [r"//[^\n]*", r"/\*.*?\*/"],
        "operators": ["+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=", "&&", "||"],
    },
    "cpp": {
        "keywords": [
            "alignas",
            "alignof",
            "and",
            "and_eq",
            "asm",
            "auto",
            "bitand",
            "bitor",
            "bool",
            "break",
            "case",
            "catch",
            "char",
            "class",
            "compl",
            "const",
            "constexpr",
            "const_cast",
            "continue",
            "decltype",
            "default",
            "delete",
            "do",
            "double",
            "dynamic_cast",
            "else",
            "enum",
            "explicit",
            "export",
            "extern",
            "false",
            "float",
            "for",
            "friend",
            "goto",
            "if",
            "inline",
            "int",
            "long",
            "mutable",
            "namespace",
            "new",
            "noexcept",
            "not",
            "not_eq",
            "nullptr",
            "operator",
            "or",
            "or_eq",
            "private",
            "protected",
            "public",
            "register",
            "reinterpret_cast",
            "return",
            "short",
            "signed",
            "sizeof",
            "static",
            "static_assert",
            "static_cast",
            "struct",
            "switch",
            "template",
            "this",
            "thread_local",
            "throw",
            "true",
            "try",
            "typedef",
            "typeid",
            "typename",
            "union",
            "unsigned",
            "using",
            "virtual",
            "void",
            "volatile",
            "wchar_t",
            "while",
            "xor",
            "xor_eq",
        ],
        "builtins": ["std", "cout", "cin", "endl", "string", "vector", "map"],
        "function_pattern": r"\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\()",
        "number_pattern": r"\b[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?\b",
        "string_patterns": [r'"[^"\\]*(\\.[^"\\]*)*"', r"'[^'\\]*(\\.[^'\\]*)*'"],
        "comment_patterns": [r"//[^\n]*", r"/\*.*?\*/"],
        "operators": ["+", "-", "*", "/", "%", "=", "==", "!=", "<", ">", "<=", ">=", "&&", "||"],
    },
    "fortran": {
        "keywords": [
            "associate",
            "program",
            "end",
            "subroutine",
            "function",
            "module",
            "use",
            "implicit",
            "none",
            "integer",
            "real",
            "double",
            "precision",
            "complex",
            "logical",
            "character",
            "parameter",
            "dimension",
            "allocatable",
            "pointer",
            "target",
            "intent",
            "in",
            "out",
            "inout",
            "if",
            "then",
            "else",
            "elseif",
            "endif",
            "do",
            "while",
            "enddo",
            "select",
            "case",
            "default",
            "stop",
            "return",
            "call",
            "contains",
        ],
        "builtins": ["write", "read", "print", "open", "close", "allocate", "deallocate"],
        "function_pattern": r"\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\()",
        "number_pattern": r"\b[0-9]+\.?[0-9]*([dDeE][+-]?[0-9]+)?(_[A-Za-z_][A-Za-z0-9_]*)?\b",
        "string_patterns": [r'"[^"]*"', r"'[^']*'"],
        "comment_patterns": [r"!.*"],
        "operators": ["+", "-", "*", "/", "**", "=", "==", "/=", "<", ">", "<=", ">="],
    },
    "yaml": {
        "keywords": ["true", "false", "null", "yes", "no"],
        "function_pattern": r"^[A-Za-z_][A-Za-z0-9_-]*(?=:)",  # Keys
        "number_pattern": r"\b[0-9]+\.?[0-9]*\b",
        "string_patterns": [r'"[^"]*"', r"'[^']*'"],
        "comment_patterns": [r"#[^\n]*"],
        "operators": [],
    },
    "toml": {
        "keywords": ["true", "false"],
        "function_pattern": r"^\[.*\]$",  # Sections
        "number_pattern": r"\b[0-9]+\.?[0-9]*\b",
        "string_patterns": [r'""".*?"""', r"'''.*?'''", r'"[^"]*"', r"'[^']*'"],
        "comment_patterns": [r"#[^\n]*"],
        "operators": [],
    },
    "ini": {
        "keywords": [],
        "function_pattern": r"^\[.*\]$",  # Sections
        "number_pattern": r"\b[0-9]+\.?[0-9]*\b",
        "string_patterns": [r'"[^"]*"', r"'[^']*'"],
        "comment_patterns": [r"[;#][^\n]*"],
        "operators": [],
    },
    "bash": {
        "keywords": [
            "if",
            "then",
            "else",
            "elif",
            "fi",
            "case",
            "esac",
            "for",
            "while",
            "until",
            "do",
            "done",
            "in",
            "function",
            "select",
            "time",
            "coproc",
            "break",
            "continue",
            "return",
            "exit",
            "source",
            "alias",
            "unalias",
            "export",
            "readonly",
            "local",
            "declare",
            "typeset",
            "set",
            "unset",
            "shift",
            "test",
        ],
        "builtins": [
            "echo",
            "printf",
            "read",
            "cd",
            "pwd",
            "pushd",
            "popd",
            "dirs",
            "let",
            "eval",
            "exec",
            "true",
            "false",
            "trap",
            "wait",
            "kill",
            "sleep",
            "type",
            "which",
            "command",
            "builtin",
            "enable",
            "help",
            "logout",
        ],
        "function_pattern": r"\b[A-Za-z_][A-Za-z0-9_]*(?=\s*\(\s*\))",
        "number_pattern": r"\b[0-9]+\b",
        "string_patterns": [
            r'"[^"\\]*(\\.[^"\\]*)*"',  # Double quotes
            r"'[^']*'",  # Single quotes
        ],
        "comment_patterns": [r"#[^\n]*"],
        "operators": [
            "&&",
            "||",
            "|",
            "&",
            ";",
            ";;",
            "(",
            ")",
            "{",
            "}",
            "[",
            "]",
            "!",
            "=",
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
        ],
    },
    "batch": {
        "keywords": [
            "if",
            "else",
            "for",
            "do",
            "in",
            "goto",
            "call",
            "exit",
            "setlocal",
            "endlocal",
            "enabledelayedexpansion",
            "disabledelayedexpansion",
            "not",
            "exist",
            "defined",
            "errorlevel",
            "equ",
            "neq",
            "lss",
            "leq",
            "gtr",
            "geq",
        ],
        "builtins": [
            "echo",
            "set",
            "cd",
            "chdir",
            "md",
            "mkdir",
            "rd",
            "rmdir",
            "del",
            "erase",
            "copy",
            "move",
            "ren",
            "rename",
            "type",
            "cls",
            "pause",
            "start",
            "title",
            "color",
            "dir",
            "path",
            "prompt",
            "pushd",
            "popd",
            "shift",
            "timeout",
        ],
        "function_pattern": r":[A-Za-z_][A-Za-z0-9_]*",  # Labels
        "number_pattern": r"\b[0-9]+\b",
        "string_patterns": [r'"[^"]*"'],
        "comment_patterns": [r"(?i)^rem\s+.*|(?i)\brem\s+.*"],
        "operators": ["==", "&&", "||", "|", "&", "(", ")", "%"],
    },
    "naif_pck": {
        "keywords": [
            r"\\begintext",
            r"\\begindata",
            r"\\beginliteral",
            r"\\endliteral",
        ],
        "builtins": [
            "BODY",
            "FRAME",
            "RADII",
            "NUT_PREC_ANGLES",
            "NUT_PREC_RA",
            "NUT_PREC_DEC",
            "NUT_PREC_PM",
            "POLE_RA",
            "POLE_DEC",
            "PM",
            "LONG_AXIS",
        ],
        "function_pattern": r"\bBODY[0-9]+_[A-Z_]+",  # NAIF variable names like BODY399_RADII
        "number_pattern": r"[+-]?[0-9]+\.?[0-9]*([eEdD][+-]?[0-9]+)?",  # Scientific notation with D or E
        "string_patterns": [r"'[^']*'"],  # Single quotes for strings in NAIF files
        "comment_patterns": [],  # Comments are handled by \begintext blocks, not line comments
        "operators": ["=", "+", "-", "(", ")", "@"],
    },
}


# Extension to language mapping
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "javascript",
    ".tsx": "javascript",
    ".json": "json",
    ".jcop": "json",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".sass": "css",
    ".md": "markdown",
    ".markdown": "markdown",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".hh": "cpp",
    ".f": "fortran",
    ".f90": "fortran",
    ".f95": "fortran",
    ".f03": "fortran",
    ".f08": "fortran",
    ".for": "fortran",
    ".ideck": "fortran_namelist",
    ".nml": "fortran_namelist",
    ".namelist": "fortran_namelist",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "ini",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "bash",
    ".ksh": "bash",
    ".bat": "batch",
    ".cmd": "batch",
    ".tf": "naif_pck",
    ".tls": "naif_pck",
    ".pc": "naif_pck",
    ".tpc": "naif_pck",
}


def get_language_from_path(path: str) -> str:
    """Determine the language from a file path based on extension.

    Args:
        path: File path or dataset name

    Returns:
        Language identifier or "plain" if unknown
    """
    if not path:
        return "plain"

    ext = os.path.splitext(path)[-1].lower()

    return EXTENSION_TO_LANGUAGE.get(ext, "plain")
