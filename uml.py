#!/usr/bin/env python3
"""
Python Class Inheritance Analyzer

This script scans a Python repository, extracts class definitions and their
inheritance relationships, and generates a UML-like diagram using Graphviz.

Usage:
    python inheritance_analyzer.py /path/to/your/python/repo

Requirements:
    - Python 3.6+
    - graphviz package (pip install graphviz)
"""

import os
import ast
import sys
import argparse
from pathlib import Path
import graphviz

class ClassInfo:
    """Store information about a Python class."""
    def __init__(self, name, file_path, module_path, bases=None):
        self.name = name
        self.file_path = file_path
        self.module_path = module_path
        self.bases = bases or []
        self.methods = []
        self.attributes = []

class InheritanceAnalyzer:
    """Analyze Python files to extract class inheritance information."""
    
    def __init__(self):
        self.unique_classes = {}  # Store classes by their name (this is now our main storage)
        self.imports = {}  # Track imports in each file
        self.current_file = None
        self.current_module = None
        # Dictionary to store full module paths -> ClassInfo (used for resolution only)
        self.module_classes = {}
    
    def parse_import(self, node):
        """Parse import statements to track imported classes."""
        if isinstance(node, ast.Import):
            for name in node.names:
                alias = name.asname or name.name
                self.imports[alias] = name.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ''
            for name in node.names:
                alias = name.asname or name.name
                if node.level > 0:  # Relative import
                    # Handle relative imports based on current module
                    parts = self.current_module.split('.')
                    if node.level > len(parts):
                        continue
                    base_module = '.'.join(parts[:-node.level])
                    if module:
                        full_module = f"{base_module}.{module}" if base_module else module
                    else:
                        full_module = base_module
                else:
                    full_module = module
                
                self.imports[alias] = f"{full_module}.{name.name}" if full_module else name.name
    
    def resolve_base_class(self, base_expr):
        """Resolve the actual class name from a base class expression."""
        if isinstance(base_expr, ast.Name):
            # Direct class name like 'BaseClass'
            base_name = base_expr.id
            # Check if this is an imported name
            if base_name in self.imports:
                return self.imports[base_name].split('.')[-1]
            return base_name
        elif isinstance(base_expr, ast.Attribute):
            # Something like 'module.BaseClass'
            if isinstance(base_expr.value, ast.Name):
                module_alias = base_expr.value.id
                if module_alias in self.imports:
                    module_path = self.imports[module_alias]
                    return f"{module_path}.{base_expr.attr}"
                return f"{module_alias}.{base_expr.attr}"
        
        # If we can't resolve it, return a string representation
        return ast.unparse(base_expr) if hasattr(ast, 'unparse') else str(base_expr)
    
    def extract_class_info(self, node):
        """Extract information from a class definition."""
        class_name = node.name
        bases = []
        
        # Skip abstract base classes and metaclasses
        skip_bases = ['ABC', 'ABCMeta', 'metaclass', 'object']
        
        for base in node.bases:
            base_class = self.resolve_base_class(base)
            if base_class and base_class not in skip_bases:
                bases.append(base_class)
        
        class_info = ClassInfo(
            name=class_name,
            file_path=self.current_file,
            module_path=self.current_module,
            bases=bases
        )
        
        # Extract methods and attributes
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                class_info.methods.append(item.name)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info.attributes.append(target.id)
        
        # Store by full module path for resolution purposes
        full_class_name = f"{self.current_module}.{class_name}"
        self.module_classes[full_class_name] = class_info
        
        # Store in unique classes - this is our main storage now
        # If there's already a class with this name, keep the first one we found
        if class_name not in self.unique_classes:
            self.unique_classes[class_name] = class_info
    
    def visit_node(self, node):
        """Recursively visit AST nodes to extract relevant information."""
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            self.parse_import(node)
        elif isinstance(node, ast.ClassDef):
            self.extract_class_info(node)
        
        # Recursively visit child nodes
        for child in ast.iter_child_nodes(node):
            self.visit_node(child)
    
    def analyze_file(self, file_path):
        """Analyze a single Python file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                content = f.read()
                tree = ast.parse(content)
                
                self.current_file = file_path
                
                # Determine the module path based on file path
                rel_path = os.path.relpath(file_path)
                module_path = rel_path.replace('/', '.').replace('\\', '.')
                if module_path.endswith('.py'):
                    module_path = module_path[:-3]
                
                self.current_module = module_path
                self.imports = {}  # Reset imports for each file
                
                self.visit_node(tree)
                
            except Exception as e:
                print(f"Error analyzing file {file_path}: {e}")
    
    def scan_directory(self, directory):
        """Scan a directory recursively for Python files."""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self.analyze_file(file_path)
    
    def find_inheritance_relationships(self):
        """Find all inheritance relationships between classes."""
        relationships = []
        processed_pairs = set()  # Track processed (base, derived) pairs to avoid duplicates
        
        # Work with unique classes only
        for class_name, class_info in self.unique_classes.items():
            for base in class_info.bases:
                # Get the simple base name (without module path)
                simple_base = base.split('.')[-1]
                
                # Skip if this is a standard base class we want to ignore
                if simple_base in ['ABC', 'ABCMeta', 'metaclass', 'object']:
                    continue
                
                # Find the base class in our unique classes or module classes
                if simple_base in self.unique_classes:
                    # Use the simple name for the relationship
                    base_class = simple_base
                else:
                    # Check if we have it in the module classes
                    base_class = None
                    for potential_base in self.module_classes:
                        if potential_base.endswith('.' + simple_base):
                            base_class = simple_base  # Still use the simple name
                            break
                    
                    # If we still don't have it, use the original base name
                    if base_class is None:
                        base_class = simple_base
                
                # Skip if we've already processed this relationship
                relation_key = (base_class, class_name)
                if relation_key in processed_pairs:
                    continue
                
                processed_pairs.add(relation_key)
                relationships.append((base_class, class_name))
        
        return relationships
    
    def generate_graph(self, output_file='class_diagram', format='png', focus_class=None):
        """Generate a UML-like diagram using Graphviz."""
        dot = graphviz.Digraph(
            name='Class Inheritance Diagram',
            comment='UML Class Diagram',
            format=format,
            engine='dot'
        )
        
        dot.attr('graph', rankdir='BT', splines='ortho')
        dot.attr('node', shape='record', style='filled', fillcolor='lightblue',
                 fontname='Arial', margin='0.3,0.1')
        dot.attr('edge', arrowhead='empty', style='solid')
            
        # Find inheritance relationships from unique classes
        relationships = self.find_inheritance_relationships()
        
        # Filter classes if focus_class is specified
        if focus_class:
            # Find all classes that inherit from focus_class (directly or indirectly)
            relevant_classes = set()
            relevant_classes.add(focus_class)
            
            # Add all descendants
            for base, derived in relationships:
                if base == focus_class or base in relevant_classes:
                    relevant_classes.add(derived)
            
            # Filter relationships
            relationships = [(base, derived) for base, derived in relationships 
                             if base in relevant_classes or derived in relevant_classes]
        
        # Add classes to the graph
        added_classes = set()
        
        for base, derived in relationships:
            if base not in added_classes:
                if base in self.unique_classes:
                    class_info = self.unique_classes[base]
                    label = self._format_class_label(class_info)
                    dot.node(base, label=label)
                else:
                    # External class
                    dot.node(base, label=f"{{{base}||}}", shape='record', 
                             style='filled', fillcolor='lightgrey')
                added_classes.add(base)
            
            if derived not in added_classes:
                if derived in self.unique_classes:
                    class_info = self.unique_classes[derived]
                    label = self._format_class_label(class_info)
                    dot.node(derived, label=label)
                else:
                    dot.node(derived, label=f"{{{derived}||}}")
                added_classes.add(derived)
            
            dot.edge(derived, base)
        
        # Add any classes that don't have relationships but match the focus
        if focus_class and focus_class in self.unique_classes and focus_class not in added_classes:
            class_info = self.unique_classes[focus_class]
            label = self._format_class_label(class_info)
            dot.node(focus_class, label=label)
            added_classes.add(focus_class)
        
        try:
            dot.render(output_file, cleanup=True)
            print(f"Diagram generated: {output_file}.{format}")
            return f"{output_file}.{format}"
        except Exception as e:
            print(f"Error generating diagram: {e}")
            return None
    
    def _format_class_label(self, class_info):
        """Format the label for a class node in the UML diagram."""
        class_name = class_info.name
        
        # Format attributes
        if class_info.attributes:
            attributes = '\\l'.join(f"- {attr}" for attr in class_info.attributes) + '\\l'
        else:
            attributes = ''
        
        # Format methods
        if class_info.methods:
            methods = '\\l'.join(f"+ {method}()" for method in class_info.methods) + '\\l'
        else:
            methods = ''
        
        # Build the label with UML-style compartments
        label = f"{{{class_name}|{attributes}|{methods}}}"
        return label

def main():
    parser = argparse.ArgumentParser(description='Generate UML class diagram from Python code')
    parser.add_argument('path', help='Path to Python repository or file')
    parser.add_argument('-o', '--output', default='class_diagram', 
                        help='Output file name (without extension)')
    parser.add_argument('-f', '--format', default='png', choices=['png', 'svg', 'pdf'],
                        help='Output file format')
    parser.add_argument('-c', '--class-focus', help='Focus on a specific class and its hierarchy')
    
    args = parser.parse_args()
    
    analyzer = InheritanceAnalyzer()
    
    path = args.path
    if os.path.isfile(path) and path.endswith('.py'):
        analyzer.analyze_file(path)
    elif os.path.isdir(path):
        analyzer.scan_directory(path)
    else:
        print(f"Invalid path: {path}")
        return 1
    
    # Print a summary of what we found
    print(f"Found {len(analyzer.unique_classes)} unique class names")
    print(f"Found {len(analyzer.module_classes)} total class references")
    
    analyzer.generate_graph(
        output_file=args.output,
        format=args.format,
        focus_class=args.class_focus
    )
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
