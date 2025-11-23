#!/bin/bash
# Script to compile the LaTeX project report
# Requires: pdflatex (install via: brew install --cask mactex or apt-get install texlive-full)

echo "Compiling LaTeX report..."
pdflatex -interaction=nonstopmode project_report.tex
pdflatex -interaction=nonstopmode project_report.tex  # Run twice for references

if [ -f "project_report.pdf" ]; then
    echo "✓ Report compiled successfully: project_report.pdf"
    echo "Opening PDF..."
    open project_report.pdf 2>/dev/null || xdg-open project_report.pdf 2>/dev/null || echo "Please open project_report.pdf manually"
else
    echo "✗ Compilation failed. Please check for LaTeX errors."
    echo "Install LaTeX: brew install --cask mactex (Mac) or apt-get install texlive-full (Linux)"
fi

