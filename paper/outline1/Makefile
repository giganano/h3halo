
TEXSOURCES 		:= $(wildcard *.tex)
TEXCOMPILER 	:= pdflatex
BIBCOMPILER 	:= bibtex

all: ms.pdf

ms.pdf: ms.tex ms.bib $(TEXSOURCES)
	@ $(TEXCOMPILER) $(basename $<)
	@ $(BIBCOMPILER) $(basename $<)
	@ $(TEXCOMPILER) $(basename $<)
	@ $(TEXCOMPILER) $(basename $<)

.PHONY: clean
clean:
	@ rm -f ms.pdf
	@ rm -f *.aux
	@ rm -f *.bbl
	@ rm -f *.blg
	@ rm -f *.log
	@ rm -f *.out

