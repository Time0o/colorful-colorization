TEX_DIR=tex
RESOURCE_DIR=resources
OUT_DIR=out

define run_pdflatex
	pdflatex -halt-on-error -output-directory $(OUT_DIR) $(1)
endef

RESOURCES:=$(wildcard $(RESOURCE_DIR)/*)

all: $(OUT_DIR)/report.pdf $(OUT_DIR)/slides.pdf

$(OUT_DIR)/report.pdf: $(TEX_DIR)/report.tex report.bib $(RESOURCES)
	$(call run_pdflatex, $<)
	bibtex $(patsubst $(TEX_DIR)/%.tex, $(OUT_DIR)/%.aux, $<)
	$(call run_pdflatex, $<)
	$(call run_pdflatex, $<)

$(OUT_DIR)/slides.pdf: $(TEX_DIR)/slides.tex slides.bib $(RESOURCES)
	$(call run_pdflatex, $<)
	bibtex $(patsubst $(TEX_DIR)/%.tex, $(OUT_DIR)/%.aux, $<)
	$(call run_pdflatex, $<)
	$(call run_pdflatex, $<)

.PHONY: clean

clean:
	@rm -f $(OUT_DIR)/*
