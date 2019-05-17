TEX_DIR=tex
RESOURCE_DIR=resources
OUT_DIR=out

TARGET=report

define run_pdflatex
	pdflatex -halt-on-error -output-directory $(OUT_DIR) $(1)
endef

_TEX_ALL:=$(wildcard $(TEX_DIR)/*)
TEX_NON_TARGET:=$(filter-out $(TEX_DIR)/$(TARGET).pdf, $(_TEX_ALL))

RESOURCES:=$(wildcard $(RESOURCE_DIR)/*)

$(OUT_DIR)/$(TARGET).pdf: $(TEX_DIR)/$(TARGET).tex $(TEX_NON_TARGET) $(RESOURCES) $(TARGET).bib
	$(call run_pdflatex, $<)
	bibtex $(patsubst $(TEX_DIR)/%.tex, $(OUT_DIR)/%.aux, $<)
	$(call run_pdflatex, $<)
	$(call run_pdflatex, $<)

.PHONY: clean

clean:
	@rm -f $(OUT_DIR)/*
