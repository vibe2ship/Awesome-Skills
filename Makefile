INSTALL_DIR := $(HOME)/.claude/commands
SKILLS_DIR  := skills
SKILLS      := $(wildcard $(SKILLS_DIR)/*.md)

.PHONY: install uninstall list

install: ## Install all skills to ~/.claude/commands/
	@mkdir -p $(INSTALL_DIR)
	@count=0; \
	for f in $(SKILLS); do \
		cp "$$f" $(INSTALL_DIR)/; \
		count=$$((count + 1)); \
		echo "  installed: $$(basename $$f)"; \
	done; \
	echo ""; \
	echo "Done. $$count skill(s) installed to $(INSTALL_DIR)"

uninstall: ## Remove installed skills from ~/.claude/commands/
	@for f in $(SKILLS); do \
		rm -f $(INSTALL_DIR)/$$(basename "$$f"); \
		echo "  removed: $$(basename $$f)"; \
	done; \
	echo "Done."

list: ## List all available skills
	@echo "Available skills:"; \
	for f in $(SKILLS); do \
		name=$$(basename "$$f" .md); \
		echo "  /$$name"; \
	done
