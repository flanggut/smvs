# Builds lib and app
all:
	$(MAKE) -C lib
	$(MAKE) -C app
	cp app/smvsrecon smvsrecon

test:
	$(MAKE) -C tests

clean:
	$(RM) smvsrecon
	$(MAKE) -C app $@
	$(MAKE) -C lib $@
	$(MAKE) -C tests $@

.PHONY: all clean
