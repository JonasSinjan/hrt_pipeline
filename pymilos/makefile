LIB_DIR = lib

default: pymilos

pymilos: setup.py #pymilos.pyx $(LIB_DIR)/libmilos.a
	make -C $(LIB_DIR) libmilos.a
	python3 setup.py build_ext --inplace && rm -f pymilos.c && rm -Rf build

#$(LIB_DIR)/libmilos.a:
#	make -C $(LIB_DIR) libmilos.a

clean:
	rm *.so
