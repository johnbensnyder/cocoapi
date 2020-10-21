all:
    # install pycocotools locally
	/shared/conda/bin/python setup.py build_ext --inplace
	rm -rf build

install:
	# install pycocotools to the Python site-packages
	/shared/conda/bin/python setup.py build_ext install
	rm -rf build
