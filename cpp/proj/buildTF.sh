#!/bin/bash

CXX=g++-6
PYTHON=python # e.g. could be python2.7 or python3 to be explicit

MY_FLAGS=
# See https://github.com/githubharald/CTCWordBeamSearch/issues/12 and
# https://www.tensorflow.org/guide/extend/op#compile_the_op_using_your_system_compiler_tensorflow_binary_installation
#MY_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0


# check if parallel decoding is enabled: specify PARALLEL NUMTHREADS, e.g. PARALLEL 8
if [ "$1" == "PARALLEL" ]; then

	# default to 4 threads if not specified
	if [ -z "$2" ]; then
		NUMTHREADS="4"
	else
		NUMTHREADS=$2
	fi

	echo "Parallel decoding with $NUMTHREADS threads"
	PARALLEL="-DWBS_PARALLEL -DWBS_THREADS=$NUMTHREADS"
else
	echo "Single-threaded decoding"
	PARALLEL=""
fi


# get and print TF version
TF_VERSION=$($PYTHON -c "import tensorflow as tf ;  print(tf.__version__)")
echo "Your TF version is $TF_VERSION"
echo "TF versions 1.3.0, 1.4.0, 1.5.0 and 1.6.0 are tested"


# compile it for TF1.3
if [ "$TF_VERSION" == "1.3.0" ]; then

	echo "Compiling for TF 1.3.0 now ..."

	TF_INC=$($PYTHON -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

	$CXX -Wall -O2 --std=c++11 -shared -o TFWordBeamSearch.so ../src/TFWordBeamSearch.cpp ../src/main.cpp ../src/WordBeamSearch.cpp ../src/PrefixTree.cpp ../src/Metrics.cpp ../src/MatrixCSV.cpp ../src/LanguageModel.cpp ../src/DataLoader.cpp ../src/Beam.cpp -fPIC $MY_FLAGS $PARALLEL -I$TF_INC 


# compile it for TF1.4
elif [ "$TF_VERSION" == "1.4.0" ]; then

	echo "Compiling for TF 1.4.0 now ..."
	
	TF_INC=$($PYTHON -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
	TF_LIB=$($PYTHON -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

	$CXX -Wall -O2 --std=c++11 -shared -o TFWordBeamSearch.so ../src/TFWordBeamSearch.cpp ../src/main.cpp ../src/WordBeamSearch.cpp ../src/PrefixTree.cpp ../src/Metrics.cpp ../src/MatrixCSV.cpp ../src/LanguageModel.cpp ../src/DataLoader.cpp ../src/Beam.cpp $MY_FLAGS $PARALLEL -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework

# all other versions (tested for: TF1.5 and TF1.6 and TF1.12)
else
	echo "Compiling for TF 1.5.0 or 1.6.0 now ..."

	TF_CFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
	TF_LFLAGS=( $($PYTHON -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


	$CXX -Wall -O2 --std=c++11 -shared -o TFWordBeamSearch.so ../src/TFWordBeamSearch.cpp ../src/main.cpp ../src/WordBeamSearch.cpp ../src/PrefixTree.cpp ../src/Metrics.cpp ../src/MatrixCSV.cpp ../src/LanguageModel.cpp ../src/DataLoader.cpp ../src/Beam.cpp -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} $MY_FLAGS $PARALLEL

fi
