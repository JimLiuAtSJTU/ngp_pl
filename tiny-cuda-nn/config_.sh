rm -r build/

cmake . -B build
cmake --build build --config RelWithDebInfo -j
cd bindings/torch
rm -r build/
rm -r dist/
rm -r tinycudann.egg-info/
python setup.py install