# Install all necessary packages
sudo apt-get update
sudo apt-get install libmuparserx-dev libtbb-dev dpkg

# Compile eigen
cd build_dir
cmake source_dir
make install
