cd dependencies/DBoW2
echo "Configuring and building dependencies/DBoW2 ..."
cmake . -B build
cmake --build build --config Release -j $1

cd ../g2o
echo "Configuring and building dependencies/g2o ..."
cmake . -B build
cmake --build build --config Release -j $1

cd ../line_lbd
echo "Configuring and building dependencies/line_lbd ..."
cmake . -B build
cmake --build build --config Release -j $1

cd ../../third-party/farthest_point_sampling
echo "Configuring and building third-party/farthest_point_sampling ..."
cmake . -B build
cmake --build build --config Release -j $1

cd ../../
echo "Configuring and building RO-MAP ..."
cmake . -B build
cmake --build build --config RelWithDebInfo -j $1

echo "Done ..."
