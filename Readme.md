```
mkdir build
cmake -S.. -DCMAKE_TOOLCHAIN_FILE=../cmake/aurora.cmake -DSD_AURORA=true
make VERBOSE=1
```