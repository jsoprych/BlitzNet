# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/john/CODE/CPP/PROJECTS/NeuralNet

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/john/CODE/CPP/PROJECTS/NeuralNet/build

# Include any dependencies generated for this target.
include CMakeFiles/NeuralNetTests.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/NeuralNetTests.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/NeuralNetTests.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/NeuralNetTests.dir/flags.make

CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.o: CMakeFiles/NeuralNetTests.dir/flags.make
CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.o: /home/john/CODE/CPP/PROJECTS/NeuralNet/tests/test_main.cpp
CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.o: CMakeFiles/NeuralNetTests.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/john/CODE/CPP/PROJECTS/NeuralNet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.o -MF CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.o.d -o CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.o -c /home/john/CODE/CPP/PROJECTS/NeuralNet/tests/test_main.cpp

CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/john/CODE/CPP/PROJECTS/NeuralNet/tests/test_main.cpp > CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.i

CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/john/CODE/CPP/PROJECTS/NeuralNet/tests/test_main.cpp -o CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.s

# Object files for target NeuralNetTests
NeuralNetTests_OBJECTS = \
"CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.o"

# External object files for target NeuralNetTests
NeuralNetTests_EXTERNAL_OBJECTS =

/home/john/CODE/CPP/PROJECTS/NeuralNet/NeuralNetTests: CMakeFiles/NeuralNetTests.dir/tests/test_main.cpp.o
/home/john/CODE/CPP/PROJECTS/NeuralNet/NeuralNetTests: CMakeFiles/NeuralNetTests.dir/build.make
/home/john/CODE/CPP/PROJECTS/NeuralNet/NeuralNetTests: CMakeFiles/NeuralNetTests.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/john/CODE/CPP/PROJECTS/NeuralNet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable /home/john/CODE/CPP/PROJECTS/NeuralNet/NeuralNetTests"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/NeuralNetTests.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/NeuralNetTests.dir/build: /home/john/CODE/CPP/PROJECTS/NeuralNet/NeuralNetTests
.PHONY : CMakeFiles/NeuralNetTests.dir/build

CMakeFiles/NeuralNetTests.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/NeuralNetTests.dir/cmake_clean.cmake
.PHONY : CMakeFiles/NeuralNetTests.dir/clean

CMakeFiles/NeuralNetTests.dir/depend:
	cd /home/john/CODE/CPP/PROJECTS/NeuralNet/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/john/CODE/CPP/PROJECTS/NeuralNet /home/john/CODE/CPP/PROJECTS/NeuralNet /home/john/CODE/CPP/PROJECTS/NeuralNet/build /home/john/CODE/CPP/PROJECTS/NeuralNet/build /home/john/CODE/CPP/PROJECTS/NeuralNet/build/CMakeFiles/NeuralNetTests.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/NeuralNetTests.dir/depend

