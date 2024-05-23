#include "Node.h"
#include <fstream>

int main() {
    Node<double, ActivationType::Sigmoid> node(3);

    // Save node state to a file
    std::ofstream outFile("node_state.txt");
    outFile << node;
    outFile.close();

    // Load node state from a file
    Node<double, ActivationType::Sigmoid> loadedNode;
    std::ifstream inFile("node_state.txt");
    inFile >> loadedNode;
    inFile.close();

    // Print loaded node state for debugging
    std::cout << loadedNode << std::endl;

    return 0;
}
