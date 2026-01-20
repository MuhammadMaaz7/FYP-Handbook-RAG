#include <iostream>
#include <limits>

// Reads a double from stdin, re-prompting on invalid input.
double read_number(const std::string &label) {
    double value;
    while (true) {
        std::cout << "Enter " << label << ": ";
        if (std::cin >> value) {
            return value;
        }
        std::cout << "Invalid number. Please try again.\n";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

// Performs the selected arithmetic operation.
double compute(double a, double b, int choice, bool &ok) {
    ok = true;
    switch (choice) {
        case 1:
            return a + b;
        case 2:
            return a - b;
        case 3:
            return a * b;
        case 4:
            if (b == 0.0) {
                ok = false;
                return 0.0;
            }
            return a / b;
        default:
            ok = false;
            return 0.0;
    }
}

int main() {
    std::cout << "Simple Calculator\n";
    while (true) {
        std::cout << "\nChoose an operation:\n"
                  << "1) Addition\n"
                  << "2) Subtraction\n"
                  << "3) Multiplication\n"
                  << "4) Division\n"
                  << "5) Exit\n"
                  << "Selection: ";

        int choice;
        if (!(std::cin >> choice)) {
            std::cout << "Invalid selection. Exiting.\n";
            break;
        }
        if (choice == 5) {
            std::cout << "Goodbye!\n";
            break;
        }

        double first = read_number("the first number");
        double second = read_number("the second number");

        bool ok = false;
        double result = compute(first, second, choice, ok);
        if (!ok) {
            std::cout << "Error: invalid operation";
            if (choice == 4 && second == 0.0) {
                std::cout << " (division by zero)";
            }
            std::cout << ".\n";
            continue;
        }

        std::cout << "Result: " << result << "\n";
    }
    return 0;
}
