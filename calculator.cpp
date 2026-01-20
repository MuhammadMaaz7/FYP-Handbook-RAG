#include <iostream>
#include <limits>

// Reads a number from std::cin with basic validation.
double read_number(const char *prompt) {
    double value{};
    while (true) {
        std::cout << prompt;
        if (std::cin >> value) {
            return value;
        }
        std::cout << "Invalid input. Please enter a numeric value.\n";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

// Displays the menu and returns the chosen operation.
char read_operation() {
    char op{};
    while (true) {
        std::cout << "\nSelect an operation:\n"
                  << "  + : Addition\n"
                  << "  - : Subtraction\n"
                  << "  * : Multiplication\n"
                  << "  / : Division\n"
                  << "  q : Quit\n"
                  << "Choice: ";
        if (std::cin >> op) {
            if (op == '+' || op == '-' || op == '*' || op == '/' || op == 'q') {
                return op;
            }
        }
        std::cout << "Please enter a valid option (+, -, *, /, q).\n";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}

double calculate(double lhs, double rhs, char op) {
    switch (op) {
    case '+':
        return lhs + rhs;
    case '-':
        return lhs - rhs;
    case '*':
        return lhs * rhs;
    case '/':
        if (rhs == 0.0) {
            throw std::runtime_error("Division by zero is not allowed.");
        }
        return lhs / rhs;
    default:
        throw std::runtime_error("Unsupported operation.");
    }
}

int main() {
    std::cout << "Simple Calculator (C++)\n";

    while (true) {
        char op = read_operation();
        if (op == 'q') {
            std::cout << "Goodbye!\n";
            break;
        }

        double first = read_number("Enter the first number: ");
        double second = read_number("Enter the second number: ");

        try {
            double result = calculate(first, second, op);
            std::cout << "Result: " << result << "\n";
        } catch (const std::exception &ex) {
            std::cout << "Error: " << ex.what() << "\n";
        }
    }

    return 0;
}
