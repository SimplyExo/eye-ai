#include <init_helper/init_helper.hpp>
#include <iostream>

int main() {
    // just for testing
    init_helper test = init_helper("cron.service");
    auto results = test.get_logs();
    std::cout << "Exit-Code: " << results.exit_code << std::endl;
    std::cout << results.text << std::endl;

    return 0;
}