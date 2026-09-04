#include <systemd/systemd.hpp>
#include <iostream>

int main() {
    // just for testing
    systemd test = systemd("cron.service");
    auto results = test.get_logs();
    std::cout << "Exit-Code: " << results.exit_code << std::endl;
    std::cout << results.text << std::endl;

    return 0;
}