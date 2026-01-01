#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

using UserDataLogErrorCallback = void (*)(const char *msg);

extern "C" void tflite_error_callback(void *user_data_ptr, const char *format,
                                      va_list args) {
  auto user_data = (UserDataLogErrorCallback)user_data_ptr;

  // c style va_list args is necessary as its required by the tflite c api for
  // error reporting
  // NOLINTBEGIN(cppcoreguidelines-pro-type-vararg,
  // cppcoreguidelines-pro-bounds-array-to-pointer-decay)
  va_list args_copy;
  va_copy(args_copy, args);

  const int formatted_error_msg_length =
      std::vsnprintf(nullptr, 0, format, args_copy);
  std::vector<char> formatted_error_msg_buffer;
  formatted_error_msg_buffer.resize(formatted_error_msg_length + 1);
  std::vsnprintf(formatted_error_msg_buffer.data(),
                 formatted_error_msg_buffer.size(), format, args);
  const std::string formatted_error_msg(formatted_error_msg_buffer.data());
  // NOLINTEND(cppcoreguidelines-pro-type-vararg,
  // cppcoreguidelines-pro-bounds-array-to-pointer-decay)

  user_data(formatted_error_msg.c_str());
}