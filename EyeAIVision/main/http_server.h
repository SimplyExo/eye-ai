#pragma once
#include <esp_camera.h>
#include <esp_http_server.h>
#include "esp_err.h"
#include "esp_log.h"

#define PART_BOUNDARY "123456789000000000000987654321"

esp_err_t stream_handler(httpd_req_t *req);
esp_err_t single_frame_handler(httpd_req_t *req);
void startCameraServer();
