#include "http_server.h"

static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static const char * TAG_HTTP = "HTTP";

bool taking_frame = false;

httpd_handle_t stream_httpd = NULL;

void set_framesize(framesize_t framesize) {
  sensor_t *sensor = esp_camera_sensor_get();

  sensor->set_exposure_ctrl(sensor, false);
  sensor->set_gain_ctrl(sensor, false);
  sensor->set_aec2(sensor, false);

  sensor->set_framesize(sensor, framesize);
  vTaskDelay(pdMS_TO_TICKS(200));
}


esp_err_t stream_handler(httpd_req_t *req){
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char * part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if(res != ESP_OK){
    return res;
  }

  while(true){
    fb = esp_camera_fb_get();
    if (!fb) {
      ESP_LOGE(TAG_HTTP, "Camera capture failed");
      res = ESP_FAIL;
    } else {
      if(fb->width > 400){
        if(fb->format != PIXFORMAT_JPEG){
          bool jpeg_converted = frame2jpg(fb, 80, &_jpg_buf, &_jpg_buf_len);
          esp_camera_fb_return(fb);
          fb = NULL;
          if(!jpeg_converted){
            ESP_LOGE(TAG_HTTP, "JPEG compression failed");
            res = ESP_FAIL;
          }
        } else {
          _jpg_buf_len = fb->len;
          _jpg_buf = fb->buf;
        }
      }
    }
    if(res == ESP_OK){
      size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if(res == ESP_OK){
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    if(fb){
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    } else if(_jpg_buf){
      free(_jpg_buf);
      _jpg_buf = NULL;
    }
    if(res != ESP_OK){
      break;
    }
  }
  return res;
}


esp_err_t single_frame_handler(httpd_req_t *req) {
  taking_frame = true;
  set_framesize(FRAMESIZE_UXGA);

  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
      printf("Camera capture failed\n");
      httpd_resp_send_500(req);
      return ESP_FAIL;
  }

  // JPEG Konvertierung
  uint8_t *jpg_buf = fb->buf;
  size_t jpg_len = fb->len;
  if (fb->format != PIXFORMAT_JPEG) {
      if (!frame2jpg(fb, 80, &jpg_buf, &jpg_len)) {
          printf("JPEG compression failed\n");
          esp_camera_fb_return(fb);
          httpd_resp_send_500(req);
          return ESP_FAIL;
      }
  }

  httpd_resp_set_type(req, "image/jpeg");
  esp_err_t res = httpd_resp_send(req, (const char *)jpg_buf, jpg_len);

  if (fb->format != PIXFORMAT_JPEG) {
      free(jpg_buf);
  }
  esp_camera_fb_return(fb);

  set_framesize(FRAMESIZE_VGA);
  taking_frame = false;

  return res;
}

void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_uri_handlers = 2;

  httpd_uri_t index_uri = {
    .uri       = "/cam0",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL,
  };

  httpd_uri_t frame_uri = {
    .uri       = "/frame",
    .method    = HTTP_GET,
    .handler   = single_frame_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &index_uri);
    httpd_register_uri_handler(stream_httpd, &frame_uri);
  }
}
