#include <Arduino.h>
#include <vector>
#include "queue.h"

struct logMessage {
  char * msg;
  std::vector<float>* data;
};

#define SerialPort Serial
static QueueHandle_t log_queue = xQueueCreate(16, sizeof(logMessage*));;
static void serial_write(const String msg, std::vector<float> data = {}) {
  char * msgCopy = strdup(msg.c_str());
  if (msgCopy == NULL) {return;}
  std::vector<float> * dataCopy = new std::vector<float>(data);
  if (dataCopy == NULL) {
    free(msgCopy);
    return;
  }
  logMessage* message = new logMessage();
  if (message == NULL) {
    free(msgCopy);
    delete dataCopy;
    return;
  }
  message->msg = msgCopy;
  message->data = dataCopy;
  if (xQueueSend(log_queue, &message, 0) != pdPASS) {
    // If the queue is full, delete everything cleanly in one place
    free(message->msg);
    delete message->data;
    delete message; 
  }
}
void float2Bytes(byte bytes_temp[4],float float_variable) { 
  memcpy(bytes_temp, (unsigned char*) (&float_variable), 4);
}
const byte START_MARKER = 0x7E;
const byte END_MARKER = 0x7F;
const byte ESCAPE_BYTE = 0x7D;
static void SerialLogger(void * pvParameters) {
  logMessage* message;
  char * msg;
  std::vector<float> * data;
  for (;;) {
    xQueueReceive(log_queue, &message, portMAX_DELAY);
    msg = message->msg;
    data = message->data;
    SerialPort.print(msg);
    if (data != nullptr && !data->empty()) {
      SerialPort.write(START_MARKER);
      for (const auto& val : *data) {
        byte bytes[4];
        float2Bytes(bytes, val);
        for (int i = 0; i < 4; i++) {
          if (bytes[i] == END_MARKER || bytes[i] == ESCAPE_BYTE){
            SerialPort.write(ESCAPE_BYTE);
            SerialPort.write(bytes[i] ^ 0x20); // Swaps 6th bit, do again on receiver after escape byte to reverse.
          } else {
            SerialPort.write(bytes[i]);
          }
        }
      }
    SerialPort.write(END_MARKER);
  }
    SerialPort.println();
    free(msg);
    delete data;
  }
}