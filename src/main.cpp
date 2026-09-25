#include <nanoflann.hpp>
#include <Arduino.h>
#include <vector>
#include <ArduinoEigen.h>

#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "queue.h"

static SemaphoreHandle_t xCoreSyncSemaphore;
static SemaphoreHandle_t distDataMutex;
static SemaphoreHandle_t inertialDataMutex;

#include <LSM6DSO32Sensor.h>

#define IMU_I2C Wire
// Components
LSM6DSO32Sensor AccGyr(&IMU_I2C);
//elapsedTime init for integration
unsigned long startTime = micros();
bool firstLoop = true;
unsigned long endTime = micros();
double elapsedTime = endTime - startTime;

#include <SparkFun_VL53L5CX_Library.h> //http://librarymanager/All#SparkFun_VL53L5CX
#include <Wire.h>
SparkFun_VL53L5CX ToF;
VL53L5CX_ResultsData distData; // Result data class structure

int imageWidth; //Used to pretty print output

// ST_ANGLES to interpret distances
double ST_PITCH_ANGLES_DEG[64] = {
    59.00, 64.00, 67.50, 70.00, 70.00, 67.50, 64.00, 59.00,
    64.00, 70.00, 72.90, 74.90, 74.90, 72.90, 70.00, 64.00,
    67.50, 72.90, 77.40, 80.50, 80.50, 77.40, 72.90, 67.50,
    70.00, 74.90, 80.50, 85.75, 85.75, 80.50, 74.90, 70.00,
    70.00, 74.90, 80.50, 85.75, 85.75, 80.50, 74.90, 70.00,
    67.50, 72.90, 77.40, 80.50, 80.50, 77.40, 72.90, 67.50,
    64.00, 70.00, 72.90, 74.90, 74.90, 72.90, 70.00, 64.00,
    59.00, 64.00, 67.50, 70.00, 70.00, 67.50, 64.00, 59.00,
};
double ST_YAW_ANGLES_DEG[64] = {
    135.00, 125.40, 113.20,  98.13,  81.87,  66.80,  54.60,  45.00,
    144.60, 135.00, 120.96, 101.31,  78.69,  59.04,  45.00,  35.40,
    156.80, 149.04, 135.00, 108.45,  71.55,  45.00,  30.96,  23.20,
    171.87, 168.69, 161.55, 135.00,  45.00,  18.45,  11.31,   8.13,
    188.13, 191.31, 198.45, 225.00, 315.00, 341.55, 348.69, 351.87,
    203.20, 210.96, 225.00, 251.55, 288.45, 315.00, 329.04, 336.80,
    215.40, 225.00, 239.04, 258.69, 281.31, 300.96, 315.00, 324.60,
    225.00, 234.60, 246.80, 261.87, 278.13, 293.20, 305.40, 315.00,
};

double pdist[64] = {};

//Sensor Tweaks
#define ToF_Sharpness 20
#define GyroFullScale 500
#define AccFullScale 8
#define microsteps 15
#define USE_ACCELEROMETER true

//Alignment Tweaks
#define max_leaf 12
#define alignment_iterations 2
#define filter_distance 0.4
#define min_matches 30
#define DO_ALIGN false
#define SEND_INTERMEDIATE_CLOUDS false

//Note that the type used for the point cloud is also tweakable (may use half_float for less memory usage)

Eigen::Quaterniond Orientation(1.0, 0.0, 0.0, 0.0);
//Orientation is assumed perfect
Eigen::Vector3d velocity(0.0, 0.0, 0.0);
Eigen::Matrix3d velocityCov = Eigen::Matrix3d::Zero();
Eigen::Vector3d position(0.0, 0.0, 0.0);
std::vector<Eigen::Matrix3d> positionCov;

#include <utils.h>
PointCloud<float> cloud;
using my_kd_tree_t = nanoflann::KDTreeSingleIndexDynamicAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>, PointCloud<float>, 3 /* dim */
        >;
my_kd_tree_t tree_index(3, cloud, max_leaf);
// Cannot call functions at top level to add points

struct logMessage {
  char * msg;
  std::vector<float>* data;
};

#define SerialPort Serial
static QueueHandle_t log_queue;
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

Eigen::Vector3d accoffset = {15.83,-29.63,-37.28};
Eigen::Vector3d gyrooffset = {-342.2, 448.3, 790.0};

static void I2CIntegrator(void * pvParameters) {
  //IO Core
  for(;;) {
    vTaskDelay(1);
    //Blink LED
    digitalWrite(LED_BUILTIN, (millis() / 1000) % 2);
    // Read gyroscope.
    int32_t gyro[3];
    AccGyr.Get_G_Axes(gyro);
    Eigen::Vector3d gyroscope = Eigen::Map<Eigen::Vector3i>(gyro).cast<double>() + gyrooffset;
    if (firstLoop) {firstLoop = false; startTime = micros();}
    endTime = micros();
    elapsedTime = double(endTime - startTime)/1000000.0; //seconds
    startTime = micros();
    xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
    for (int i = 0; i < microsteps; i++) { //Apply quaternions evenly through several steps
      Orientation *= Eigen::Quaterniond(cos(elapsedTime * (double(-gyroscope[0])*PI)/(1000*180*2*microsteps)), sin(elapsedTime * (double(-gyroscope[0])*PI)/(1000*180*2*microsteps)), 0, 0);
      Orientation *= Eigen::Quaterniond(cos(elapsedTime * (double(gyroscope[1])*PI)/(1000*180*2*microsteps)), 0, sin(elapsedTime * (double(gyroscope[1])*PI)/(1000*180*2*microsteps)), 0);
      Orientation *= Eigen::Quaterniond(cos(elapsedTime * (double(gyroscope[2])*PI)/(1000*180*2*microsteps)), 0, 0, sin(elapsedTime * (double(gyroscope[2])*PI)/(1000*180*2*microsteps)));
    }
    Orientation.normalize();
    xSemaphoreGive(inertialDataMutex);
    if (USE_ACCELEROMETER) {
      // Read accelerometer
      int32_t acc[3];
      AccGyr.Get_X_Axes(acc);
      //SENSOR Reference Frame
      Eigen::Vector3d accelerometer = (Eigen::Map<Eigen::Vector3i>(acc).cast<double>() + accoffset) * 9.8/1000;
      //GLOBAL Reference Frame
      Eigen::Vector3d trueAccel(Orientation * accelerometer);
      //Subtract Gravity
      trueAccel[2] -= 9.8;
      xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
      //Double integration step
      velocity += trueAccel * elapsedTime;
      velocityCov += Eigen::Matrix3d::Identity()*std::pow(0.013 * elapsedTime, 2);
      serial_write(String(std::sqrt(velocityCov(0, 0))));
      serial_write(String(velocity(0)) + ", " + String(velocity(1)) + ", " + String(velocity(2)));
      position += velocity * elapsedTime;
      positionCov.push_back(velocityCov * elapsedTime); //Each value of velocityCov will be findable in this data, thus velocityCov does not need to track past values
      //positionCov should be added rather than inverse variance weighted as movement is (TODO: Generally) dependent
      xSemaphoreGive(inertialDataMutex);
    }
    vTaskDelay(1);
    // Output data.
    std::vector<float> inertialUpdateData = {
      static_cast<float>(Orientation.w()), static_cast<float>(Orientation.x()), static_cast<float>(Orientation.y()), static_cast<float>(Orientation.z()),
      static_cast<float>(position[0]), static_cast<float>(position[1]), static_cast<float>(position[2]),
      static_cast<float>(velocity[0]), static_cast<float>(velocity[1]), static_cast<float>(velocity[2]),
      static_cast<float>(elapsedTime * 1000)
    };
    serial_write("inertialUpdate: ", inertialUpdateData);
    if (ToF.isDataReady()) {
      xSemaphoreTake(distDataMutex, portMAX_DELAY);
      if (ToF.getRangingData(&distData)){
        xSemaphoreGive(xCoreSyncSemaphore);
        serial_write("ToF Data read");
      }
      xSemaphoreGive(distDataMutex);
    }
  }
}

static void interpretDistances(VL53L5CX_ResultsData distData, std::array<Eigen::Vector3f, 64>& newCloud, std::array<boolean, 64>& hasData) {
  //The ST library returns the data transposed from zone mapping shown in datasheet
  //Pretty-print data with increasing y, decreasing x to reflect reality
  xSemaphoreTake(distDataMutex, portMAX_DELAY);
  for (int y = 0 ; y <= imageWidth * (imageWidth - 1) ; y += imageWidth) {
    for (int x = imageWidth - 1 ; x >= 0 ; x--) {
      double dist = distData.distance_mm[x + y]/1000.0; //Converted to meters
      if (pdist[x+y] != dist && dist != 0){ //Some extra complexity is added to ignore instances where the sensor does not give a new distance and reports the previous distance
        // === ST Lookup Table Method ===
        // Compute sin/cos for ST-calibrated pitch/yaw angles
        double pitch_rad = ST_PITCH_ANGLES_DEG[63-(x+y)] * DEG_TO_RAD;
        double yaw_rad = ST_YAW_ANGLES_DEG[63-(x+y)] * DEG_TO_RAD;
        // Compute ST ray directions (normalized)
        // Ray direction = (cos_yaw * cos_pitch, sin_yaw * cos_pitch, sin_pitch)
        // Negate X to match our lens-flip convention
        double st_ray_dir_x = -std::cos(yaw_rad) * std::cos(pitch_rad) / std::sin(pitch_rad);
        double st_ray_dir_y = std::sin(yaw_rad) * std::cos(pitch_rad) / std::sin(pitch_rad);
        //Point in SENSOR reference frame:
        Eigen::Vector3d point = {st_ray_dir_y * dist, st_ray_dir_x * dist, dist};
        //Point in GLOBAL reference frame:
        xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
        Eigen::Vector3f point_prime = (Orientation * point).cast<float>();
        newCloud[x+y] = {point_prime.x() + float(position[0]), point_prime.y() + float(position[1]), point_prime.z() + float(position[2])}; //Quaternion + position
        xSemaphoreGive(inertialDataMutex);
        pdist[x+y] = dist;
        hasData[x+y] = true;
      } else {
        hasData[x+y] = false;
      }
    }
  }
  xSemaphoreGive(distDataMutex);
}

static void aligner(void * pvParameters){
  for(;;){
    vTaskDelay(1);//for watchdog
    // SLAM: Core 0
    // Distance Sensor Output (Populating newCloud)
    if (xSemaphoreTake(xCoreSyncSemaphore, portMAX_DELAY) == pdTRUE) { //distData is read by the other core so only 1 core accesses I2C
      std::array<Eigen::Vector3f, 64> newCloud;
      std::array<boolean, 64> hasData;
      interpretDistances(distData, newCloud, hasData);
      //Alignment
      if (!cloud.pts.empty()) {
        for (int i = 0; i < alignment_iterations; i++) {
          if (SEND_INTERMEDIATE_CLOUDS) {
            std::vector<float> intermediateCloudData;
            for(int point = 0; point < 64; point++) {
              if (hasData[point]) {
                for (int i = 0; i < 3; i++) {
                  intermediateCloudData.push_back(newCloud[point][i]);
                }
              }
            }
            serial_write("intermediatePts: ", intermediateCloudData);
          }
          Eigen::MatrixXd A = Eigen::MatrixXd::Zero(64, 6);
          Eigen::VectorXd b = Eigen::VectorXd::Zero(64);
          int n = 0;
          for (int point = 0; point < 64; point++) { //Iterate over every point
            if (hasData[point]) {
              //Search kd tree to find closest point
              const size_t num_results = 3;
              nanoflann::KNNResultSet<float> resultSet(num_results);
              size_t ret_index[num_results];
              float out_dist_sqr[num_results]; //Square of distance
              resultSet.init(ret_index, out_dist_sqr);
              float query_pt[3] = {newCloud[point][0], newCloud[point][1], newCloud[point][2]};
              tree_index.findNeighbors(resultSet, query_pt, {});
              if (out_dist_sqr[2] <= filter_distance*filter_distance) { // For filtering, the closest point needs to be relatively close
                n++;
                //Normal vector is cross product of two vectors between points on the plane
                PointCloud<float>::Point pt1 = cloud.pts[ret_index[0]];
                PointCloud<float>::Point pt2 = cloud.pts[ret_index[1]];
                PointCloud<float>::Point pt3 = cloud.pts[ret_index[2]];
                //Eigen::Vector3f point1 = {pt1.x, pt1.y, pt1.z};
                float a_1 = pt1.x - pt2.x; float a_2 = pt1.y - pt2.y; float a_3 = pt1.z - pt2.z;
                float b_1 = pt1.x - pt3.x; float b_2 = pt1.y - pt3.y; float b_3 = pt1.z - pt3.z;
                float nx = (a_2 * b_3) - (a_3 * b_2); // normal vector values
                float ny = (a_3 * b_1) - (a_1 * b_3);
                float nz = (a_1 * b_2) - (a_2 * b_1);
                //This can be any scale, because increasing the scale scales up A and b, which is cancelled at Eigen::pseudoInverse(A)*b.
                float dx = pt1.x; float dy = pt1.y; float dz = pt1.z;
                float sx = newCloud[point][0]; float sy = newCloud[point][1]; float sz = newCloud[point][2];
                Eigen::VectorXd row(6); //Without (6), this has a runtime CommaInitializer error
                row << nz*sy - ny*sz, nx*sz - nz*sx, ny*sx - nx*sy, nx, ny, nz;
                double value = nx*dx + ny*dy + nz*dz - nx*sx - ny*sy - nz*sz;
                A.row(n - 1) = row;
                b(n - 1) = value;
              }
            }
          }
          serial_write("All points processed for iteration: " + String(i + 1) + ", and there were " + String(n) + " good points");
          Eigen::Matrix4d transform_opt;
          if (A.rows() == 0 || A.cols() == 0 || !A.allFinite() || A.cwiseAbs().maxCoeff() == 0.0 || n < min_matches || !DO_ALIGN) {
            //Revert to using identity matrix
            if (DO_ALIGN) {
              serial_write("bad or not enough data for cloud alignment");
            }
            transform_opt << 1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1;
          } else {
            //Free extra size of MatrixXd based on final value of n
            A.conservativeResize(n, 6);
            b.conservativeResize(n);
            Eigen::VectorXd x_opt = Eigen::pseudoInverse(A)*b;
            //Turn x_opt into the 4x4 matrix transform it optimized for
            transform_opt << 1, -x_opt(2), x_opt(1), x_opt(3), x_opt(2), 1, -x_opt(0), x_opt(4), -x_opt(1), x_opt(0), 1, x_opt(5), 0, 0, 0, 1;
          }
          Eigen::Quaterniond transform_quat(transform_opt.topLeftCorner<3,3>());
          Eigen::Transform<double, 3, Eigen::Affine> transform(transform_opt); //Can be applied directly to 3d vectors now
          xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
          Orientation *= transform_quat;
          position = transform * position;
          xSemaphoreGive(inertialDataMutex);
          //Apply optimal transformation to newCloud
          for(int point = 0; point < 64; point++) {
            if (hasData[point]) {
              newCloud[point] = transform.cast<float>() * newCloud[point];
            }
          }
        }
      }
      //All iterations completed, newCloud now has points that line up with previous points (Or nothing happened if cloud.pts.empty())
      //Update Kd Tree
      size_t old_size = cloud.kdtree_get_point_count();
      std::vector<float> newCloudData;
      for(int point = 0; point < 64; point++) {
        if (hasData[point]) {
          cloud.pts.push_back({newCloud[point][0], newCloud[point][1], newCloud[point][2]});
          for (int i = 0; i < 3; i++) {
            newCloudData.push_back(newCloud[point][i]);
          }
        }
      }
      serial_write("newPts: ", newCloudData);
      size_t new_size = cloud.kdtree_get_point_count();
      //Add new points to index
      //This is the only O(n) part because tree is reformed after each chunk, luckily only done 15Hz not 15*64Hz
      tree_index.addPoints(old_size, new_size - 1);
      dump_mem_usage();
    }
  }
}

TaskHandle_t InitTask;
TaskHandle_t Core0Task;
TaskHandle_t Core1Task;
TaskHandle_t SerialLog;

static void calibrator(void * pvParameters) {
  int calibratorStartTime = micros();
  int n = 0;
  Eigen::Vector3d accSum = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyroSum = Eigen::Vector3d::Zero();
  while((micros() - calibratorStartTime)/1000000.0 < 5.0) {
    vTaskDelay(1);
    int32_t acc[3];
    AccGyr.Get_X_Axes(acc);
    Eigen::Vector3d accelerometer = (Eigen::Map<Eigen::Vector3i>(acc).cast<double>());
    accelerometer[2] -= 1000;
    accSum += accelerometer;
    int32_t gyro[3];
    AccGyr.Get_G_Axes(gyro);
    Eigen::Vector3d gyroscope = Eigen::Map<Eigen::Vector3i>(gyro).cast<double>();
    gyroSum += gyroscope;
    n++;
  }
  accoffset = -accSum / n;
  gyrooffset = -gyroSum / n;
  xTaskCreatePinnedToCore(
    aligner,
    "Core0Task",
    32768,
    NULL,
    2,
    &Core0Task,
    0
  );
  xTaskCreatePinnedToCore(
    I2CIntegrator,
    "Core1Task",
    8192,
    NULL,
    3,
    &Core1Task,
    1
  );
  xTaskCreatePinnedToCore(
    SerialLogger,
    "SerialLog",
    4096,
    NULL,
    2,
    &SerialLog,
    1
  );
  vTaskDelete(NULL);
}

void setup()
{
  delay(5);
  // Led.
  pinMode(LED_BUILTIN, OUTPUT);

  // Initialize serial for output.
  SerialPort.begin(115200);
  delay(50);
  while (!SerialPort) {
    delay(10);
  }
  SerialPort.write("Setting up...");

  // Initialize I2C bus.
  IMU_I2C.begin();
  AccGyr.begin();
  AccGyr.Enable_X();
  AccGyr.Enable_G();
  AccGyr.Set_G_FS(GyroFullScale);
  AccGyr.Set_X_FS(AccFullScale);
  AccGyr.Set_X_ODR(208.0f); // Set Accelerometer to 208 Hz
  AccGyr.Set_G_ODR(208.0f); // Set Gyroscope to 208 Hz
  delay(20);

  Wire.begin(); //This resets to 100kHz I2C
  Wire.setClock(400000); //IMU has max I2C freq of 400kHz 
  //SerialPort.println("Initializing sensor board. This can take up to 10s. Please wait.");
  if (ToF.begin() == false) {
    //SerialPort.println(F("ToF Sensor not found - check your wiring. Freezing"));
    while (1); 
  }
  ToF.setSharpenerPercent(ToF_Sharpness);
  ToF.setResolution(8*8); //Enable all 64 pads
  ToF.setRangingFrequency(15);
  imageWidth = sqrt(ToF.getResolution()); //Calculate printing width
  ToF.startRanging();
  dump_mem_usage();
  delay(1);
  xCoreSyncSemaphore = xSemaphoreCreateBinary();
  distDataMutex = xSemaphoreCreateMutex();
  inertialDataMutex = xSemaphoreCreateMutex();
  log_queue = xQueueCreate(16, sizeof(logMessage*));
  xTaskCreatePinnedToCore(
    calibrator,
    "InitTask",
    8192,
    NULL,
    2,
    &InitTask,
    0
  );
}

void loop() {}