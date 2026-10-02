#include <nanoflann.hpp>
#include <Arduino.h>
#include <vector>
#include <ArduinoEigen.h>

#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"

static SemaphoreHandle_t xCoreSyncSemaphore;
static SemaphoreHandle_t distDataMutex;
static SemaphoreHandle_t inertialDataMutex;

#include <LSM6DSO32Sensor.h>

#define IMU_I2C Wire
// Components
LSM6DSO32Sensor AccGyr(&IMU_I2C);

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
#define filter_stdevs 2
#define DO_ALIGN false

//Note that the type used for the point cloud is also tweakable (may use half_float for less memory usage)

#include <utils.h>
PointCloud<float> cloud;
using my_kd_tree_t = nanoflann::KDTreeSingleIndexDynamicAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>, PointCloud<float>, 3 /* dim */
        >;
my_kd_tree_t tree_index(3, cloud, max_leaf);
// Cannot call functions at top level to add points

#include <serial_write.h>

Eigen::Vector3d accOffset = {15.83,-29.63,-37.28};
Eigen::Vector3d accVari = Eigen::Vector3d::Zero();
Eigen::Vector3d gyroOffset = {-342.2, 448.3, 790.0};
Eigen::Vector3d gyroVari = Eigen::Vector3d::Zero();

Eigen::Quaterniond Orientation(1.0, 0.0, 0.0, 0.0);
Eigen::Vector3d velocity(0.0, 0.0, 0.0);
Eigen::Matrix3d velocityCov = Eigen::Matrix3d::Zero();
Eigen::Vector3d position(0.0, 0.0, 0.0);

std::vector<Eigen::Matrix<double, 6, 6>> poseCov; 
// X, Y, Z -> relative to last iteration
// 3 orientation values -> absolute

static void I2CIntegrator(void * pvParameters) {
  //elapsedTime init for integration
  unsigned long startTime = micros();
  unsigned long endTime;
  double elapsedTime;
  Eigen::Matrix<double, 6, 6> tempCov = Eigen::Matrix<double, 6, 6>::Zero();
  //IO Core
  for(;;) {
    vTaskDelay(1);

    // Blink LED
    digitalWrite(LED_BUILTIN, (millis() / 1000) % 2);

    // Get Elapsed Time
    endTime = micros();
    elapsedTime = double(endTime - startTime)/1000000.0; //seconds
    startTime = endTime;

    // Read gyroscope.
    int32_t gyro[3];
    AccGyr.Get_G_Axes(gyro);
    Eigen::Vector3d gyroscope = Eigen::Map<Eigen::Vector3i>(gyro).cast<double>() + gyroOffset;
    double radConvert = elapsedTime * (0.001) * (PI/180.0);
    Eigen::Vector3d dAngleHalf = gyroscope * radConvert * (0.5); //Avoid int casting
    xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
    Orientation *= Eigen::Quaterniond(cos(dAngleHalf.norm()), sin(dAngleHalf[0]), sin(dAngleHalf[1]), sin(dAngleHalf[2]));
    Orientation.normalize();
    xSemaphoreGive(inertialDataMutex);

    // Read accelerometer
    int32_t acc[3];
    AccGyr.Get_X_Axes(acc);
    // SENSOR Reference Frame
    Eigen::Vector3d accelerometer = (Eigen::Map<Eigen::Vector3i>(acc).cast<double>() + accOffset) * 9.8066/1000;
    xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
    // GLOBAL Reference Frame
    Eigen::Vector3d trueAccel(Orientation * accelerometer);
    // Subtract Gravity
    trueAccel[2] -= 9.8066;
    // Double integration step
    velocity += trueAccel * elapsedTime;
    position += velocity * elapsedTime;
    // TODO: Update Kalman Filter
    //UNCERTAINTIES:
    // accelerometer: accVari * 9.8066/1000
    // orientation: previous value + gyroVari
    
    xSemaphoreGive(inertialDataMutex);
    vTaskDelay(1);

    // Output data.
    std::vector<float> inertialUpdateData = {
      static_cast<float>(Orientation.w()), static_cast<float>(Orientation.x()), static_cast<float>(Orientation.y()), static_cast<float>(Orientation.z()),
      static_cast<float>(position[0]), static_cast<float>(position[1]), static_cast<float>(position[2]),
      static_cast<float>(velocity[0]), static_cast<float>(velocity[1]), static_cast<float>(velocity[2]),
      static_cast<float>(elapsedTime * 1000)
    };
    serial_write("inertialUpdate: ", inertialUpdateData);

    // Read VL53L5CX I2C data
    if (ToF.isDataReady()) {
      xSemaphoreTake(distDataMutex, portMAX_DELAY);
      if (ToF.getRangingData(&distData)){
        xSemaphoreGive(xCoreSyncSemaphore); // Might replace with queue?
        serial_write("ToF Data read");
      }
      xSemaphoreGive(distDataMutex);
      poseCov.push_back(tempCov);
      tempCov = Eigen::Matrix<double, 6, 6>::Zero();
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
      if (pdist[x+y] != dist && dist != 0) { //Some extra complexity is added to ignore instances where the sensor does not give a new distance and reports the previous distance
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
        newCloud[x+y] = {st_ray_dir_y * dist, st_ray_dir_x * dist, dist};
        /*Point in GLOBAL reference frame:
        xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
        Eigen::Vector3f point_prime = (Orientation * point).cast<float>();
        newCloud[x+y] = {point_prime.x() + float(position[0]), point_prime.y() + float(position[1]), point_prime.z() + float(position[2])}; //Quaternion + position
        xSemaphoreGive(inertialDataMutex);*/
        pdist[x+y] = dist;
        hasData[x+y] = true;
      } else {
        hasData[x+y] = false;
      }
    }
  }
  xSemaphoreGive(distDataMutex);
}

static void surfaceMatcher(void * pvParameters) {
  for(;;) {
    vTaskDelay(1);//for watchdog
    // SLAM: Core 0
    // Distance Sensor Output (Populating newCloud)
    if (xSemaphoreTake(xCoreSyncSemaphore, portMAX_DELAY) == pdTRUE) { //distData is read by the other core so only 1 core accesses I2C
      std::array<Eigen::Vector3f, 64> newCloud;
      std::array<boolean, 64> hasData;
      interpretDistances(distData, newCloud, hasData);
      // TODO: Turn pointCloud in sensor space into representative planes
      if (!cloud.pts.empty()) {
        // TODO: Find any matching surfaces that appear to be the same surface
        // TODO: Write the alignment of the surfaces as a square to minimize
      }
    }
  }
}
      

static void aligner(void * pvParameters) {
  for(;;) {
    //Alignment
    if (!cloud.pts.empty()) {
      Eigen::MatrixXd A = Eigen::MatrixXd::Zero(64, 6);
      Eigen::VectorXd b = Eigen::VectorXd::Zero(64);
      int n = 0;
      if (A.rows() == 0 || A.cols() == 0 || !A.allFinite() || A.cwiseAbs().maxCoeff() == 0.0) {
        //Failed
      } else {
        //Free extra size of MatrixXd based on final value of n
        A.conservativeResize(n, 6);
        b.conservativeResize(n);
        Eigen::VectorXd x_opt = Eigen::pseudoInverse(A)*b;
      }
    }
  }
}

TaskHandle_t InitTask;
TaskHandle_t Core0Task;
TaskHandle_t matchTask;
TaskHandle_t Core1Task;
TaskHandle_t SerialLog;

static void calibrator(void * pvParameters) {
  int calibratorStartTime = micros();
  int n = 0;
  Eigen::Vector3d accMean = Eigen::Vector3d::Zero();
  Eigen::Vector3d accM2 = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyroMean = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyroM2 = Eigen::Vector3d::Zero();
  while((micros() - calibratorStartTime)/1000000.0 < 8.0) { //Repeat Welford's Algorithm to get variances
    vTaskDelay(1);
    n++;
    int32_t acc[3];
    AccGyr.Get_X_Axes(acc);
    Eigen::Vector3d accelerometer = (Eigen::Map<Eigen::Vector3i>(acc).cast<double>());
    accelerometer[2] -= 1000;
    Eigen::Vector3d accDelta = accelerometer - accMean;
    accMean += accDelta / n;
    accM2 += (accelerometer - accMean) * accDelta;
    int32_t gyro[3];
    AccGyr.Get_G_Axes(gyro);
    Eigen::Vector3d gyroscope = Eigen::Map<Eigen::Vector3i>(gyro).cast<double>();
    Eigen::Vector3d gyroDelta = gyroscope - gyroMean;
    gyroMean += gyroDelta / n;
    gyroM2 += (gyroscope - gyroMean) * gyroDelta;
  }
  accOffset = -accMean;
  accVari = accM2 / (n-1); //Bessel's Correction
  gyroOffset = -gyroMean;
  gyroVari = gyroM2 / (n-1);
  xTaskCreatePinnedToCore(
    aligner,
    "Core0Task",
    32768,
    NULL,
    3,
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
    surfaceMatcher,
    "matchTask",
    8192,
    NULL,
    3,
    &matchTask,
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
  vTaskDelete(NULL); //Tasks reaching the end causes an error
}

void setup() {
  delay(5);
  // Led.
  pinMode(LED_BUILTIN, OUTPUT);

  // Initialize serial for output.
  Serial.begin(115200);
  delay(50);
  while (!Serial) {
    delay(10);
  }
  Serial.write("Setting up...");

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
  Serial.println("Initializing sensor board. This can take up to 10s. Please wait.");
  if (ToF.begin() == false) {
    Serial.println(F("ToF Sensor not found - check your wiring. Freezing"));
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