#include <nanoflann.hpp>
#include <Arduino.h>
#include <vector>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <ArduinoEigen.h>

#include <LSM6DSO32Sensor.h>

#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "queue.h"

#ifdef ARDUINO_SAM_DUE
  #define DEV_I2C Wire1
#else
  #define DEV_I2C Wire
#endif
#define SerialPort Serial
static QueueHandle_t serial_queue;
static void serial_write(const String msg) {
  char * msgCopy = strdup(msg.c_str());
  if (msgCopy == NULL) {return;}
  if (xQueueSend(serial_queue, &msgCopy, 0) != pdPASS) {
    free(msgCopy);
  }
}

static void SerialLogger(void * pvParameters) {
  char *msg;
  for (;;) {
    xQueueReceive(serial_queue, &msg, portMAX_DELAY);
    SerialPort.println(msg);
    free(msg);
  }
}

// Components
LSM6DSO32Sensor AccGyr(&DEV_I2C);

unsigned long startTime = micros();
bool firstLoop = true;
unsigned long endTime = micros();
double elapsedTime = endTime - startTime;
unsigned long startTimeICP = micros();
bool firstLoopICP = true;
unsigned long endTimeICP = micros();
double elapsedTimeICP = endTimeICP - startTimeICP;

#include <SparkFun_VL53L5CX_Library.h> //http://librarymanager/All#SparkFun_VL53L5CX
#include <Wire.h>
SparkFun_VL53L5CX ToF;
VL53L5CX_ResultsData distData; // Result data class structure, 1356 byes of RAM
static SemaphoreHandle_t xCoreSyncSemaphore;
static SemaphoreHandle_t distDataMutex;
static SemaphoreHandle_t inertialDataMutex;

int imageWidth = 0; //Used to pretty print output

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

double pdist[64];

//TWEAKABLE VALUES
#define ToF_Sharpness 20
#define microsteps 15
#define max_leaf 12
#define ICP_Iterations 3
#define filter_distance 0.4
#define min_matches 30
#define VELOCITY_FILTER_RATIO 0.1
#define reflectance_percent_to_meters 0.02 // 20% is the maximum difference in reflectivity to be paired
#define radians_to_meters 2
#define position_to_meters 0.05
#define DAMPEN_MOTION true
#define USE_ICP true
#define USE_ACCELEROMETER false
#define SEND_INTERMEDIATE_CLOUDS false

const double accoffset[3] = {36.8,-2.87,-36.5};
const double gyrooffset[3] = {-342.2, 448.3, 790.0};
//Note that the type used for the point cloud is also tweakable (may use half_float for less memory usage)

Eigen::Quaterniond Orientation(1.0, 0.0, 0.0, 0.0);
Eigen::Vector3d velocity(0.0, 0.0, 0.0);
Eigen::Vector3d position(0.0, 0.0, 0.0);
//From utils.h, gives definition of PointCloud that can be used to initialize a kd tree
template <typename T>
struct PointCloud
{
    struct Point
    {
        T x, y, z, r;
    };

    using coord_t = T;  //!< The type of each coordinate

    std::vector<Point> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else if (dim == 2)
            return pts[idx].z;
        else
            return pts[idx].r;
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};
inline void dump_mem_usage() {
    FILE* f = fopen("/proc/self/statm", "rt");
    if (!f) return;
    char   str[300];
    size_t n = fread(str, 1, 200, f);
    str[n]   = 0;
    printf("MEM: %s\n", str);
    fclose(f);
}

PointCloud<float> cloud;
using my_kd_tree_t = nanoflann::KDTreeSingleIndexDynamicAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>, PointCloud<float>, 4 /* dim */
        >;
my_kd_tree_t tree_index(4, cloud, max_leaf);
// Cannot call functions at top level to add points

static void I2CIntegrator(void * pvParameters) {
  for(;;) {
    vTaskDelay(1);
    //IO Core
    if ((millis() / 1000) % 2 == 0) {
      digitalWrite(LED_BUILTIN, HIGH);
    } else {
      digitalWrite(LED_BUILTIN, LOW);
    }
    // Read accelerometer and gyroscope.
    int32_t acc[3];
    int32_t gyro[3];
    double accelerometer[3];
    double gyroscope[3];
    AccGyr.Get_X_Axes(acc);
    AccGyr.Get_G_Axes(gyro);
    //Subtract Sample sums
    for(int i = 0; i < 3; i++) {
      //Changing to better type
      accelerometer[i] = acc[i];
      gyroscope[i] = gyro[i];
      //Calibration Offset
      accelerometer[i] += accoffset[i];
      gyroscope[i] += gyrooffset[i];
    }
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
    Eigen::Vector3d trueAccel(Orientation * Eigen::Vector3d(accelerometer[0], accelerometer[1], accelerometer[2]));
    //From milliGs to m/s^2
    trueAccel *= 9.8/1000;
    trueAccel[2] -= 9.8;
    xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
    if (USE_ACCELEROMETER) {
      for(int i = 0; i < 3; i++) {
        //Double integration step (Not messing with trapezoids, ICP should fix anyways)
        velocity[i] += trueAccel[i] * elapsedTime;
        position[i] += velocity[i] * elapsedTime;
      }
    }
    xSemaphoreGive(inertialDataMutex);
    vTaskDelay(1);
    // Output data.
    String orientationEstimate = "Orient, "+String(Orientation.w())+", "+String(Orientation.x())+", "+String(Orientation.y())+", "+String(Orientation.z());
    String positionEstimate = "Pos, "+String(position[0])+", "+String(position[1])+", "+String(position[2]);
    String velocityEstimate = "Vel, "+String(velocity[0])+", "+String(velocity[1])+", "+String(velocity[2]);
    String inertialUpdate = orientationEstimate + ", " + positionEstimate + ", " + velocityEstimate + ", Time = " + String(elapsedTime * 1000) + " ms";
    serial_write(inertialUpdate);
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

static void interpretDistances(VL53L5CX_ResultsData distData, std::array<Eigen::Vector3f, 64>& newCloud, std::array<float, 64>& reflect, std::array<boolean, 64>& hasData) {
  //The ST library returns the data transposed from zone mapping shown in datasheet
  //Pretty-print data with increasing y, decreasing x to reflect reality
  xSemaphoreTake(distDataMutex, portMAX_DELAY);
  for (int y = 0 ; y <= imageWidth * (imageWidth - 1) ; y += imageWidth)
  {
    for (int x = imageWidth - 1 ; x >= 0 ; x--)
    {
      double dist = distData.distance_mm[x + y]/1000.0;
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
        //Point in Sensor's reference frame:
        Eigen::Vector3d point = {st_ray_dir_y * dist, st_ray_dir_x * dist, dist}; // divide by 1000 to convert mm to meters
        //Point in GLOBAL reference frame:
        xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
        Eigen::Vector3f point_prime = (Orientation * point).cast<float>();
        newCloud[x+y] = {point_prime.x() + float(position[0]), point_prime.y() + float(position[1]), point_prime.z() + float(position[2])}; //Quaternion + position
        xSemaphoreGive(inertialDataMutex);
        reflect[x+y] = float(distData.reflectance[x+y] * reflectance_percent_to_meters); // Max 255, normal max 100 as a percentage of reflected light.
        pdist[x+y] = dist;
        hasData[x+y] = true;
      } else {
        hasData[x+y] = false;
      }
    }
  }
  xSemaphoreGive(distDataMutex);
}

static void runICP(void * pvParameters){
  for(;;){
    vTaskDelay(1);//for watchdog
    // SLAM: Core 0
    // Distance Sensor Output (Populating newCloud)
    if (xSemaphoreTake(xCoreSyncSemaphore, portMAX_DELAY) == pdTRUE){ //distData is read by the other core so only 1 core accesses I2C
      std::array<Eigen::Vector3f, 64> newCloud;
      std::array<float, 64> reflect;
      std::array<boolean, 64> hasData;
      interpretDistances(distData, newCloud, reflect, hasData);
      //ICP
      if (!cloud.pts.empty()){
        for (int i = 0; i < ICP_Iterations; i++){
          if (SEND_INTERMEDIATE_CLOUDS) {
            String pointMsg = "OldPts";
            for(int point = 0; point < 64; point++) {
              if (hasData[point]) {
                pointMsg += "," + String(newCloud[point][0]) + "," + String(newCloud[point][1]) + "," + String(newCloud[point][2]);
              }
            }
            serial_write(pointMsg);//TODO: Sending data takes 5ms, that is a problem! I think I need to send the data as a binary.
          }
          Eigen::MatrixXd A = Eigen::MatrixXd::Zero(70, 6);
          Eigen::VectorXd b = Eigen::VectorXd::Zero(70);
          int n = 0;
          for (int point = 0; point < 64; point++){ //Iterate over every point
            if (hasData[point]) {
              //HERE IS WHERE TO FIND COMPUTE CYCLE OPTIMIZATIONS
              //Search kd tree to find closest point
              const size_t num_results = 3;
              nanoflann::KNNResultSet<float> resultSet(num_results);
              size_t ret_index[num_results];
              float out_dist_sqr[num_results]; //Square of distance
              resultSet.init(ret_index, out_dist_sqr);
              float query_pt[4] = {newCloud[point][0], newCloud[point][1], newCloud[point][2], reflect[point]};
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
          if (DAMPEN_MOTION) {
            std::array<Eigen::VectorXd, 6> final_rows;
            final_rows[0].resize(6);
            final_rows[0] << radians_to_meters, 0, 0, 0, 0, 0;
            final_rows[1].resize(6);
            final_rows[1] << 0, radians_to_meters, 0, 0, 0, 0;
            final_rows[2].resize(6);
            final_rows[2] << 0, 0, radians_to_meters, 0, 0, 0;
            final_rows[3].resize(6);
            final_rows[3] << 0, 0, 0, position_to_meters, 0, 0;
            final_rows[4].resize(6);
            final_rows[4] << 0, 0, 0, 0, position_to_meters, 0;
            final_rows[5].resize(6);
            final_rows[5] << 0, 0, 0, 0, 0, position_to_meters;
            for (int i = 0; i < 6; i++) {
              A.row(n + i) = final_rows[i];
              b(n + i) = 0;
            }
          }
          serial_write("All points processed for iteration: " + String(i + 1) + ", and there were " + String(n) + " good points");
          Eigen::Matrix4d transform_opt;
          if (A.rows() == 0 || A.cols() == 0 || !A.allFinite() || A.cwiseAbs().maxCoeff() == 0.0 || n < min_matches || !USE_ICP){
            //Revert to using identity matrix
            if (USE_ICP) {
              serial_write("bad or not enough data for cloud alignment");
            }
            transform_opt << 1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1;
          } else {
            //Free extra size of MatrixXd based on final value of n
            if (DAMPEN_MOTION) {
              A.conservativeResize(n + 6, 6);
              b.conservativeResize(n + 6);
            } else {
              A.conservativeResize(n, 6);
              b.conservativeResize(n);
            }
            Eigen::VectorXd x_opt = Eigen::pseudoInverse(A)*b;
            //Turn x_opt into the 4x4 matrix transform it optimized for
            transform_opt << 1, -x_opt(2), x_opt(1), x_opt(3), x_opt(2), 1, -x_opt(0), x_opt(4), -x_opt(1), x_opt(0), 1, x_opt(5), 0, 0, 0, 1;
          }
          Eigen::Quaterniond transform_quat(transform_opt.topLeftCorner<3,3>());
          Eigen::Transform<double, 3, Eigen::Affine> transform(transform_opt); //Can be applied directly to 3d vectors now

          //Rather than applying the euler angles, this allows us to only apply the transformation that was optimized for, not what it pretends to be
          //Apply optimal transformation to sensor quaternion
          xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
          Orientation *= transform_quat;
          //Apply optimal transformation to sensor position (same as pointcloud)
          Eigen::Vector3d delta = transform * position - position;
          position = transform * position;
          //Update velocity (drag towards new value at 1/2 ratio)
          if (firstLoopICP) {
            firstLoopICP = false;
            elapsedTimeICP = 2; //Set to 2 seconds if unknown, effectively a higher filter ratio
            startTimeICP = micros();
          } else {
            endTimeICP = micros();
            elapsedTimeICP = double(endTimeICP - startTimeICP)/1000000.0; //seconds
            startTimeICP = micros();
          }
          velocity += delta * VELOCITY_FILTER_RATIO/elapsedTimeICP;
          xSemaphoreGive(inertialDataMutex);
          //Apply optimal transformation to newCloud
          for(int point = 0; point < 64; point++) {
            if (hasData[point]) {
              newCloud[point] = transform.cast<float>() * newCloud[point];
            }
          }
        }
      }
      //All ICP iterations completed, newCloud now has points that line up with previous points (Or nothing happened if cloud.pts.empty())
      //Update Kd Tree
      size_t old_size = cloud.kdtree_get_point_count();
      String pointMsg = "NewPts";
      for(int point = 0; point < 64; point++) {
        if (hasData[point]) {
          //Add point to cloud
          cloud.pts.push_back({newCloud[point][0], newCloud[point][1], newCloud[point][2], reflect[point]});
          //Send point data
          pointMsg += "," + String(newCloud[point][0]) + "," + String(newCloud[point][1]) + "," + String(newCloud[point][2]);
        }
      }
      serial_write(pointMsg);//TODO: Sending data takes 5ms, that is a problem! I think I need to send the data as a binary.
      size_t new_size = cloud.kdtree_get_point_count();
      //Add new points to index
      //This is the only O(n) part because tree is reformed after each chunk, luckily only done 15Hz not 15*64Hz
      tree_index.addPoints(old_size, new_size - 1);
      //TODO: Loop Closure/RANSAC?
      dump_mem_usage();
    }
  }
}

TaskHandle_t Core0Task;
TaskHandle_t Core1Task;
TaskHandle_t SerialLog;

void setup()
{
  for (int i = 0; i < 64; i++) {
    pdist[i] = 0.0;
  }
  delay(20);
  // Led.
  pinMode(LED_BUILTIN, OUTPUT);

  // Initialize serial for output.
  SerialPort.begin(115200);
  delay(300);
  while (!SerialPort) {
    delay(10);
  }
  SerialPort.write("Setting up...");

  // Initialize I2C bus.
  DEV_I2C.begin();
  AccGyr.begin();
  AccGyr.Enable_X();
  AccGyr.Enable_G();
  AccGyr.Set_G_FS(500);
  AccGyr.Set_X_FS(8);
  delay(20);

  Wire.begin(); //This resets to 100kHz I2C
  Wire.setClock(400000); //IMU has max I2C freq of 400kHz 
  SerialPort.println("Initializing sensor board. This can take up to 10s. Please wait.");
  if (ToF.begin() == false) {
    SerialPort.println(F("ToF Sensor not found - check your wiring. Freezing"));
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
    runICP,
    "Core0Task",
    32768,
    NULL,
    2,
    &Core0Task,
    0
  ); //Last param pins this task to core 0
  xTaskCreatePinnedToCore(
    I2CIntegrator,
    "Core1Task",
    16384,
    NULL,
    3,
    &Core1Task,
    1
  );
  serial_queue = xQueueCreate(16, sizeof(const char *));
  xTaskCreatePinnedToCore(
    SerialLogger,
    "SerialLog",
    4096,
    NULL,
    2,
    &SerialLog,
    1
  );
}

void loop() {}