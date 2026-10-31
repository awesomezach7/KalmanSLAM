//TWEAKABLE PARAMETERS

#define ToF_Sharpness 20
#define GyroFullScale 500
#define AccFullScale 8
#define USE_ACCELEROMETER true

#define max_leaf 12
#define point_plane_max_stdevs 2
#define filter_stdevs 2
#define DO_ALIGN false

//GLOBALS AND LIBRARIES

#include <nanoflann.hpp>
#include <Arduino.h>
#include <vector>
#include <random>
#include <ArduinoEigen.h>

#define watchdog_delay pdMS_TO_TICKS(1)

static SemaphoreHandle_t inertialDataMutex;
static SemaphoreHandle_t i2cMutex;

#include <serial_write.h>

#include <utils.h>
PointCloud<float> cloud;
using my_kd_tree_t = nanoflann::KDTreeSingleIndexDynamicAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>, PointCloud<float>, 3 /* dim */
        >;
my_kd_tree_t tree_index(3, cloud, max_leaf);

#include <LSM6DSO32Sensor.h>

#define IMU_I2C Wire
#define IMU_I2C_MAX_FREQ 400000
LSM6DSO32Sensor AccGyr(&IMU_I2C);
#define gravAccel 9.8066

#include <SparkFun_VL53L5CX_Library.h>
#include <Wire.h>
SparkFun_VL53L5CX ToF;
#define ToFWidth 8

constexpr double ST_PITCH_ANGLES_DEG[64] = {
    59.00, 64.00, 67.50, 70.00, 70.00, 67.50, 64.00, 59.00,
    64.00, 70.00, 72.90, 74.90, 74.90, 72.90, 70.00, 64.00,
    67.50, 72.90, 77.40, 80.50, 80.50, 77.40, 72.90, 67.50,
    70.00, 74.90, 80.50, 85.75, 85.75, 80.50, 74.90, 70.00,
    70.00, 74.90, 80.50, 85.75, 85.75, 80.50, 74.90, 70.00,
    67.50, 72.90, 77.40, 80.50, 80.50, 77.40, 72.90, 67.50,
    64.00, 70.00, 72.90, 74.90, 74.90, 72.90, 70.00, 64.00,
    59.00, 64.00, 67.50, 70.00, 70.00, 67.50, 64.00, 59.00,
};
constexpr double ST_YAW_ANGLES_DEG[64] = {
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

Eigen::Matrix3f decompressCov(std::array<float, 6> compressed) {
  float terms[9] = {
    compressed[0], compressed[1], compressed[2], 
    compressed[1], compressed[3], compressed[4],
    compressed[2], compressed[4], compressed[5]};
  return Eigen::Map<Eigen::Matrix3f>(terms);
}
std::array<float, 6> compressCov(Eigen::Matrix3f uncompressed) {
  return {uncompressed(0, 0), uncompressed(0, 1), uncompressed(0, 2), uncompressed(1, 1), uncompressed(1, 2), uncompressed(2, 2)};
}
static Eigen::Vector3f getNorm(const Eigen::Matrix3f& covMatrix) {
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(covMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.matrixU().col(2);
}
struct plane{
  std::array<float, 6> covariances;
  Eigen::Vector3f mean;
  int iteration;
};

int iteration = 0;
std::vector<plane> allPlanes;

Eigen::Quaterniond Orientation(1.0, 0.0, 0.0, 0.0);
Eigen::Vector3d prevPosition(0.0, 0.0, 0.0);
Eigen::Vector3d position(0.0, 0.0, 0.0);

static void imuIntegrator(Eigen::Vector3d accOffset, Eigen::Vector3d accVari, Eigen::Vector3d gyroOffset, Eigen::Vector3d gyroVari) {
  unsigned long startTime = micros();
  unsigned long elapsedTimeMicros;
  double elapsedTimeSeconds;
  
  Eigen::Vector3d velocity(0.0, 0.0, 0.0);
  
  for(;;) {
    vTaskDelay(watchdog_delay);
    digitalWrite(LED_BUILTIN, (millis() / 1000) % 2);
    
    elapsedTimeMicros = micros() - startTime;
    elapsedTimeSeconds = double(elapsedTimeMicros) / 1000000.0;
    startTime = micros();

    int32_t gyro[3];
    xSemaphoreTake(i2cMutex, portMAX_DELAY);
    AccGyr.Get_G_Axes(gyro);
    xSemaphoreGive(i2cMutex);
    Eigen::Vector3d gyroscope = {static_cast<double>(gyro[0]), static_cast<double>(gyro[1]), static_cast<double>(gyro[2])};
    gyroscope += gyroOffset;
    double radConvert = elapsedTimeSeconds * (0.001) * (PI/180.0);
    Eigen::Vector3d dAngleHalf = gyroscope * radConvert * (0.5);
    xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
    Orientation *= Eigen::Quaterniond(cos(dAngleHalf.norm()), sin(dAngleHalf[0]), 
                                      sin(dAngleHalf[1]), sin(dAngleHalf[2]));
    Orientation.normalize();
    xSemaphoreGive(inertialDataMutex);

    int32_t acc[3];
    xSemaphoreTake(i2cMutex, portMAX_DELAY);
    AccGyr.Get_X_Axes(acc);
    xSemaphoreGive(i2cMutex);
    Eigen::Vector3d accelerometer = {static_cast<double>(acc[0]), static_cast<double>(acc[1]), static_cast<double>(acc[2])};
    accelerometer = (accelerometer + accOffset) * gravAccel/1000;
    xSemaphoreTake(inertialDataMutex, portMAX_DELAY);
    Eigen::Vector3d trueAccel(Orientation * accelerometer);
    trueAccel[2] -= gravAccel;
    velocity += trueAccel * elapsedTimeSeconds;
    position += velocity * elapsedTimeSeconds;

    std::vector<float> inertialUpdateData = {
      static_cast<float>(Orientation.w()), static_cast<float>(Orientation.x()), static_cast<float>(Orientation.y()), static_cast<float>(Orientation.z()),
      static_cast<float>(position[0]), static_cast<float>(position[1]), static_cast<float>(position[2]),
      static_cast<float>(velocity[0]), static_cast<float>(velocity[1]), static_cast<float>(velocity[2]),
      static_cast<float>(elapsedTimeSeconds * 1000.0)
    };
    xSemaphoreGive(inertialDataMutex);
    serial_write("inertialUpdate: ", inertialUpdateData);
  }
}

struct ptWithDeviation {
  Eigen::Vector3f vec;
  double deviation;
};

#define MMtoM 0.001

// From [todo]
static std::vector<ptWithDeviation> interpretDistances(VL53L5CX_ResultsData distData) {
  std::vector<ptWithDeviation> localCloud;
  for (int y = 0 ; y <= ToFWidth * (ToFWidth - 1) ; y += ToFWidth) {
    for (int x = ToFWidth - 1 ; x >= 0 ; x--) {
      double distMeters = distData.distance_mm[x + y] * MMtoM;
      if (pdist[x+y] != distMeters && distMeters != 0) {
        double pitch_rad = ST_PITCH_ANGLES_DEG[63-(x+y)] * DEG_TO_RAD;
        double yaw_rad = ST_YAW_ANGLES_DEG[63-(x+y)] * DEG_TO_RAD;
        double st_ray_dir_x = -std::cos(yaw_rad) * std::cos(pitch_rad) / std::sin(pitch_rad);
        double st_ray_dir_y = std::sin(yaw_rad) * std::cos(pitch_rad) / std::sin(pitch_rad);
        ptWithDeviation point;
        point.vec = Eigen::Vector3f({static_cast<float>(st_ray_dir_y * distMeters), static_cast<float>(st_ray_dir_x * distMeters), static_cast<float>(distMeters)});
        point.deviation = distData.range_sigma_mm[x+y] * MMtoM;
        localCloud.push_back(point);
        pdist[x+y] = distMeters;
      }
    }
  }
  return localCloud;
}

static std::array<std::uniform_int_distribution<int>, 3> makeIntDistros(int ptCount) {
  std::uniform_int_distribution<int> distribOne(0, ptCount-1);
  std::uniform_int_distribution<int> distribTwo(0, ptCount-2);
  std::uniform_int_distribution<int> distribThree(0, ptCount-3);
  return {distribOne, distribTwo, distribThree};
}

#include <optional>
static plane runRANSAC(std::array<std::uniform_int_distribution<int>, 3> distributions, std::mt19937 gen, 
                       std::vector<ptWithDeviation>& localCloud, int ptCount) {
  std::vector<Eigen::Vector3f> randPts;
  std::vector<int> rands;
  for (auto& dist : distributions) {
    int rand = dist(gen);
    for (const auto& prevRand : rands) {
      rand += int(rand >= prevRand);
    }
    randPts.push_back(localCloud[rand].vec);
  }
  Eigen::Vector3f unitNormal = (randPts[0]-randPts[1]).cross(randPts[0]-randPts[2]).normalized();

  std::vector<int> planeMatches;
  for (int i = 0; i < ptCount; i++) {
    float normDist = std::abs((localCloud[i].vec-randPts[0]).dot(unitNormal));
    if (normDist < point_plane_max_stdevs * localCloud[i].deviation) {
      planeMatches.push_back(i);
    }
  }
  int matches = planeMatches.size();
  if (matches >= 12) {
    Eigen::Vector3f mean;
    Eigen::Matrix3f covariance;
    for (int i = 0; i < matches; ++i) {
      Eigen::Vector3f diff = localCloud[planeMatches[i]].vec - mean;
      mean += diff / (i + 1);
      covariance += diff * diff.transpose() * i / (i + 1);
    }
    covariance = covariance/(matches-1);
    plane newPlane;
    newPlane.covariances = compressCov(covariance);
    newPlane.mean = mean;
    newPlane.iteration = iteration;
    cloud.pts.push_back({newPlane.mean[0], newPlane.mean[1], newPlane.mean[2]}); // TODO: Convert to Absolute
    std::sort(planeMatches.rbegin(), planeMatches.rend()); // TODO: double check it works
    for (int index : planeMatches) {
      if (index < localCloud.size()) {
        localCloud.erase(localCloud.begin() + index);
      }
    }
    return newPlane;
  }
  plane newPlane;
  newPlane.iteration = -1;
  return newPlane;
}

#define CLOSE_PLANE_COUNT 20
static void constrainByPlane(plane newPlane) {
  if (!cloud.pts.empty()) {
    const float query_pt[3] = {newPlane.mean[0], newPlane.mean[1], newPlane.mean[2]}; // TODO: Convert to Absolute
    unsigned int ret_index[CLOSE_PLANE_COUNT];
    float out_dist_sqr[CLOSE_PLANE_COUNT];
    nanoflann::KNNResultSet<float> resultSet(CLOSE_PLANE_COUNT);
    resultSet.init(ret_index, out_dist_sqr);

    tree_index.findNeighbors(resultSet, query_pt);
    for (const auto& index : ret_index) {
      Eigen::Vector3f currentMean = newPlane.mean;
      Eigen::Matrix3f currentCov = decompressCov(newPlane.covariances);
      Eigen::Vector3f currentNorm = getNorm(currentCov);
      int currentIteration = iteration;
      plane pastPlane = allPlanes[index];
      Eigen::Vector3f pastMean = pastPlane.mean;
      Eigen::Matrix3f pastCov = decompressCov(pastPlane.covariances);
      Eigen::Vector3f pastNorm = getNorm(pastCov);
      int pastIteration = pastPlane.iteration;
      // TODO: Find if these surfaces appear to describe the same plane
      Eigen::Matrix3f alignmentCov;
      // TODO: Write their alignment as a square to minimize
    }
  }
}

static void runMatching() {
  vTaskDelay(watchdog_delay);
  VL53L5CX_ResultsData distData;
  xSemaphoreTake(i2cMutex, portMAX_DELAY);
  if (ToF.isDataReady() && ToF.getRangingData(&distData)) {
    xSemaphoreGive(i2cMutex);
    serial_write("ToF Data read");
  } else {
    xSemaphoreGive(i2cMutex);
    return;
  }
  iteration++;
  std::vector<ptWithDeviation> localCloud = interpretDistances(distData);
  int ptCount = localCloud.size();
  int n = 0;
  int startCloudSize = cloud.kdtree_get_point_count();
  std::random_device randDevice;
  std::mt19937 gen(randDevice());
  std::array<std::uniform_int_distribution<int>, 3> distributions = makeIntDistros(ptCount);
  while (n < 2 * pow(ptCount, 2)) {
    n++;
    plane newPlane = runRANSAC(distributions, gen, localCloud, ptCount);
    if (newPlane.iteration != -1) {
      allPlanes.push_back(newPlane);
      ptCount = localCloud.size();
      distributions = makeIntDistros(ptCount);
      constrainByPlane(newPlane);
    }
  }
  int endCloudSize = cloud.kdtree_get_point_count();
  if (endCloudSize != startCloudSize) {
    tree_index.addPoints(startCloudSize, endCloudSize-1);
  }
}

static void surfaceMatcher(void * pvParameters) {
  for(;;) {
    runMatching();
  }
}

static void aligner(void * pvParameters) {
  for(;;) {
    vTaskDelay(watchdog_delay);
    if (!cloud.pts.empty()) {
      Eigen::MatrixXd A = Eigen::MatrixXd::Zero(64, 6);
      Eigen::VectorXd b = Eigen::VectorXd::Zero(64);
      int n = 0;
      if (A.rows() == 0 || A.cols() == 0 || !A.allFinite() || A.cwiseAbs().maxCoeff() == 0.0) {
        //serial_write("Constraint Solver Edge Case");
      } else {
        A.conservativeResize(n, 6);
        b.conservativeResize(n);
        Eigen::VectorXd x_opt = Eigen::pseudoInverse(A)*b;
      }
    }
  }
}

// INITIALIZATION CODE

TaskHandle_t Core0Task;
TaskHandle_t matchTask;
TaskHandle_t Core1Task;
TaskHandle_t SerialLog;

#define gravMilliGs 1000.0
static void imuCalibrator(void * pvParameters) {
  unsigned long calibratorStartTime = micros();
  int n = 0;
  Eigen::Vector3d accMean = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyroMean = Eigen::Vector3d::Zero();
  Eigen::Vector3d accM2 = Eigen::Vector3d::Zero();
  Eigen::Vector3d gyroM2 = Eigen::Vector3d::Zero();
  while((micros() - calibratorStartTime)/1000000.0 < 8.0) {
    vTaskDelay(watchdog_delay);
    n++;
    int32_t acc[3];
    AccGyr.Get_X_Axes(acc);
    Eigen::Vector3d accelerometer = {static_cast<double>(acc[0]), static_cast<double>(acc[1]), static_cast<double>(acc[2])};
    accelerometer[2] -= gravMilliGs;
    Eigen::Vector3d accDelta = accelerometer - accMean;
    accMean += accDelta / n;
    accM2 += (accelerometer - accMean).cwiseProduct(accDelta);
    int32_t gyro[3];
    AccGyr.Get_G_Axes(gyro);
    Eigen::Vector3d gyroscope = {static_cast<double>(gyro[0]), static_cast<double>(gyro[1]), static_cast<double>(gyro[2])};
    Eigen::Vector3d gyroDelta = gyroscope - gyroMean;
    gyroMean += gyroDelta / n;
    gyroM2 += (gyroscope - gyroMean).cwiseProduct(gyroDelta);
  }
  Eigen::Vector3d accOffset = -accMean;
  Eigen::Vector3d accVari = accM2 / (n-1);
  Eigen::Vector3d gyroOffset = -gyroMean;
  Eigen::Vector3d gyroVari = gyroM2 / (n-1);
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
    aligner,
    "Core0Task",
    32768,
    NULL,
    3,
    &Core0Task,
    0
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
  return imuIntegrator(accOffset, accVari, gyroOffset, gyroVari);
}

void setup() {
  delay(5);
  pinMode(LED_BUILTIN, OUTPUT);

  Serial.begin(115200);
  delay(20);
  while (!Serial) {
    delay(10);
  }
  Serial.write("Setting up...");

  IMU_I2C.begin();
  AccGyr.begin();
  AccGyr.Enable_X();
  AccGyr.Enable_G();
  AccGyr.Set_G_FS(GyroFullScale);
  AccGyr.Set_X_FS(AccFullScale);
  AccGyr.Set_X_ODR(208.0f);
  AccGyr.Set_G_ODR(208.0f);
  delay(20);

  Wire.begin();
  Wire.setClock(IMU_I2C_MAX_FREQ);
  Serial.println("Initializing sensor board. This can take up to 10s. Please wait.");
  if (ToF.begin() == false) {
    Serial.println(F("ToF Sensor not found - check your wiring. Freezing"));
    while (1); 
  }
  ToF.setSharpenerPercent(ToF_Sharpness);
  ToF.setResolution(std::pow(ToFWidth, 2));
  ToF.setRangingFrequency(15);
  ToF.startRanging();
  dump_mem_usage();
  delay(1);
  inertialDataMutex = xSemaphoreCreateMutex();
  i2cMutex = xSemaphoreCreateMutex();
  
  xTaskCreatePinnedToCore(
    imuCalibrator,
    "InitTask",
    8192,
    NULL,
    2,
    &Core1Task,
    0
  );
}

void loop() {}