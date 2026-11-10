#include <Arduino.h>
#include <ArduinoEigen.h>

class plane{
    public:
        plane(Eigen::Matrix3f cov, Eigen::Vector3f meanIn, int i);
        ~plane();
        Eigen::Matrix3f getCov();
        Eigen::Vector3f getMean();
        Eigen::Vector3f getNorm();
        int getIteration();
    private:
        std::array<float, 6> covariances;
        Eigen::Vector3f mean;
        int iteration;
};

Eigen::Vector3f plane::getNorm() {
  Eigen::JacobiSVD<Eigen::Matrix3f> svd(getCov(), Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.matrixU().col(2);
}

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

plane::plane(Eigen::Matrix3f cov, Eigen::Vector3f meanIn, int i) {
    covariances = compressCov(cov);
    mean = meanIn;
    iteration = i;
}

plane::~plane(){}

Eigen::Matrix3f plane::getCov() {
    return decompressCov(covariances);
}

Eigen::Vector3f plane::getMean() {
    return mean;
}

int plane::getIteration() {
    return iteration;
}